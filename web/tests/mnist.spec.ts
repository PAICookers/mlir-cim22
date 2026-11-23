import { expect, test } from "@playwright/test";

test("Training active stages allow cancellation and show actual results", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const streams = new Set<{
      url: string;
      onmessage?: (e: MessageEvent) => void;
    }>();
    window.EventSource = class {
      onopen?: (e: Event) => void;
      onmessage?: (e: MessageEvent) => void;
      constructor(public url: string) {
        streams.add(this);
        queueMicrotask(() => this.onopen?.(new Event("open")));
      }
      close() {
        streams.delete(this);
      }
    } as unknown as typeof EventSource;
    window.addEventListener("test:mnist", (event) => {
      for (const s of streams)
        if (s.url === "/api/mnist/events") {
          s.onmessage?.(
            new MessageEvent("message", {
              data: JSON.stringify((event as CustomEvent).detail),
            }),
          );
        }
    });
  });
  await page.route("**/api/mnist/cancel", (r) => r.fulfill({ json: {} }));
  await page.goto("/#mnist");
  const title = page.getByRole("heading", { name: "图片分类模型配置与训练" });
  await expect(title).toBeVisible();
  await page.getByRole("button", { name: "模型训练", exact: true }).click();
  await expect(page).toHaveURL(/#training$/);
  await expect(title).toBeVisible();
  await expect(page.getByText("本地服务已连接")).toBeVisible();
  const task = {
    id: "test",
    epoch: 1,
    epochs: 3,
    model: "small",
    batch_size: 64,
    learning_rate: 0.002,
    data_dir: "/cache",
  };
  for (const status of ["downloading", "training", "evaluating"]) {
    await page.evaluate(
      (task) =>
        window.dispatchEvent(
          new CustomEvent("test:mnist", { detail: { task } }),
        ),
      { ...task, status },
    );
    await expect(
      page.getByRole("button", { name: "取消", exact: true }),
    ).toBeEnabled();
    await expect(page.getByRole("button", { name: "开始训练" })).toBeDisabled();
  }
  const request = page.waitForRequest("**/api/mnist/cancel");
  await page.getByRole("button", { name: "取消", exact: true }).click();
  expect((await request).method()).toBe("POST");
  await page.evaluate(
    (task) =>
      window.dispatchEvent(new CustomEvent("test:mnist", { detail: { task } })),
    { ...task, status: "failed", error: "无法加载 MNIST 数据集" },
  );
  await expect(page.getByRole("alert")).toHaveText("无法加载 图片分类 数据集");
  await page.evaluate(
    (task) =>
      window.dispatchEvent(new CustomEvent("test:mnist", { detail: { task } })),
    {
      ...task,
      status: "completed",
      test_accuracy: 0.98,
      train_loss: 0.02,
      total_seconds: 12.34,
      last_message: "正在加载 MNIST 测试集",
      log_text: `正在检查并下载 MNIST 数据\n${"epoch=1 loss=0.1234567890 ".repeat(500)}\n训练完成`,
      download_url: "/api/mnist/archive?task_id=test",
    },
  );
  await expect(page.getByText("98.00%")).toBeVisible();
  await expect(page.getByText("正在加载 图片分类 测试集")).toBeVisible();
  await expect(page.locator("body")).not.toContainText("MNIST");
  await expect(page.getByText("总耗时：12.340 s")).toBeVisible();
  await expect(
    page.getByText("实际配置：small · 3 轮 · batch 64 · lr 0.002"),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "下载训练结果", exact: true }),
  ).toHaveCount(1);
  const width = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(width.scroll).toBeLessThanOrEqual(width.viewport);
});
