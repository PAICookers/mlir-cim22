import { expect, test, type Route } from "@playwright/test";

const ready = {
  id: "fixture",
  status: "ready",
  weights: "/model/weights.pt",
  dataset: "/model/test",
  total: 1,
  completed: 0,
  compiled: true,
  compile_seconds: 1,
  error: null,
  eta_seconds: null,
  cancel_requested: false,
  classes: ["0"],
  rows: [],
  totals: {
    original: { correct: 0, accuracy: null, seconds: 0 },
    cim: { correct: 0, accuracy: null, seconds: 0 },
  },
};

test("a delayed action response cannot replace a newer SSE state", async ({
  page,
}) => {
  // Control SSE delivery independently of HTTP response order, without a model.
  await page.addInitScript(() => {
    const streams = new Set<{
      onmessage?: (event: MessageEvent) => void;
    }>();
    window.EventSource = class {
      onopen?: (event: Event) => void;
      onmessage?: (event: MessageEvent) => void;
      constructor() {
        streams.add(this);
        queueMicrotask(() => this.onopen?.(new Event("open")));
      }
      close() {
        streams.delete(this);
      }
    } as unknown as typeof EventSource;
    window.addEventListener("test:snapshot", (event) => {
      const data = JSON.stringify((event as CustomEvent).detail);
      for (const stream of streams) {
        stream.onmessage?.(new MessageEvent("message", { data }));
      }
    });
  });
  await page.route("**/api/defaults", (route) =>
    route.fulfill({ json: { weights: ready.weights, dataset: ready.dataset } }),
  );
  await page.route("**/api/state", (route) =>
    route.fulfill({ json: { task: ready } }),
  );
  let action: Route | undefined;
  await page.route("**/api/action", (route) => {
    action = route;
  });
  await page.goto("/");
  await expect(page.getByText("本地服务已连接")).toBeVisible();
  await page.evaluate((task) => {
    window.dispatchEvent(new CustomEvent("test:snapshot", { detail: { task } }));
  }, ready);
  await page.getByRole("button", { name: "开始完整测试" }).click();
  await expect.poll(() => !!action).toBe(true);
  await page.evaluate(
    (task) => {
      window.dispatchEvent(
        new CustomEvent("test:snapshot", { detail: { task } }),
      );
    },
    { ...ready, status: "completed", completed: 1 },
  );
  await expect(page.getByText("已完成", { exact: true })).toBeVisible();
  await action!.fulfill({ json: { task: { ...ready, status: "running" } } });
  // Enabled means the delayed request has finished and React rendered its state.
  await expect(
    page.getByRole("button", { name: "更换输入 / 新建测试" }),
  ).toBeEnabled();
  await expect(page.getByText("已完成", { exact: true })).toBeVisible();
});
