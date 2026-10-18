import { useEffect, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Clock3,
  Cpu,
  FileBox,
  FolderOpen,
  Layers3,
  LoaderCircle,
  Play,
  RotateCcw,
  Square,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";

type Row = {
  index: number;
  sample: string;
  label: number;
  original_prediction: number;
  cim_prediction: number;
  original_correct: boolean;
  cim_correct: boolean;
  original_seconds: number;
  cim_seconds: number;
};
type Summary = { correct: number; accuracy: number | null; seconds: number };
type Task = {
  id: string;
  status: string;
  weights: string;
  dataset: string;
  total: number;
  completed: number;
  compiled: boolean;
  compile_seconds: number | null;
  error: string | null;
  eta_seconds: number | null;
  cancel_requested: boolean;
  classes: string[];
  rows: Row[];
  totals: { original: Summary; cim: Summary };
};
type Snapshot = { task: Task | null };
const names: Record<string, string> = {
  imported: "待编译",
  starting: "启动中",
  validating: "核对输入",
  loading: "载入模型",
  compiling: "编译中",
  ready: "就绪",
  running: "测试中",
  cancelled: "已取消",
  completed: "已完成",
  interrupted: "已中断",
};
const activeStates = new Set([
  "starting",
  "validating",
  "loading",
  "compiling",
  "running",
]);
const seconds = (value: number | null | undefined) =>
  value == null ? "—" : `${value.toFixed(3)} s`;
const duration = (value: number | null | undefined) =>
  value == null
    ? "估算中"
    : value < 60
      ? `${Math.ceil(value)} 秒`
      : `${Math.floor(value / 60)} 分 ${Math.ceil(value % 60)} 秒`;

async function request(path: string, body: object): Promise<void> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json();
    throw new Error(
      typeof data.detail === "string" ? data.detail : "操作失败，请检查输入",
    );
  }
}

export default function App() {
  const [task, setTask] = useState<Task | null>(null);
  const [weights, setWeights] = useState("");
  const [dataset, setDataset] = useState("");
  const [connected, setConnected] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  const [page, setPage] = useState(0);
  const [showImport, setShowImport] = useState(true);

  useEffect(() => {
    let disposed = false;
    let initialized = false;
    fetch("/api/defaults")
      .then((r) => r.json())
      .then((defaults) => {
        if (disposed) return;
        setWeights((value) => value || defaults.weights);
        setDataset((value) => value || defaults.dataset);
      })
      .catch((e) => {
        if (!disposed) setError(String(e.message));
      });
    const events = new EventSource("/api/events");
    events.onopen = () => {
      setConnected(true);
      setError("");
    };
    // Only this ordered stream updates tasks; HTTP replies may arrive late.
    events.onmessage = (event) => {
      if (disposed) return;
      const { task } = JSON.parse(event.data) as Snapshot;
      setTask(task);
      if (!initialized) {
        initialized = true;
        setShowImport(!task);
        if (task) {
          setWeights(task.weights);
          setDataset(task.dataset);
        }
      }
    };
    events.onerror = () => setConnected(false);
    return () => {
      disposed = true;
      events.close();
    };
  }, []);

  async function operate(path: string, body: object) {
    setPending(true);
    setError("");
    try {
      await request(path, body);
      if (path === "/api/import") {
        setShowImport(false);
        setPage(0);
        setFilter("all");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "请求失败");
    } finally {
      setPending(false);
    }
  }

  const active = !!task && activeStates.has(task.status);
  const disabled = active || pending || !connected;
  const finished = task?.completed === task?.total;
  const rows = (task?.rows ?? []).filter(
    (r) =>
      filter === "all" ||
      (filter === "wrong"
        ? !r.original_correct || !r.cim_correct
        : r.original_prediction !== r.cim_prediction),
  );
  const pages = Math.max(1, Math.ceil(rows.length / 12));
  const currentPage = Math.min(page, pages - 1);
  const progress = task ? (task.completed / task.total) * 100 : 0;
  const status = task ? (names[task.status] ?? task.status) : "等待导入";
  const className = (index: number) => `类别 ${task?.classes[index] ?? index}`;

  return (
    <div className="min-h-screen">
      <header className="border-b border-border bg-white">
        <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 py-5 lg:px-10">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-xl bg-primary text-white">
              <Cpu size={23} />
            </div>
            <div>
              <span className="text-lg font-bold tracking-tight">
                CIM22{" "}
                <span className="ml-2 font-normal text-muted-foreground">
                  /
                </span>
              </span>
              <span className="ml-3 text-sm font-medium">模型工作台</span>
            </div>
          </div>
          <span className="flex items-center gap-2 text-xs text-muted-foreground">
            <span
              className={`size-2 rounded-full ${connected ? "bg-emerald-500" : "bg-amber-500"}`}
            />
            {connected ? "本地服务已连接" : "连接中断 · 自动重连"}
          </span>
        </div>
      </header>
      <main className="mx-auto max-w-[1400px] px-6 py-9 lg:px-10">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="eyebrow mb-2">
              DVS GESTURE · INFERENCE COMPARISON
            </div>
            <h1 className="text-3xl font-semibold tracking-tight">
              让模型结果，一目了然。
            </h1>
            <p className="mt-3 text-sm text-muted-foreground">
              导入固定 DVS 模型，对比原始模型与 CIM 的识别结果和推理耗时。
            </p>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-blue-100 bg-blue-50 px-4 py-2 text-xs font-medium text-primary">
            <Cpu size={14} />
            软件 runner<span className="mx-1 text-blue-200">|</span>本地单任务
          </div>
        </div>
        {(!connected || error || task?.error) && (
          <div
            role="alert"
            className="mb-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"
          >
            {error ||
              (!connected
                ? "与服务的连接已断开，正在重连。重连后会恢复任务状态。"
                : task?.error)}
          </div>
        )}
        <div className="grid gap-6 lg:grid-cols-[1.15fr_1fr]">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <h2 className="flex items-center gap-2.5 text-base font-semibold">
                  <FileBox size={19} className="text-primary" />
                  模型与编译
                </h2>
                <span className="eyebrow">01 / PREPARE</span>
              </div>
            </CardHeader>
            <CardContent>
              <div className="mb-5 flex items-center gap-4 rounded-xl bg-muted/70 p-4">
                <div className="grid size-12 shrink-0 place-items-center rounded-xl bg-white text-primary">
                  <Layers3 size={25} />
                </div>
                <div>
                  <p className="text-sm font-semibold">
                    DVSGesture · MobileNet SNN
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    C16 · T32 · 11 类 · 固定模型
                  </p>
                </div>
              </div>
              {showImport ? (
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    void operate("/api/import", { weights, dataset });
                  }}
                  className="space-y-4"
                >
                  <div>
                    <label
                      htmlFor="weights"
                      className="mb-2 block text-xs font-medium"
                    >
                      模型权重
                    </label>
                    <Input
                      id="weights"
                      value={weights}
                      onChange={(e) => setWeights(e.target.value)}
                      placeholder="/path/to/weights.pt"
                      required
                      disabled={disabled}
                    />
                  </div>
                  <div>
                    <label
                      htmlFor="dataset"
                      className="mb-2 block text-xs font-medium"
                    >
                      测试集目录
                    </label>
                    <Input
                      id="dataset"
                      value={dataset}
                      onChange={(e) => setDataset(e.target.value)}
                      placeholder="/path/to/data/test"
                      required
                      disabled={disabled}
                    />
                    <p className="mt-2 text-xs text-muted-foreground">
                      填写 Linux 本地路径，读取类别目录下的 .npz 缓存帧。
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      type="submit"
                      disabled={disabled || !weights || !dataset}
                    >
                      {pending ? (
                        <LoaderCircle className="animate-spin" />
                      ) : (
                        <FolderOpen />
                      )}
                      导入模型与测试集
                    </Button>
                    {task && (
                      <Button
                        type="button"
                        variant="ghost"
                        onClick={() => setShowImport(false)}
                      >
                        返回
                      </Button>
                    )}
                  </div>
                </form>
              ) : (
                <div className="space-y-3 text-xs">
                  <p className="break-all text-muted-foreground">
                    <span className="mr-2 font-medium text-foreground">
                      权重
                    </span>
                    {task?.weights}
                  </p>
                  <p className="break-all text-muted-foreground">
                    <span className="mr-2 font-medium text-foreground">
                      测试集
                    </span>
                    {task?.dataset}
                  </p>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="-ml-3"
                    disabled={disabled}
                    onClick={() => setShowImport(true)}
                  >
                    <FolderOpen />
                    更换输入 / 新建测试
                  </Button>
                </div>
              )}
              <div className="mt-5 flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4 text-xs">
                <span className="text-muted-foreground">
                  固定输入{" "}
                  <code className="ml-2 rounded bg-muted px-2 py-1 text-foreground">
                    32 × 1 × 2 × 128 × 128
                  </code>
                </span>
                <span className="text-muted-foreground">T · B · C · H · W</span>
              </div>
              <div className="mt-5 flex items-center justify-between gap-3">
                <Button
                  variant="outline"
                  disabled={disabled || !task || task.compiled}
                  onClick={() =>
                    void operate("/api/action", { action: "compile" })
                  }
                >
                  {active && task?.status !== "running" ? (
                    <LoaderCircle className="animate-spin" />
                  ) : (
                    <Cpu />
                  )}
                  {task?.compiled ? "编译已就绪" : "编译模型"}
                </Button>
                <div className="text-right">
                  <p className="text-xs text-muted-foreground">
                    编译与首次执行
                  </p>
                  <p className="number mt-1 text-sm font-medium">
                    {seconds(task?.compile_seconds)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          <Card className="flex flex-col">
            <CardHeader>
              <div className="flex items-center justify-between">
                <h2 className="flex items-center gap-2.5 text-base font-semibold">
                  <Activity size={19} className="text-primary" />
                  运行控制
                </h2>
                <span className="eyebrow">02 / RUN</span>
              </div>
            </CardHeader>
            <CardContent className="flex flex-1 flex-col">
              <div className="flex items-center justify-between">
                <span
                  className={`inline-flex items-center gap-2 rounded-md px-2.5 py-1.5 text-xs font-medium ${task?.status === "interrupted" ? "bg-amber-50 text-amber-800" : "bg-blue-50 text-primary"}`}
                >
                  {active ? (
                    <LoaderCircle size={13} className="animate-spin" />
                  ) : (
                    <span className="size-1.5 rounded-full bg-current" />
                  )}
                  {task?.cancel_requested
                    ? "正在停止 · 当前样本保存后结束"
                    : status}
                </span>
                <span className="text-xs text-muted-foreground">
                  {task?.total ?? "—"} 个测试样本
                </span>
              </div>
              <div className="mb-3 mt-8 flex items-end justify-between">
                <p className="number">
                  <span className="text-4xl font-semibold tracking-tight">
                    {task?.completed ?? 0}
                  </span>
                  <span className="ml-2 text-sm text-muted-foreground">
                    / {task?.total ?? "—"} 已完成
                  </span>
                </p>
                <span className="number text-sm font-medium text-primary">
                  {progress.toFixed(0)}%
                </span>
              </div>
              <Progress value={progress} aria-label="测试进度" />
              <div className="mt-3 flex items-center gap-1.5 text-xs text-muted-foreground">
                <Clock3 size={13} />
                {task?.status === "running"
                  ? `预计剩余 ${duration(task.eta_seconds)}`
                  : task?.status === "completed"
                    ? "全部样本已保存"
                    : "每完成一个样本，自动保存对比结果"}
              </div>
              <div className="mt-7 flex flex-wrap gap-2">
                <Button
                  disabled={
                    disabled ||
                    !task ||
                    finished ||
                    (!task.compiled &&
                      !["cancelled", "interrupted", "ready"].includes(
                        task.status,
                      ))
                  }
                  onClick={() =>
                    void operate("/api/action", { action: "full" })
                  }
                >
                  {task?.completed ? <RotateCcw /> : <Play />}
                  {task?.completed || task?.status === "interrupted"
                    ? "继续剩余测试"
                    : "开始完整测试"}
                </Button>
                <Button
                  variant="outline"
                  disabled={disabled || !task?.compiled || finished}
                  onClick={() =>
                    void operate("/api/action", { action: "single" })
                  }
                >
                  运行一个样本
                  <ArrowRight />
                </Button>
                <Button
                  variant="destructive"
                  disabled={
                    !active || pending || !connected || task?.cancel_requested
                  }
                  onClick={() => void operate("/api/cancel", {})}
                >
                  <Square />
                  取消
                </Button>
              </div>
              <div className="mt-auto pt-7">
                <div className="flex items-start gap-2 rounded-lg bg-muted/60 p-3 text-xs leading-6 text-muted-foreground">
                  <CircleHelp size={15} className="mt-1 shrink-0" />
                  <p>
                    刷新页面不会停止计算。取消后可继续；服务重启后，续跑会重新编译。编译耗时单独统计。
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="mb-5 mt-9 flex items-center justify-between">
          <h2 className="text-lg font-semibold">
            结果对比{" "}
            <span className="ml-3 text-xs font-normal text-muted-foreground">
              基于 {task?.completed ?? 0} 个已完成样本
            </span>
          </h2>
          {!!task?.completed && (
            <Button asChild variant="outline" size="sm">
              <a href="/api/results.csv" download>
                <ArrowDownToLine />
                导出 CSV
              </a>
            </Button>
          )}
        </div>
        <div className="mb-6 grid gap-5 sm:grid-cols-2">
          {(["original", "cim"] as const).map((side) => {
            const summary = task?.totals[side];
            return (
              <Card
                key={side}
                className={side === "cim" ? "border-blue-200" : ""}
              >
                <CardContent className="pt-5">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold">
                      {side === "original" ? "原始模型" : "CIM 模型"}
                    </h3>
                    <span className="rounded-md bg-muted px-2 py-1 text-[11px] text-muted-foreground">
                      {side === "original" ? "PyTorch · CPU" : "软件 runner"}
                    </span>
                  </div>
                  <div className="mt-5 grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs text-muted-foreground">总准确率</p>
                      <p
                        className={`number mt-2 text-3xl font-semibold tracking-tight ${side === "cim" ? "text-primary" : ""}`}
                      >
                        {summary?.accuracy == null
                          ? "—"
                          : `${(summary.accuracy * 100).toFixed(2)}%`}
                      </p>
                      <p className="mt-2 text-xs text-muted-foreground">
                        {summary?.correct ?? 0} / {task?.completed ?? 0}{" "}
                        识别正确
                      </p>
                    </div>
                    <div className="border-l border-border pl-5">
                      <p className="text-xs text-muted-foreground">
                        推理总耗时
                      </p>
                      <p className="number mt-2 text-2xl font-semibold tracking-tight">
                        {task?.completed ? seconds(summary?.seconds) : "—"}
                      </p>
                      <p className="mt-2 text-xs text-muted-foreground">
                        各样本模型调用耗时之和
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
        <Card className="overflow-hidden">
          <CardHeader className="flex-row flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold">逐样本结果</h3>
              <p className="mt-1 text-xs text-muted-foreground">
                真实类别、预测结果及耗时；不包含数据读取和文件保存。
              </p>
            </div>
            <label className="text-xs text-muted-foreground">
              显示{" "}
              <select
                aria-label="筛选样本"
                value={filter}
                onChange={(e) => {
                  setFilter(e.target.value);
                  setPage(0);
                }}
                className="ml-2 rounded-lg border border-input bg-white px-3 py-2 text-foreground"
              >
                <option value="all">全部样本</option>
                <option value="wrong">包含识别错误</option>
                <option value="different">两边预测不同</option>
              </select>
            </label>
          </CardHeader>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-muted/70 text-muted-foreground">
                <tr>
                  <th>样本</th>
                  <th>真实类别</th>
                  <th>原始模型预测</th>
                  <th>耗时</th>
                  <th>CIM 预测</th>
                  <th>耗时</th>
                </tr>
              </thead>
              <tbody>
                {rows
                  .slice(currentPage * 12, currentPage * 12 + 12)
                  .map((row) => (
                    <tr key={row.index}>
                      <td>
                        <span className="number mr-3 text-muted-foreground">
                          {String(row.index + 1).padStart(3, "0")}
                        </span>
                        <span title={row.sample}>{row.sample}</span>
                      </td>
                      <td className="whitespace-nowrap">
                        {className(row.label)}
                      </td>
                      <td>
                        <Verdict
                          correct={row.original_correct}
                          label={className(row.original_prediction)}
                        />
                      </td>
                      <td className="number whitespace-nowrap">
                        {seconds(row.original_seconds)}
                      </td>
                      <td>
                        <Verdict
                          correct={row.cim_correct}
                          label={className(row.cim_prediction)}
                        />
                      </td>
                      <td className="number whitespace-nowrap">
                        {seconds(row.cim_seconds)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          {!rows.length && (
            <div className="grid place-items-center px-4 py-14 text-center">
              <div className="mb-3 grid size-12 place-items-center rounded-full bg-muted text-muted-foreground">
                <Layers3 size={23} />
              </div>
              <p className="text-sm font-medium">
                {task?.completed
                  ? "没有符合筛选条件的样本"
                  : "等待第一个样本结果"}
              </p>
              <p className="mt-2 text-xs text-muted-foreground">
                {task?.completed
                  ? "尝试切换为全部样本。"
                  : "完成编译后，运行一个样本即可查看两边的识别结果。"}
              </p>
            </div>
          )}
          <div className="flex items-center justify-between border-t border-border px-6 py-3 text-xs text-muted-foreground">
            <span>共 {rows.length} 条 · 已自动保存至本地</span>
            <div className="flex items-center gap-3">
              <Button
                variant="ghost"
                size="icon"
                aria-label="上一页"
                disabled={currentPage === 0}
                onClick={() => setPage(currentPage - 1)}
              >
                <ChevronLeft />
              </Button>
              <span className="number">
                {currentPage + 1} / {pages}
              </span>
              <Button
                variant="ghost"
                size="icon"
                aria-label="下一页"
                disabled={currentPage + 1 >= pages}
                onClick={() => setPage(currentPage + 1)}
              >
                <ChevronRight />
              </Button>
            </div>
          </div>
        </Card>
        <footer className="mt-6 flex flex-wrap justify-between gap-2 text-[11px] text-muted-foreground">
          <span>CIM22 · 本地模型工作台</span>
          <span>固定输入 · 每次推理前复位神经元 · 软件运行</span>
        </footer>
      </main>
    </div>
  );
}

function Verdict({ correct, label }: { correct: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-2 whitespace-nowrap ${correct ? "text-emerald-700" : "text-rose-600"}`}
    >
      {correct ? <Check size={14} /> : <X size={14} />}
      {label}
      <span className="text-[10px]">{correct ? "正确" : "错误"}</span>
    </span>
  );
}
