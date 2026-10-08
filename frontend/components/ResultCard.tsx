import type { PredictResponse } from "@/lib/api";
import { API_BASE } from "@/lib/api";

export type ResultState =
  | { status: "empty" }
  | { status: "loading" }
  | { status: "result"; data: PredictResponse; checkedUrl: string }
  | { status: "validation-error"; message: string }
  | { status: "unavailable"; message: string }
  | { status: "timeout"; message: string };

function ScoreBar({ score, threshold }: { score: number; threshold: number }) {
  const pct = Math.max(0, Math.min(1, score)) * 100;
  const thrPct = Math.max(0, Math.min(1, threshold)) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-sm font-medium">Risk score</p>
        <p className="font-mono text-sm">
          {score.toFixed(2)} / 1.00 · threshold {threshold.toFixed(2)}
        </p>
      </div>
      <div
        role="progressbar"
        aria-valuenow={Number(score.toFixed(3))}
        aria-valuemin={0}
        aria-valuemax={1}
        aria-label="Risk score"
        className="relative mt-1 h-3 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800"
      >
        <div
          className="h-full rounded-full bg-slate-700 dark:bg-slate-300"
          style={{ width: `${pct}%` }}
        />
        <div
          aria-hidden="true"
          className="absolute inset-y-0 w-0.5 bg-slate-950 dark:bg-white"
          style={{ left: `${thrPct}%` }}
        />
      </div>
    </div>
  );
}

function badgeClass(tone: "safe" | "medium" | "high" | "info" | "neutral") {
  switch (tone) {
    case "safe":
      return "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-200";
    case "medium":
      return "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200";
    case "high":
      return "border-rose-300 bg-rose-50 text-rose-900 dark:border-rose-800 dark:bg-rose-950 dark:text-rose-200";
    case "info":
      return "border-sky-300 bg-sky-50 text-sky-900 dark:border-sky-800 dark:bg-sky-950 dark:text-sky-200";
    case "neutral":
      return "border-slate-300 bg-slate-100 text-slate-800 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200";
  }
}

export default function ResultCard({ state }: { state: ResultState }) {
  if (state.status === "empty") {
    return (
      <section
        aria-live="polite"
        className="mt-6 rounded-md border border-slate-200 p-4 dark:border-slate-800"
      >
        <p className="text-sm text-slate-600 dark:text-slate-400">
          No check yet. Enter a URL above and press Check.
        </p>
      </section>
    );
  }

  if (state.status === "loading") {
    return (
      <section
        aria-live="polite"
        aria-busy="true"
        className="mt-6 rounded-md border border-slate-200 p-4 dark:border-slate-800"
      >
        <p className="text-sm">Checking URL…</p>
        <div className="mt-2 h-3 animate-pulse rounded-full bg-slate-200 dark:bg-slate-800" />
      </section>
    );
  }

  if (
    state.status === "validation-error" ||
    state.status === "unavailable" ||
    state.status === "timeout"
  ) {
    const title =
      state.status === "validation-error"
        ? "ⓘ Can't check that URL"
        : state.status === "timeout"
          ? "ⓘ Request timed out"
          : "ⓘ Checker unavailable";
    const body =
      state.status === "unavailable" && !state.message
        ? `Can't reach the API at ${API_BASE}. Start the backend (\`uvicorn backend.main:app --port 8000\`) and try again.`
        : state.message;
    return (
      <section
        aria-live="polite"
        className="mt-6 rounded-md border border-slate-200 p-4 dark:border-slate-800"
      >
        <p
          className={`inline-block rounded-full border px-3 py-1 text-sm font-semibold ${badgeClass("neutral")}`}
        >
          {title}
        </p>
        <p className="mt-2 text-sm">{body}</p>
      </section>
    );
  }

  const { data, checkedUrl } = state;
  const isAllowlist = data.source === "allowlist";
  const borderline = Math.abs(data.score - data.threshold) <= 0.05;
  const safeWithSignals = data.label === "safe" && data.reasons.length > 0;

  let tone: "safe" | "medium" | "high" | "info";
  let heading: string;
  let sub: string;
  if (isAllowlist) {
    tone = "info";
    heading = "ℹ Known safe site (allowlist)";
    sub =
      "This domain is well-known, so the verdict comes from the allowlist.";
  } else if (data.risk_level === "low") {
    tone = "safe";
    heading = "✓ Looks safe";
    sub =
      "No strong phishing signals found. Still verify the sender before entering credentials.";
  } else if (
    data.label === "phishing" &&
    data.risk_level === "high" &&
    data.strong_signals === true
  ) {
    tone = "high";
    heading = "⚠ Likely phishing";
    sub =
      "This URL has patterns strongly associated with phishing. Don't enter credentials. Verify via a known source.";
  } else if (data.strong_signals === true) {
    tone = "medium";
    heading = "⚠ Suspicious: some phishing-like patterns";
    sub =
      "This URL has patterns often seen in phishing. Don't enter credentials. Verify via a known source.";
  } else {
    tone = "medium";
    heading =
      "⚠ Suspicious: flagged by the overall pattern, no single strong signal";
    sub =
      "The model flagged the overall pattern but found no single strong signal. Don't enter credentials. Verify via a known source.";
  }

  return (
    <section
      aria-live="polite"
      className="mt-6 rounded-md border border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900"
    >
      <h2
        tabIndex={-1}
        className={`inline-block rounded-full border px-3 py-1 text-sm font-semibold ${badgeClass(tone)}`}
      >
        {heading} — {data.risk_level} risk
      </h2>
      <p className="mt-2 text-sm">{sub}</p>

      <div className="mt-4">
        <ScoreBar score={data.score} threshold={data.threshold} />
      </div>
      {borderline && (
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400">
          Borderline: score is close to the threshold.
        </p>
      )}

      <h3 className="mt-4 text-sm font-semibold">
        {safeWithSignals
          ? "Signals found in the URL (not enough to flag)"
          : "Signals found in the URL"}
      </h3>
      {data.reasons.length === 0 ? (
        <p className="mt-1 text-sm">No specific signals returned.</p>
      ) : (
        <ul className="mt-1 list-disc space-y-1 pl-5 text-sm">
          {data.reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      )}

      <p className="mt-3 text-sm">
        Source: {data.source}
        {data.override ? " — model score was overridden." : ""}
      </p>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
        {data.disclaimer}
      </p>
      <p className="mt-2 break-all font-mono text-sm text-slate-600 dark:text-slate-400">
        {checkedUrl}
      </p>
    </section>
  );
}
