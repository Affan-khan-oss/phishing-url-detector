export type PredictResponse = {
  label: "safe" | "phishing";
  score: number;
  threshold: number;
  risk_level: "low" | "medium" | "high";
  reasons: string[];
  strong_signals: boolean;
  disclaimer: string;
  source: "model" | "allowlist";
  override: boolean;
};

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class PredictError extends Error {
  kind: "validation" | "unavailable" | "timeout";
  constructor(kind: "validation" | "unavailable" | "timeout", message: string) {
    super(message);
    this.kind = kind;
  }
}

export async function predictUrl(
  url: string,
  opts: { timeoutMs?: number } = {},
): Promise<PredictResponse> {
  const timeoutMs = opts.timeoutMs ?? 10000;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
      signal: controller.signal,
    });
    if (res.status === 422) {
      let detail = "Enter a valid URL, e.g. example.com/login.";
      try {
        const data = await res.json();
        if (typeof data?.detail === "string" && data.detail.trim()) {
          detail = data.detail;
        }
      } catch {
        // keep default
      }
      throw new PredictError("validation", detail);
    }
    if (!res.ok) {
      throw new PredictError(
        "unavailable",
        `Checker returned status ${res.status}. Try again.`,
      );
    }
    return (await res.json()) as PredictResponse;
  } catch (err) {
    if (err instanceof PredictError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new PredictError(
        "timeout",
        "The check took longer than 10 seconds. The backend may be busy — try again.",
      );
    }
    throw new PredictError(
      "unavailable",
      `Can't reach the API at ${API_BASE}. Start the backend (\`uvicorn backend.main:app --port 8000\`) and try again.`,
    );
  } finally {
    clearTimeout(timer);
  }
}
