export type Example = {
  url: string;
  caption: string;
};

export const EXAMPLES: Example[] = [
  {
    url: "google.com",
    caption: "Known safe site - allowlist, low risk.",
  },
  {
    url: "github.com/about",
    caption:
      "Known safe site - allowlist (model scored 0.75, overridden). Shows the short-path edge case.",
  },
  {
    url: "example.com/about/team/contact-us",
    caption: "Model safe - low risk, longer path.",
  },
  {
    url: "http://paypal-login-secure-update.tk/signin",
    caption: "Likely phishing - high risk (urgent lure + cheap TLD).",
  },
];

type Props = {
  onPick: (url: string) => void;
};

export default function ExampleChips({ onPick }: Props) {
  return (
    <div className="mt-4">
      <p className="text-sm font-medium">Try an example:</p>
      <ul className="mt-2 space-y-2">
        {EXAMPLES.map((ex) => (
          <li
            key={ex.url}
            className="rounded-md border border-slate-200 bg-white p-2 dark:border-slate-800 dark:bg-slate-900"
          >
            <button
              type="button"
              onClick={() => onPick(ex.url)}
              className="min-h-[44px] w-full break-all rounded px-2 py-1 text-left font-mono text-sm underline decoration-slate-400 underline-offset-2"
            >
              {ex.url}
            </button>
            <p className="px-2 pb-1 text-sm text-slate-600 dark:text-slate-400">
              {ex.caption}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}
