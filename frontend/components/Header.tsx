export default function Header() {
  return (
    <header className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight">
        Phishing URL Checker
      </h1>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-400">
        Paste a URL to check lexical risk signals. Experimental not a
        safety guarantee.
      </p>
    </header>
  );
}
