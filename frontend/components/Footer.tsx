export default function Footer() {
  return (
    <footer className="mt-8 border-t border-slate-200 pt-4 dark:border-slate-800">
      <h2 id="limitations" className="text-sm font-semibold">
        Limitations — experimental, URL text only.
      </h2>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600 dark:text-slate-400">
        <li>
          Checks URL text only; can&apos;t see page content, hosting, or new
          domains.
        </li>
        <li>No HTTPS signal in v1; http(s):// is stripped before scoring.</li>
        <li>Dataset bias: scores reflect URL structure as much as intent.</li>
        <li>
          Headline metrics overstate reality: v3 gets ~0.79 acc / 0.91 recall
          at thr 0.35, but misses ~77% of bare-phishing hosts and flags
          /about, /login paths even on well-known domains.
        </li>
        <li>
          Allowlist match means the domain is well-known; it does not
          guarantee this specific page is safe.
        </li>
        <li>
          Not a safety guarantee. When in doubt, navigate via a known source.
        </li>
      </ul>
    </footer>
  );
}
