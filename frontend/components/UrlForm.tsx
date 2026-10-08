type Props = {
  value: string;
  loading: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
};

export default function UrlForm({ value, loading, onChange, onSubmit }: Props) {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <label
        htmlFor="url-input"
        className="mb-1 block text-sm font-medium"
      >
        URL to check
      </label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          id="url-input"
          type="text"
          inputMode="url"
          autoComplete="off"
          spellCheck={false}
          placeholder="example.com/login"
          value={value}
          disabled={loading}
          onChange={(e) => onChange(e.target.value)}
          className="min-h-[44px] flex-1 rounded-md border border-slate-300 bg-white px-3 py-2 font-mono text-sm text-slate-900 placeholder:text-slate-400 disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100"
        />
        <button
          type="submit"
          disabled={loading}
          className="min-h-[44px] rounded-md bg-slate-900 px-5 py-2 text-sm font-semibold text-white disabled:opacity-60 dark:bg-slate-100 dark:text-slate-900"
        >
          {loading ? "Checking…" : "Check"}
        </button>
      </div>
    </form>
  );
}
