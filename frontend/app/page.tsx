"use client";

import { useRef, useState } from "react";
import Header from "@/components/Header";
import UrlForm from "@/components/UrlForm";
import ExampleChips from "@/components/ExampleChips";
import ResultCard, { type ResultState } from "@/components/ResultCard";
import Footer from "@/components/Footer";
import { PredictError, predictUrl } from "@/lib/api";

export default function Home() {
  const [input, setInput] = useState("");
  const [state, setState] = useState<ResultState>({ status: "empty" });
  const [loading, setLoading] = useState(false);
  const requestId = useRef(0);

  async function handleSubmit() {
    const url = input.trim();
    if (!url) {
      setState({
        status: "validation-error",
        message: "Enter a valid URL, e.g. example.com/login.",
      });
      return;
    }
    const id = ++requestId.current;
    setLoading(true);
    setState({ status: "loading" });
    try {
      const data = await predictUrl(url);
      if (id !== requestId.current) return;
      setState({ status: "result", data, checkedUrl: url });
    } catch (err) {
      if (id !== requestId.current) return;
      if (err instanceof PredictError) {
        if (err.kind === "validation") {
          setState({ status: "validation-error", message: err.message });
        } else if (err.kind === "timeout") {
          setState({ status: "timeout", message: err.message });
        } else {
          setState({ status: "unavailable", message: err.message });
        }
      } else {
        setState({
          status: "unavailable",
          message: "Something went wrong. Try again.",
        });
      }
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <div className="mx-auto w-full max-w-xl px-4 py-8">
        <Header />
        <main>
          <UrlForm
            value={input}
            loading={loading}
            onChange={setInput}
            onSubmit={handleSubmit}
          />
          <ExampleChips onPick={setInput} />
          <ResultCard state={state} />
        </main>
        <Footer />
      </div>
    </div>
  );
}
