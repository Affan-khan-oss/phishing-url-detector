import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Phishing URL Checker",
  description: "Paste a URL to check lexical phishing risk signals.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col font-sans">{children}</body>
    </html>
  );
}
