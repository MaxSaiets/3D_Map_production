import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--bg)",
        foreground: "var(--ink)",
        bg: "var(--bg)",
        "bg-2": "var(--bg-2)",
        "bg-3": "var(--bg-3)",
        paper: "var(--paper)",
        "paper-2": "var(--paper-2)",
        ink: "var(--ink)",
        "ink-2": "var(--ink-2)",
        "ink-3": "var(--ink-3)",
        "ink-4": "var(--ink-4)",
        forest: "var(--forest)",
        "forest-2": "var(--forest-2)",
        "forest-3": "var(--forest-3)",
        moss: "var(--moss)",
        bronze: "var(--bronze)",
        "bronze-2": "var(--bronze-2)",
        terracotta: "var(--terracotta)",
        line: "var(--line)",
        "line-2": "var(--line-2)",
        "line-soft": "var(--line-soft)",
      },
      fontFamily: {
        serif: ["Cormorant Garamond", "Georgia", "serif"],
        sans: ["Manrope", "Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      borderRadius: {
        xl: "18px",
        "2xl": "28px",
      },
      boxShadow: {
        soft: "0 12px 32px -16px rgba(27,42,34,.22), 0 2px 6px -2px rgba(27,42,34,.08)",
        lift: "0 30px 60px -30px rgba(27,42,34,.35), 0 8px 20px -10px rgba(27,42,34,.12)",
      },
    },
  },
  plugins: [],
};
export default config;

