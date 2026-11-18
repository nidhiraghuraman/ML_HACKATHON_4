/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#060913", 900: "#0a0f1e", 850: "#0d1426", 800: "#111a30", 700: "#1a2440", 600: "#26324f" },
        cyan: { 300: "#67e8f9", 400: "#22d3ee", 500: "#06b6d4" },
      },
      fontFamily: { sans: ["Inter", "system-ui", "Segoe UI", "Ubuntu", "sans-serif"], mono: ["JetBrains Mono", "ui-monospace", "monospace"] },
      boxShadow: { glow: "0 0 0 1px rgba(34,211,238,.25), 0 8px 30px -8px rgba(34,211,238,.25)" },
      keyframes: { fadein: { from: { opacity: 0, transform: "translateY(6px)" }, to: { opacity: 1, transform: "none" } } },
      animation: { fadein: "fadein .35s ease-out both" },
    },
  },
  plugins: [],
};
