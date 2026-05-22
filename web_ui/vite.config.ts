import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // For GitHub Pages, set `VITE_BASE` (e.g., "/<repo>/"). Defaults to "/".
  base: process.env.VITE_BASE?.trim() || "/",
  plugins: [react()],
  worker: {
    format: "es"
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp"
    },
    port: 5173
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp"
    }
  }
});
