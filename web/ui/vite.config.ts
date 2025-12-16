import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // For GitHub Pages, set `VITE_BASE` (e.g., "/<repo>/"). Defaults to "/".
  base: process.env.VITE_BASE?.trim() || "/",
  plugins: [react()],
  server: {
    port: 5173
  }
});
