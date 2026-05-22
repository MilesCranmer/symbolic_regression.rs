import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  expect: { timeout: 10_000 },
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "retain-on-failure"
  },
  webServer: {
    // Always build the preview bundle before starting the server so CI and
    // fresh checkouts don't rely on a preexisting dist/ directory.
    command: "npm run build && npm run preview -- --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000
  },
  projects: [
    {
      name: "chromium-threads",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          // Ensure Chromium exposes the primitives needed for the threaded wasm build
          // even in stricter CI sandboxes.
          args: [
            "--js-flags=--experimental-wasm-threads",
            "--enable-blink-features=WebAssemblySIMD,WebAssemblyThreads",
            "--enable-features=SharedArrayBuffer,IsolateOrigins,site-per-process"
          ]
        }
      }
    },
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] }
    }
  ]
});
