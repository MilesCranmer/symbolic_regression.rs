import { test, expect } from "@playwright/test";

test("loads real WASM + worker and can run a tiny search", async ({ page }) => {
  page.on("pageerror", (err) => {
    throw err;
  });

  await page.goto("/");

  // Wait for WASM init (options become available and enable parsing).
  await expect(page.getByTestId("parse-preview")).toBeEnabled();

  // Parse the default CSV so plots can use local parsed data too.
  await page.getByRole("button", { name: "Data" }).click();
  await page.getByTestId("parse-preview").click();

  // Shrink search budget so CI stays fast.
  await page.getByRole("button", { name: "Configure" }).click();

  // Expand Advanced hyperparameters section to access fields inside.
  await page.getByText("Advanced hyperparameters").click();

  await page.getByTestId("opt-populations").fill("1");
  await page.getByTestId("opt-population-size").fill("16");
  await page.getByTestId("opt-ncycles").fill("20");

  // Run search (threads explicitly enabled).
  await page.getByRole("button", { name: "Run" }).click();

  // Shrink iterations right before initialize (options apply at init time).
  await page.getByTestId("opt-niterations").fill("1");

  await page.getByTestId("threads-enabled").check();

  await page.getByTestId("search-init").click();

  // Verify the worker actually instantiated a SharedArrayBuffer-backed wasm memory on localhost.
  await expect
    .poll(async () => {
      const s = await page.evaluate(() => (window as any).__sr_thread_status);
      return (
        s &&
        s.crossOriginIsolated === true &&
        s.sharedArrayBufferAvailable === true &&
        s.hasSharedMemory === true &&
        s.bufferType === "SharedArrayBuffer"
      );
    })
    .toBeTruthy();

  await expect(page.getByTestId("search-status")).toHaveText("ready");

  await page.getByTestId("search-start").click();
  await expect(page.getByTestId("search-status")).toHaveText(/done|paused|running/);

  // Wait for at least one solution and click it to trigger evaluation.
  const table = page.getByTestId("solutions-table");
  await expect(table).toBeVisible();

  // If the search is still running, give it a moment to publish the first front_update.
  await page.waitForTimeout(1500);

  const firstRow = table.locator("tbody tr").first();
  await expect(firstRow).toBeVisible();
  await firstRow.click();

  // Expect evaluation to populate selected equation (and remove "no metrics" state).
  await expect(page.getByTestId("selected-equation")).toBeVisible();
  await expect(page.getByTestId("no-metrics")).toHaveCount(0);

  // Regression: single-threaded mode should still initialize and produce outputs.
  await page.getByRole("button", { name: "Reset" }).click();
  await expect(page.getByTestId("search-status")).toHaveText("idle");

  await page.getByTestId("threads-enabled").uncheck();

  await page.getByTestId("opt-niterations").fill("1");
  await page.getByTestId("search-init").click();
  await expect(page.getByTestId("search-status")).toHaveText("ready");

  await page.getByTestId("search-start").click();
  await expect(page.getByTestId("search-status")).toHaveText(/done|paused|running/);

  const table2 = page.getByTestId("solutions-table");
  await expect(table2).toBeVisible();
  await page.waitForTimeout(1500);
  await expect(table2.locator("tbody tr").first()).toBeVisible();
});
