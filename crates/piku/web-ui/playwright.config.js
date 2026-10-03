import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.PIKU_WEB_URL || "http://127.0.0.1:9090";
const outputDir = process.env.PIKU_WEB_EVAL_OUTPUT
  || "../../../.artifacts/playwright-tests";
const reportDir = process.env.PIKU_WEB_EVAL_REPORT
  || "../../../.artifacts/playwright-report";
const reviewCapture = Boolean(process.env.PIKU_WEB_EVAL_OUTPUT);

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.js",
  outputDir,
  timeout: reviewCapture ? 60_000 : 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: 1,
  reporter: [
    ["list"],
    ["html", { outputFolder: reportDir, open: "never" }],
  ],
  use: {
    ...devices["Desktop Chrome"],
    baseURL,
    headless: true,
    viewport: { width: 1280, height: 720 },
    // A successful managed evaluator run is evidence too. Ordinary local and
    // CI checks stay lean; review runs retain the complete visual record.
    screenshot: reviewCapture ? "on" : "only-on-failure",
    trace: reviewCapture ? "on" : "retain-on-failure",
    video: reviewCapture
      ? { mode: "on", size: { width: 1280, height: 720 } }
      : "retain-on-failure",
  },
});
