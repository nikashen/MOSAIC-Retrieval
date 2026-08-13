"use strict";

let chromium;
try { ({ chromium } = require("playwright")); }
catch (error) { ({ chromium } = require("playwright-core")); }

const baseUrl = process.env.PAGES_URL || "http://127.0.0.1:8764/";
const viewports = [
  { name: "desktop", width: 1440, height: 1000 },
  { name: "mobile", width: 390, height: 844 },
];

(async () => {
  const browser = await chromium.launch({ headless: true });
  const results = [];
  for (const viewport of viewports) {
    const page = await browser.newPage({ viewport });
    const errors = [];
    page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
    page.on("pageerror", error => errors.push(error.message));
    await page.goto(baseUrl, { waitUntil: "networkidle" });
    const layout = await page.evaluate(() => {
      const width = document.documentElement.clientWidth;
      const overflowing = [...document.querySelectorAll("body *")].filter(element => {
        const rect = element.getBoundingClientRect();
        return rect.left < -1 || rect.right > width + 1;
      }).map(element => element.tagName + "." + String(element.className).slice(0, 40)).slice(0, 20);
      return { width, bodyScrollWidth: document.body.scrollWidth, overflowing, title: document.title };
    });
    await page.getByRole("tab", { name: "MSR-VTT · Video/Text" }).click();
    const delta = await page.locator("#metric-delta").textContent();
    await page.locator("#query").selectOption("city");
    const firstResult = await page.locator(".synthetic-card p").first().textContent();
    await page.getByRole("button", { name: "Text only" }).click();
    const textGate = await page.locator("#text-gate").textContent();
    results.push({ viewport, errors, layout, delta, firstResult, textGate });
    await page.close();
  }
  await browser.close();
  console.log(JSON.stringify(results, null, 2));
  const failed = results.some(result => result.errors.length || result.layout.bodyScrollWidth > result.layout.width || result.layout.overflowing.length || result.delta !== "+7.90" || result.firstResult !== "Neon street flow" || result.textGate !== "1.00");
  process.exitCode = failed ? 1 : 0;
})().catch(error => { console.error(error); process.exitCode = 1; });
