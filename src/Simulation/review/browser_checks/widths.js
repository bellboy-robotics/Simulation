// Width and height of every chart as drawn (its SVG viewBox), and how much GitHub's 880px column would shrink it.
const puppeteer = require("puppeteer");
(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto("file:///Users/ronit/Work/repos/Simulation/output/review/arm-planner-review.html", { waitUntil: "networkidle0" });
  await page.waitForFunction(() => window.REVIEW_GROWN !== undefined, { timeout: 60000 });
  const r = await page.evaluate(() => Array.from(document.querySelectorAll(".flow svg")).map((svg, i) => {
    const [, , w, h] = (svg.getAttribute("viewBox") || "0 0 0 0").split(/\s+/).map(Number);
    return `chart ${i}: ${Math.round(w)} x ${Math.round(h)} px, text at ${Math.min(100, Math.round(880 / w * 100))}% in an 880px column`;
  }));
  console.log(r.join("\n"));
  await browser.close();
})();
