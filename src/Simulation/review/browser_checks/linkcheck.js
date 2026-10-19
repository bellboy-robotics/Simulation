// Opens chart 3, clicks its box that links to chart 7, and a "Goes on in" link to chart 9; reports whether each opened.
const puppeteer = require("puppeteer");
(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto("file:///Users/ronit/Work/repos/Simulation/output/review/arm-planner-review.html", { waitUntil: "networkidle0" });
  await page.waitForFunction(() => document.querySelectorAll(".flow svg").length >= 11, { timeout: 60000 });
  const isOpen = (id) => page.evaluate((id) => {
    for (let n = document.getElementById(id).nextElementSibling; n && n.tagName !== "H2"; n = n.nextElementSibling)
      if (n.tagName === "DETAILS") return n.open;
    return null;
  }, id);
  const open = (id) => page.evaluate((id) => {
    for (let n = document.getElementById(id).nextElementSibling; n; n = n.nextElementSibling) if (n.tagName === "DETAILS") { n.open = true; break; }
  }, id);
  const targets = await page.evaluate(() => Array.from(new Set(Array.from(document.querySelectorAll(".flow svg a")).map((a) => a.getAttribute("xlink:href") || a.getAttribute("href")))));
  console.log("chart links:", targets.join(" "));
  await open("chart-3--single-moves");
  console.log("chart 7 open before:", await isOpen("chart-7--last-safety-check"));
  const box = await page.$('.flow svg a[*|href="#chart-7--last-safety-check"]');
  await box.click(); await new Promise((r) => setTimeout(r, 800));
  console.log("chart 7 open after clicking the box in chart 3:", await isOpen("chart-7--last-safety-check"));
  const link = await page.$('a[href="#chart-9--obstacle-costs-inside-the-solvers"]:not(svg a)');
  await link.click(); await new Promise((r) => setTimeout(r, 800));
  console.log("chart 9 open after clicking a text link:", await isOpen("chart-9--obstacle-costs-inside-the-solvers"));
  await browser.close();
})();
