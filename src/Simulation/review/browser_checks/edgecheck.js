// Per page: boxes grown, text still overflowing, and boxes or labels sticking out past their chart's edges.
const puppeteer = require("puppeteer");
(async () => {
  const browser = await puppeteer.launch({ headless: true });
  for (const [name, url] of [["normal", "file:///Users/ronit/Work/repos/Simulation/output/review/arm-planner-review.html"],
                             ["mismatch", "file://" + process.cwd() + "/mismatch.html"]]) {
    const page = await browser.newPage();
    await page.setViewport({ width: 1100, height: 900 });
    await page.goto(url, { waitUntil: "networkidle0" });
    await page.waitForFunction(() => window.REVIEW_GROWN !== undefined && document.querySelectorAll(".flow svg").length >= 11, { timeout: 60000 });
    await page.evaluate(() => document.querySelectorAll("details").forEach((d) => (d.open = true)));
    await new Promise((r) => setTimeout(r, 500));
    const r = await page.evaluate(() => {
      const cut = [];
      document.querySelectorAll(".flow svg").forEach((svg, i) => {
        const frame = svg.getBoundingClientRect();
        svg.querySelectorAll("g.node, g.edgeLabel, g.cluster").forEach((g) => {
          const b = g.getBoundingClientRect();
          if (b.width && (b.left < frame.left - 1 || b.right > frame.right + 1 || b.top < frame.top - 1 || b.bottom > frame.bottom + 1)) cut.push(`chart ${i}: ${g.id || g.className.baseVal}`);
        });
      });
      let still = 0;
      document.querySelectorAll(".flow g.node foreignObject").forEach((fo) => {
        const d = fo.querySelector("div"); if (d && d.scrollWidth > Number(fo.getAttribute("width")) + 1) still += 1;
      });
      return { grown: window.REVIEW_GROWN, stillOverflowing: still, outsideChart: cut.length, examples: cut.slice(0, 4) };
    });
    console.log(name, JSON.stringify(r));
    if (name === "mismatch") { const f = await page.$$(".flow"); await f[2].screenshot({ path: "edge-chart2.png" }); await f[3].screenshot({ path: "edge-chart3.png" }); }
    await page.close();
  }
  await browser.close();
})();
