Headless-Chrome checks for the review preview (output/review/arm-planner-review.html).

Run from this folder after `npm i puppeteer@23.11.1` (one time):

- `node edgecheck.js` - per page: boxes grown to fit, text still overflowing, boxes sticking out of a chart.
  It also checks `mismatch.html`, a copy whose text is drawn 20% bigger than Mermaid measured (make it by
  adding `.flow foreignObject div, .flow .nodeLabel { font-size: 16.8px !important; }` to the page's style).
- `node linkcheck.js` - clicks a chart box and a text link that name other charts, and reports whether each
  target's collapsed chart opened. Its chart anchors match the current chart titles; update them with the titles.
- `node widths.js` - each chart's drawn size, and the share of full text size it gets in GitHub's 880px column.
