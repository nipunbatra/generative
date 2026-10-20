// one-off: find slides where the body's content runs into the title or the take-away line (print mode, final builds)
import puppeteer from 'puppeteer';
const PORT = process.env.PORT || 8771;   // the review server's port (vi.html review: 8774)
const deck = process.argv[2] || 'index.html';
const b = await puppeteer.launch({headless:true, executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
const p = await b.newPage(); await p.setViewport({width:1280,height:720});
await p.goto(`http://127.0.0.1:${PORT}/${deck}?print`, {waitUntil:'load'});
await p.waitForFunction(() => window.deckReady);
await new Promise(r => setTimeout(r, 1500));
const res = await p.evaluate(() => {
  const out = [];
  const vis = el => { const cs = getComputedStyle(el); return cs.display !== 'none' && cs.visibility !== 'hidden' && +cs.opacity > 0.05; };
  for (const s of document.querySelectorAll('section.slide')) {
    const body = s.querySelector(':scope > .body'); if (!body) continue;
    const h2 = s.querySelector(':scope > h2'); const qs = [...s.querySelectorAll(':scope > .q, :scope > p.src')].filter(vis);
    let top = 1e9, bot = -1e9;
    for (const el of body.querySelectorAll('*')) {
      if (el.closest('aside.notes') || el.closest('.katex-mathml') || el.ownerSVGElement || el.classList.contains('fo')) continue;
      if (el.closest('.katex') && !el.classList.contains('katex')) continue;
      if (!vis(el)) continue;
      const e = el.getBoundingClientRect(); if (!e.width || !e.height) continue;
      top = Math.min(top, e.top); bot = Math.max(bot, e.bottom);
    }
    if (top === 1e9) continue;
    const msgs = [];
    if (h2) { const r = h2.getBoundingClientRect(); if (top < r.bottom - 2) msgs.push(`hits title by ${Math.round(r.bottom - top)}px`); }
    for (const q of qs) { const r = q.getBoundingClientRect(); if (bot > r.top + 2) msgs.push(`hits take-away by ${Math.round(bot - r.top)}px`); }
    if (msgs.length) out.push(`${s.id}: ${msgs.join(', ')}`);
  }
  return out;
});
console.log(res.join('\n') || 'no overlaps');
await b.close();
