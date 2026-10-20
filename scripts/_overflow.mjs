// one-off: list slides whose content spills outside the 1280x720 frame or overlaps (print mode, final builds)
import puppeteer from 'puppeteer';
const PORT = process.env.PORT || 8771;   // the review server's port (vi.html review: 8774)
const deck = process.argv[2] || 'index.html';
const b = await puppeteer.launch({headless:true, executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
const p = await b.newPage(); await p.setViewport({width:1280,height:720});
const errs = []; p.on('pageerror', e => errs.push(String(e)));
await p.goto(`http://127.0.0.1:${PORT}/${deck}?print`, {waitUntil:'load'});
await p.waitForFunction(() => window.deckReady);
await new Promise(r => setTimeout(r, 1500));
const res = await p.evaluate(() => {
  const out = [];
  for (const s of document.querySelectorAll('section.slide')) {
    const r = s.getBoundingClientRect();
    let worst = null;
    for (const el of s.querySelectorAll('*')) {
      if (el.closest('aside.notes') || el.closest('.katex-html') && el.closest('.katex-display') === null && false) continue;
      if (el.closest('aside.notes') || el.closest('.katex-mathml')) continue;
      if (el.ownerSVGElement) continue;                       // only the <svg> root counts
      if (el.closest('.katex') && !el.classList.contains('katex')) continue;
      if (el.classList.contains('fo')) continue;
      const cs = getComputedStyle(el); if (cs.display === 'none' || cs.visibility === 'hidden') continue;
      const e = el.getBoundingClientRect(); if (!e.width || !e.height) continue;
      const over = Math.max(e.bottom - r.bottom, e.right - r.right, r.left - e.left, r.top - e.top);
      if (over > 3 && (!worst || over > worst.over)) worst = {over: Math.round(over), tag: el.tagName + '.' + (el.className.baseVal ?? el.className)};
    }
    if (worst) out.push(`${s.id}: ${worst.over}px  ${worst.tag}`.slice(0, 140));
  }
  return out;
});
console.log(res.join('\n') || 'no overflow');
if (errs.length) console.log('ERRORS', [...new Set(errs)].join('\n'));
await b.close();
