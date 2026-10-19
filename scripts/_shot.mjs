// one-off: screenshot chosen slides (final build) from part1/part2 via the review server
import puppeteer from 'puppeteer';
const [deck, ids] = [process.argv[2], process.argv[3].split(',')];
const out = process.argv[4];
const b = await puppeteer.launch({headless:true, executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
const p = await b.newPage(); await p.setViewport({width:1280,height:720});
const errs = []; p.on('pageerror', e => errs.push(String(e))); p.on('console', m => { if (m.type()==='error') errs.push(m.text()); });
for (const id of ids) {
  await p.goto(`http://127.0.0.1:8771/${deck}#${id}/99`, {waitUntil:'load'});
  await p.waitForFunction(() => window.deckReady);
  await new Promise(r => setTimeout(r, 700));
  await p.screenshot({path: `${out}/${id}.png`});
}
await b.close();
if (errs.length) console.log([...new Set(errs)].join('\n'));
