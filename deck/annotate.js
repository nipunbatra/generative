// Right-click annotations for review.
//   Right-click anything on a slide → type a note → Save (or ⌘/Ctrl+Enter).
//   A numbered pin marks the spot; click a pin to edit or delete it.
//   Press A to list every note (copy them all as text, or clear).
//   Shift+right-click still opens the normal browser menu.
// Notes live in this browser's localStorage. window.deckNotes() returns them,
// so Claude can read them straight from the browser pane.
(() => {
  const body = document.body;
  if (body.classList.contains('print')) return;
  const KEY = 'deck-notes:' + (location.pathname.split('/').pop() || 'index.html');
  const load = () => { try { return JSON.parse(localStorage.getItem(KEY) || '[]'); } catch { return []; } };
  const store = notes => { try { localStorage.setItem(KEY, JSON.stringify(notes)); } catch { /* private mode */ } };
  let notes = load();

  // ------------------------------------------------------------ what was clicked?
  function describe(el) {
    const tex = el.closest('.katex')?.querySelector('annotation[encoding="application/x-tex"]');
    if (tex) return { kind: 'maths', text: tex.textContent.trim() };
    const svg = el.closest('svg.diagram');
    if (svg) {
      const t = el.closest('text, foreignObject')?.textContent.trim();
      return { kind: 'diagram ' + (svg.getAttribute('aria-label') || ''), text: t || '' };
    }
    if (el.tagName === 'IMG') return { kind: 'image', text: el.getAttribute('src') };
    const block = el.closest('h2, p, li, td, th, pre, .node, .callout, .stat, .cap, .q, .kicker') || el;
    return { kind: block.tagName.toLowerCase() + (block.className ? '.' + String(block.className).split(' ')[0] : ''),
             text: (block.innerText || block.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 160) };
  }

  // ------------------------------------------------------------ pins
  function renderPins() {
    document.querySelectorAll('.anno-pin').forEach(p => p.remove());
    notes.forEach(n => {
      const slide = document.getElementById(n.slide) || document.querySelectorAll('section.slide')[n.slideNo - 1];
      if (!slide) return;
      const pin = document.createElement('button');
      pin.className = 'anno-pin interactive';
      pin.textContent = n.n;
      pin.title = n.note;
      pin.style.left = n.x + 'px';
      pin.style.top = n.y + 'px';
      pin.onclick = e => { e.stopPropagation(); openBox(e.clientX, e.clientY, n); };
      slide.appendChild(pin);
    });
  }

  // ------------------------------------------------------------ the note box
  const box = document.createElement('div');
  box.className = 'anno-box interactive';
  box.innerHTML = '<div class="anno-target"></div><textarea rows="4" placeholder="What should change here?"></textarea>' +
    '<div class="anno-actions"><button data-a="del">Delete</button><span></span>' +
    '<button data-a="cancel">Cancel</button><button data-a="save" class="primary">Save</button></div>';
  body.appendChild(box);
  const ta = box.querySelector('textarea');
  let pending = null;

  function openBox(cx, cy, existing, draft) {
    pending = existing || draft;
    box.querySelector('.anno-target').textContent =
      `slide ${pending.slideNo}, click ${pending.build} · ${pending.target.kind}: ${pending.target.text}`.slice(0, 220);
    ta.value = existing ? existing.note : '';
    box.querySelector('[data-a="del"]').style.visibility = existing ? 'visible' : 'hidden';
    box.style.display = 'block';
    box.style.left = Math.min(cx, innerWidth - 380) + 'px';
    box.style.top = Math.min(cy, innerHeight - 220) + 'px';
    ta.focus();
  }
  function closeBox() { box.style.display = 'none'; pending = null; }
  function save() {
    const text = ta.value.trim();
    if (!pending || !text) return closeBox();
    const existing = notes.find(n => n.n === pending.n);
    if (existing) existing.note = text;
    else notes.push({ ...pending, note: text, n: (notes.at(-1)?.n || 0) + 1, at: new Date().toISOString() });
    store(notes); renderPins(); closeBox();
  }
  box.addEventListener('click', e => {
    e.stopPropagation();
    const a = e.target.dataset.a;
    if (a === 'save') save();
    if (a === 'cancel') closeBox();
    if (a === 'del' && pending) { notes = notes.filter(n => n.n !== pending.n); store(notes); renderPins(); closeBox(); }
  });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); save(); }
    if (e.key === 'Escape') { e.preventDefault(); closeBox(); }
  });

  document.getElementById('deck').addEventListener('contextmenu', e => {
    if (e.shiftKey) return;
    const slide = e.target.closest('section.slide');
    if (!slide) return;
    e.preventDefault();
    const r = slide.getBoundingClientRect();
    const scale = r.width / 1280 || 1;
    const all = [...document.querySelectorAll('section.slide')];
    openBox(e.clientX, e.clientY, null, {
      slide: slide.id, slideNo: all.indexOf(slide) + 1, build: Number(slide.dataset.build || 1),
      title: slide.querySelector('h2')?.childNodes[0]?.textContent.trim() || '',
      x: Math.round((e.clientX - r.left) / scale), y: Math.round((e.clientY - r.top) / scale),
      target: describe(e.target),
    });
  });

  // ------------------------------------------------------------ list (key A)
  const panel = document.createElement('div');
  panel.className = 'anno-panel interactive';
  body.appendChild(panel);
  const asText = () => notes.map(n =>
    `${n.n}. slide ${n.slideNo} "${n.title}" (click ${n.build}) · ${n.target.kind}: ${n.target.text}\n   → ${n.note}`).join('\n');
  function showPanel() {
    panel.innerHTML = `<h4>Review notes (${notes.length})</h4><pre>${asText().replace(/</g, '&lt;') || 'No notes yet. Right-click anything on a slide.'}</pre>` +
      '<div class="anno-actions"><button data-a="copy" class="primary">Copy all</button><button data-a="clear">Clear all</button><button data-a="close">Close</button></div>';
    panel.style.display = 'block';
  }
  panel.addEventListener('click', async e => {
    e.stopPropagation();
    const a = e.target.dataset.a;
    if (a === 'copy') { try { await navigator.clipboard.writeText(asText()); e.target.textContent = 'Copied'; } catch { /* no clipboard */ } }
    if (a === 'clear' && confirm('Delete all review notes for this deck?')) { notes = []; store(notes); renderPins(); showPanel(); }
    if (a === 'close') panel.style.display = 'none';
  });
  addEventListener('keydown', e => {
    if (e.target.closest?.('textarea, input')) return;
    if (e.key === 'a' || e.key === 'A') panel.style.display === 'block' ? (panel.style.display = 'none') : showPanel();
  });

  window.deckNotes = () => load();
  renderPins();
})();
