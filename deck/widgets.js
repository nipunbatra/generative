// Small live widgets. Each has a meaningful static state for the PDF.
(() => {
  const log = Math.log;
  const f3 = v => (Math.abs(v) < 5e-4 ? '0.000' : (v < 0 ? '−' : '') + Math.abs(v).toFixed(3));

  // KL between P = (0.8, 0.2) and Q = (q, 1 - q), both directions
  function kl(root) {
    const input = root.querySelector('input');
    const out = root.querySelector('.readout');
    const pq = root.querySelector('.kl-pq');
    const qp = root.querySelector('.kl-qp');
    const P = [0.8, 0.2];
    const update = () => {
      const q = Number(input.value), Q = [q, 1 - q];
      const a = P[0] * log(P[0] / Q[0]) + P[1] * log(P[1] / Q[1]);
      const b = Q[0] * log(Q[0] / P[0]) + Q[1] * log(Q[1] / P[1]);
      out.textContent = q.toFixed(2);
      pq.textContent = a.toFixed(3);
      qp.textContent = b.toFixed(3);
    };
    input.addEventListener('input', update);
    update();
  }

  // Neat/messy writer: ELBO + KL(q || posterior) = log p(x) for every q
  function elbo(root) {
    const slide = root.closest('section');
    const input = root.querySelector('input');
    const out = root.querySelector('.q0');
    const set = (cls, v) => { const el = slide.querySelector(cls); if (el) el.textContent = f3(v); };
    const update = () => {
      const q0 = Number(input.value), q = [q0, 1 - q0];
      const lik = [0.8, 0.2], prior = [0.5, 0.5], post = [0.8, 0.2];
      const ell = q[0] * log(lik[0]) + q[1] * log(lik[1]);
      const klp = q[0] * log(q[0] / prior[0]) + q[1] * log(q[1] / prior[1]);
      const gap = q[0] * log(q[0] / post[0]) + q[1] * log(q[1] / post[1]);
      out.textContent = q0.toFixed(2);
      set('.v-ell', ell); set('.v-klp', -klp); set('.v-elbo', ell - klp); set('.v-gap', gap); set('.v-lpx', ell - klp + gap);
    };
    input.addEventListener('input', update);
    update();
  }

  // Latent map magnifier: hover a cell of the decoded grid to enlarge it
  function explorer(root) {
    const img = root.querySelector('img');
    const lens = document.createElement('div');
    lens.className = 'lens';
    root.appendChild(lens);
    const n = 18;
    root.addEventListener('mousemove', e => {
      const r = img.getBoundingClientRect();
      const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
      if (fx < 0 || fy < 0 || fx > 1 || fy > 1) { lens.style.display = 'none'; return; }
      const cx = Math.min(n - 1, Math.floor(fx * n)), cy = Math.min(n - 1, Math.floor(fy * n));
      lens.style.display = 'block';
      lens.style.backgroundImage = `url(${img.getAttribute('src')})`;
      lens.style.backgroundSize = `${n * 100}% ${n * 100}%`;
      lens.style.backgroundPosition = `${cx / (n - 1) * 100}% ${cy / (n - 1) * 100}%`;
    });
    root.addEventListener('mouseleave', () => { lens.style.display = 'none'; });
  }


  // Latent-space app: decode any 2-D code in the browser with the trained decoder's own weights.
  const DIGIT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948', '#7c5a3a', '#5f6b7a'];
  async function latent(root) {
    const name = root.dataset.model;
    const base = 'figures/data/explorer-' + name;
    let meta, W, sprite;
    try {
      [meta, W] = await Promise.all([
        fetch(base + '.json').then(r => r.json()),
        fetch(base + '.bin').then(r => r.arrayBuffer()).then(b => new Float32Array(b)),
      ]);
      sprite = new Image();
      sprite.src = 'figures/data/explorer-sprite.png';
      await sprite.decode();
    } catch { root.classList.add('lx-offline'); return; }
    const [H1, H2, OUT] = meta.sizes;
    const o1 = 0, ob1 = H1 * 2, o2 = ob1 + H1, ob2 = o2 + H2 * H1, o3 = ob2 + H2, ob3 = o3 + OUT * H2;
    const h1 = new Float32Array(H1), h2 = new Float32Array(H2);
    function decode(z1, z2) {
      for (let i = 0; i < H1; i++) h1[i] = Math.max(0, W[o1 + 2 * i] * z1 + W[o1 + 2 * i + 1] * z2 + W[ob1 + i]);
      for (let i = 0; i < H2; i++) { let a = W[ob2 + i]; const r = o2 + i * H1; for (let j = 0; j < H1; j++) a += W[r + j] * h1[j]; h2[i] = Math.max(0, a); }
      const px = new Float32Array(OUT);
      for (let i = 0; i < OUT; i++) { let a = W[ob3 + i]; const r = o3 + i * H2; for (let j = 0; j < H2; j++) a += W[r + j] * h2[j]; px[i] = 1 / (1 + Math.exp(-a)); }
      return px;
    }
    const plot = root.querySelector('.lx-plot'), g = plot.getContext('2d');
    const out = root.querySelector('.lx-out').getContext('2d');
    const orig = root.querySelector('.lx-orig').getContext('2d');
    const origFig = root.querySelector('.lx-origfig');
    const tip = root.querySelector('.lx-tip'), tipCtx = tip.querySelector('canvas').getContext('2d'), tipLab = tip.querySelector('span');
    const s1 = root.querySelector('.lx-s1'), s2 = root.querySelector('.lx-s2');
    const r1 = root.querySelector('.lx-z1'), r2 = root.querySelector('.lx-z2');
    const [[x0, x1], [y0, y1]] = meta.range;
    const S = plot.width, M = 90;
    const X = v => M + (v - x0) / (x1 - x0) * (S - 2 * M), Y = v => S - M - (v - y0) / (y1 - y0) * (S - 2 * M);
    const iX = px => x0 + (px - M) / (S - 2 * M) * (x1 - x0), iY = py => y0 + (S - M - py) / (S - 2 * M) * (y1 - y0);
    for (const [sl, lo, hi] of [[s1, x0, x1], [s2, y0, y1]]) { sl.min = lo; sl.max = hi; sl.step = (hi - lo) / 500; }
    let z = [0, 0], picked = -1;

    function spriteTo(ctx, k) {
      ctx.clearRect(0, 0, 28, 28);
      ctx.drawImage(sprite, (k % meta.cols) * 28, Math.floor(k / meta.cols) * 28, 28, 28, 0, 0, 28, 28);
    }
    function draw() {
      g.fillStyle = '#fff'; g.fillRect(0, 0, S, S);
      g.strokeStyle = '#D9DDE5'; g.lineWidth = 2;
      g.beginPath(); g.moveTo(M, Y(0)); g.lineTo(S - M, Y(0)); g.moveTo(X(0), M); g.lineTo(X(0), S - M); g.stroke();
      g.fillStyle = '#6B7280'; g.font = '40px Avenir Next, sans-serif'; g.textAlign = 'center';
      for (const t of [x0, (x0 + x1) / 2, x1]) g.fillText(t.toFixed(0), X(t), S - M + 50);
      g.textAlign = 'right';
      for (const t of [y0, (y0 + y1) / 2, y1]) g.fillText(t.toFixed(0), M - 10, Y(t) + 8);
      meta.points.forEach(([a, b, c]) => { g.fillStyle = DIGIT[c] + 'AA'; g.beginPath(); g.arc(X(a), Y(b), 4.5, 0, 7); g.fill(); });
      g.setLineDash([10, 8]); g.strokeStyle = '#AA4E08'; g.lineWidth = 2.5;
      for (const r of [1, 2]) { g.beginPath(); g.ellipse(X(0), Y(0), Math.abs(X(r) - X(0)), Math.abs(Y(r) - Y(0)), 0, 0, 7); g.stroke(); }
      g.setLineDash([]);
      if (picked >= 0) { const [a, b] = meta.points[picked]; g.strokeStyle = '#14171F'; g.lineWidth = 3; g.beginPath(); g.arc(X(a), Y(b), 13, 0, 7); g.stroke(); }
      g.fillStyle = '#AA4E08'; g.strokeStyle = '#fff'; g.lineWidth = 4;
      g.beginPath(); g.arc(X(z[0]), Y(z[1]), 13, 0, 7); g.fill(); g.stroke();
    }
    function render() {
      const px = decode(z[0], z[1]);
      const im = out.createImageData(28, 28);
      px.forEach((v, i) => { const c = Math.round(255 * (1 - v)); im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = c; im.data[4 * i + 3] = 255; });
      out.putImageData(im, 0, 0);
      s1.value = z[0]; s2.value = z[1];
      r1.textContent = z[0].toFixed(2); r2.textContent = z[1].toFixed(2);
      origFig.style.visibility = picked >= 0 ? 'visible' : 'hidden';
      if (picked >= 0) spriteTo(orig, picked);
      draw();
    }
    const setZ = (a, b, k = -1) => { z = [a, b]; picked = k; render(); };
    const toPlot = e => { const r = plot.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * S, (e.clientY - r.top) / r.height * S, r]; };
    plot.addEventListener('click', e => { const [px, py] = toPlot(e); setZ(iX(px), iY(py)); });
    plot.addEventListener('mousemove', e => {
      const [px, py, r] = toPlot(e);
      let best = -1, bd = 14 * 14;
      meta.points.forEach(([a, b], k) => { const d = (X(a) - px) ** 2 + (Y(b) - py) ** 2; if (d < bd) { bd = d; best = k; } });
      if (best < 0) { tip.style.display = 'none'; return; }
      spriteTo(tipCtx, best);
      tipLab.textContent = 'a ' + meta.points[best][2];
      tip.style.display = 'flex';
      tip.style.left = (px / S * 100) + '%';
      tip.style.top = (py / S * 100) + '%';
    });
    plot.addEventListener('mouseleave', () => (tip.style.display = 'none'));
    s1.addEventListener('input', () => setZ(+s1.value, z[1]));
    s2.addEventListener('input', () => setZ(z[0], +s2.value));
    const randn = () => Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    root.querySelector('[data-a="prior"]').addEventListener('click', () => setZ(randn(), randn()));
    root.querySelector('[data-a="real"]').addEventListener('click', () => {
      const k = Math.floor(Math.random() * meta.points.length);
      setZ(meta.points[k][0], meta.points[k][1], k);
    });
    setZ(0, 0);
  }

  const widgets = { kl, elbo, explorer, latent };
  document.querySelectorAll('[data-widget]').forEach(el => {
    const w = widgets[el.dataset.widget];
    if (w) w(el);
  });
})();
