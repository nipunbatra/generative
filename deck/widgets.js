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

  // MNIST as a Gaussian mixture: pick the PCA size d and the number of bumps K, see real samples from that fit
  async function gmmx(root) {
    let info;
    try { info = await (await fetch('figures/data/gmm-explore/index.json')).json(); } catch { return; }
    const [dIn, kIn] = [root.querySelector('.gx-d'), root.querySelector('.gx-k')];
    const $ = c => root.querySelector(c);
    const num = v => Number(v).toLocaleString('en-US').replace('-', '−');
    const update = () => {
      const d = info.d_grid[+dIn.value], k = info.k_grid[+kIn.value];
      $('.gx-dv').textContent = d;
      $('.gx-kv').textContent = k;
      $('.gx-recon').src = `figures/data/gmm-explore/recon_d${d}.png`;
      $('.gx-samples').src = `figures/data/gmm-explore/d${d}_k${k}.png`;
      $('.gx-var').textContent = `${info.var_kept[d]}%`;
      $('.gx-params').textContent = num(info.params[d][k]);
      $('.gx-test').textContent = Number(info.test[d][k]).toFixed(1).replace('-', '−');
      $('.gx-best').textContent = info.best_k[d];
    };
    dIn.addEventListener('input', update);
    kIn.addEventListener('input', update);
    update();
  }


  // ======================================================================== the VI lecture (vi.html)
  // Every widget below draws a meaningful state on load, so ?print shows something useful.
  // The quantities match scripts/make_figures_vi.py: same grids, same formulas.
  const PAL = { x: '#245EDB', xt: '#E4ECFF', enc: '#8B2CDE', enct: '#F1E5FC', dec: '#0F766E', dect: '#D9F2EF',
    z: '#AA4E08', zt: '#FCEFD9', loss: '#BE123C', losst: '#FDE2E7', ink: '#14171F', ink2: '#4A5160', ink3: '#6B7280',
    line: '#D9DDE5', grid: '#ECEEF2', grey: '#C7CCD4' };
  const PRINT = new URLSearchParams(location.search).has('print');
  const FONT = '13px "Avenir Next", Avenir, "Segoe UI", sans-serif';
  const fx = (v, k = 3) => (Math.abs(v) < 0.5 * 10 ** -k ? 0 : v).toFixed(k).replace('-', '−');
  const SQ2PI = Math.sqrt(2 * Math.PI);
  const npdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQ2PI);
  const logsig = a => (a >= 0 ? -Math.log1p(Math.exp(-a)) : a - Math.log1p(Math.exp(a)));
  const sig = a => 1 / (1 + Math.exp(-a));
  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));

  function rng(seed) {                      // a seeded uniform (mulberry32) and standard normal (Box-Muller)
    let a = seed >>> 0, spare = null;
    const u = () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const n = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let x = 0; while (x === 0) x = u();
      const y = u(), rr = Math.sqrt(-2 * Math.log(x));
      spare = rr * Math.sin(2 * Math.PI * y);
      return rr * Math.cos(2 * Math.PI * y);
    };
    return { u, n };
  }

  function crisp(cv) {                      // draw in CSS pixels on a 2x backing store
    if (!cv.dataset.w) {
      cv.dataset.w = cv.getAttribute('width'); cv.dataset.h = cv.getAttribute('height');
      cv.style.width = cv.dataset.w + 'px'; cv.style.height = cv.dataset.h + 'px';
      cv.width = 2 * cv.dataset.w; cv.height = 2 * cv.dataset.h;
    }
    const g = cv.getContext('2d');
    g.setTransform(2, 0, 0, 2, 0, 0);
    g.clearRect(0, 0, +cv.dataset.w, +cv.dataset.h);
    return [g, +cv.dataset.w, +cv.dataset.h];
  }

  function axes(box, xr, yr) {
    const [L, T, W, Hh] = box;
    return { L, T, W, H: Hh, xr, yr,
      X: v => L + (v - xr[0]) / (xr[1] - xr[0]) * W,
      Y: v => T + Hh - (v - yr[0]) / (yr[1] - yr[0]) * Hh };
  }
  function path(g, A, xs, ys) {
    g.beginPath();
    xs.forEach((x, i) => (i ? g.lineTo(A.X(x), A.Y(ys[i])) : g.moveTo(A.X(x), A.Y(ys[i]))));
  }
  function line(g, A, xs, ys, col, lw = 2.4, dash = null, alpha = 1) {
    g.save(); g.globalAlpha = alpha; g.strokeStyle = col; g.lineWidth = lw; g.lineJoin = 'round';
    if (dash) g.setLineDash(dash);
    path(g, A, xs, ys); g.stroke(); g.restore();
  }
  function area(g, A, xs, ys, col, alpha = 1) {
    g.save(); g.globalAlpha = alpha; g.fillStyle = col;
    path(g, A, xs, ys); g.lineTo(A.X(xs[xs.length - 1]), A.Y(0)); g.lineTo(A.X(xs[0]), A.Y(0)); g.closePath(); g.fill(); g.restore();
  }
  function xaxis(g, A, ticks, lab) {
    g.save(); g.strokeStyle = '#9AA1AD'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(A.L, A.Y(A.yr[0]) + 0.5); g.lineTo(A.L + A.W, A.Y(A.yr[0]) + 0.5); g.stroke();
    g.fillStyle = PAL.ink3; g.font = FONT; g.textAlign = 'center';
    for (const t of ticks) {
      g.beginPath(); g.moveTo(A.X(t), A.Y(A.yr[0])); g.lineTo(A.X(t), A.Y(A.yr[0]) + 5); g.stroke();
      g.fillText(fx(t, Number.isInteger(t) ? 0 : 1), A.X(t), A.Y(A.yr[0]) + 19);
    }
    if (lab) { g.font = 'italic 15px "Times New Roman", serif'; g.fillText(lab, A.L + A.W / 2, A.Y(A.yr[0]) + 38); }
    g.restore();
  }
  function hline(g, A, y, col, dash = [6, 4], lw = 1.6) {
    g.save(); g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash);
    g.beginPath(); g.moveTo(A.L, A.Y(y)); g.lineTo(A.L + A.W, A.Y(y)); g.stroke(); g.restore();
  }
  function vline(g, A, x, col, dash = [], lw = 1.6) {
    g.save(); g.strokeStyle = col; g.lineWidth = lw; g.setLineDash(dash);
    g.beginPath(); g.moveTo(A.X(x), A.T); g.lineTo(A.X(x), A.T + A.H); g.stroke(); g.restore();
  }
  const set = (root, cls, v) => root.querySelectorAll(cls).forEach(el => { el.textContent = v; });
  function adam(lr) {                      // Adam on a parameter vector, for gradient ASCENT
    const st = { m: null, v: null, t: 0 };
    return (p, grad) => {
      if (!st.m) { st.m = grad.map(() => 0); st.v = grad.map(() => 0); }
      st.t += 1;
      for (let i = 0; i < p.length; i++) {
        st.m[i] = 0.9 * st.m[i] + 0.1 * grad[i];
        st.v[i] = 0.999 * st.v[i] + 0.001 * grad[i] * grad[i];
        p[i] += lr * (st.m[i] / (1 - 0.9 ** st.t)) / (Math.sqrt(st.v[i] / (1 - 0.999 ** st.t)) + 1e-8);
      }
    };
  }
  function onSlide(root) { const s = root.closest('section'); return !s || s.classList.contains('current'); }

  // The one-number decoder: z ~ N(0, 1), theta = sigmoid(z), h heads and t tails.
  function coinModel(h, t) {
    const n = h + t;
    const loglik = z => h * logsig(z) + t * logsig(-z);
    const lj = z => -0.5 * z * z - 0.5 * Math.log(2 * Math.PI) + loglik(z);
    const dlj = z => -z + h - n * sig(z);
    const N = 12001, lo = -6, dz = 12 / (N - 1);       // the same grid as make_figures_vi.py
    let mx = -Infinity; const v = new Float64Array(N);
    for (let i = 0; i < N; i++) { v[i] = lj(lo + i * dz); mx = Math.max(mx, v[i]); }
    let s = 0; for (let i = 0; i < N; i++) s += Math.exp(v[i] - mx);
    const logpx = mx + Math.log(s * dz);
    const K = 321, gx = linspace(-8, 8, K), gw = gx.map(x => Math.exp(-0.5 * x * x));
    const tot = gw.reduce((a, b) => a + b, 0); for (let k = 0; k < K; k++) gw[k] /= tot;
    const eq = (f, m, sd) => { let a = 0; for (let k = 0; k < K; k++) a += gw[k] * f(m + sd * gx[k]); return a; };
    const klp = (m, sd) => 0.5 * (m * m + sd * sd - 1 - Math.log(sd * sd));
    const fit = (m, sd) => eq(loglik, m, sd);
    return { h, t, n, loglik, lj, dlj, logpx, post: z => Math.exp(lj(z) - logpx), eq, klp, fit, elbo: (m, sd) => fit(m, sd) - klp(m, sd) };
  }

  // ---------------------------------------------------------------- Part 1: toss by toss, the posterior sharpens
  function bayes(root) {
    const cv = root.querySelector('canvas');
    const ours = root.dataset.seq || 'HHTHHHHTHH';
    let seq = ours;
    const xs = linspace(0.001, 0.999, 400), dx = xs[1] - xs[0];
    function draw() {
      const h = [...seq].filter(c => c === 'H').length, t = seq.length - h;
      const prior = xs.map(x => 6 * x * (1 - x));
      const un = xs.map(x => Math.exp((1 + h) * Math.log(x) + (1 + t) * Math.log(1 - x)));
      const Z = un.reduce((a, b) => a + b, 0) * dx;
      const post = un.map(v => v / Z), pmax = Math.max(...post);
      const lk = xs.map(x => Math.exp(h * Math.log(x) + t * Math.log(1 - x)));
      const lmax = Math.max(...lk), lik = lk.map(v => v / lmax * pmax);
      const [g, W, Hh] = crisp(cv);
      const A = axes([20, 14, W - 40, Hh - 58], [0, 1], [0, Math.max(pmax, 1.5) * 1.1]);
      area(g, A, xs, post, PAL.xt);
      line(g, A, xs, post, PAL.x, 3);
      if (seq.length) line(g, A, xs, lik, PAL.dec, 2, [2, 3]);
      line(g, A, xs, prior, PAL.z, 2, [7, 4]);
      xaxis(g, A, [0, 0.5, 1], 'θ, the chance of heads');
      set(root, '.b-h', h); set(root, '.b-t', t); set(root, '.b-a', 2 + h); set(root, '.b-b', 2 + t);
      set(root, '.b-mean', fx((2 + h) / (4 + seq.length), 3));
      root.querySelector('.b-seq').innerHTML = seq.length
        ? [...seq].map((c, i) => `<span class="${c === 'T' ? 'tail' : ''}${i === seq.length - 1 ? ' last' : ''}">${c}</span>`).join('')
        : '<em>no tosses yet</em>';
    }
    root.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
      const a = b.dataset.a;
      if (a === 'H' || a === 'T') seq = (seq + a).slice(-40);
      else if (a === 'ours') seq = ours;
      else seq = '';
      draw();
    }));
    draw();
  }

  // ---------------------------------------------------------------- Part 3: Monte Carlo, the running average settles
  function mc(root) {
    const cv = root.querySelector('canvas');
    const mu = 1, sd = 0.5, exact = mu * mu + sd * sd, per = Math.sqrt(4 * mu * mu * sd * sd + 2 * sd ** 4);
    const first = (root.dataset.first || '').split(',').filter(Boolean).map(Number);
    let R, zs, means;
    const push = z => { zs.push(z); const k = zs.length; means.push(((means[k - 2] || 0) * (k - 1) + z * z) / k); };
    const reset = () => { R = rng(1); zs = []; means = []; first.forEach(push); draw(); };
    const add = k => { for (let i = 0; i < k; i++) push(mu + sd * R.n()); draw(); };
    function draw() {
      const [g, W, Hh] = crisp(cv);
      const A = axes([30, 10, W - 50, 92], [-0.6, 2.6], [0, 0.95]);
      const xs = linspace(-0.6, 2.6, 300), ys = xs.map(x => npdf(x, mu, sd));
      area(g, A, xs, ys, PAL.enct); line(g, A, xs, ys, PAL.enc, 2.2);
      const recent = zs.slice(-300);
      g.fillStyle = PAL.enc;
      recent.forEach((z, i) => {
        const last = i === recent.length - 1;
        g.globalAlpha = last ? 1 : 0.35; g.beginPath(); g.arc(A.X(z), A.Y(0) + (last ? 0 : 0), last ? 6 : 3.5, 0, 7); g.fill();
      });
      g.globalAlpha = 1;
      xaxis(g, A, [0, 1, 2], 'z');
      const Smax = Math.max(1000, zs.length);
      const B = axes([30, 160, W - 50, Hh - 200], [0, Math.log10(Smax)], [0.4, 2.1]);
      g.fillStyle = PAL.dect;
      g.beginPath();
      const ks = linspace(0, Math.log10(Smax), 120);
      ks.forEach((k, i) => { const y = exact + 2 * per / Math.sqrt(10 ** k); i ? g.lineTo(B.X(k), B.Y(Math.min(2.1, y))) : g.moveTo(B.X(k), B.Y(Math.min(2.1, y))); });
      [...ks].reverse().forEach(k => g.lineTo(B.X(k), B.Y(Math.max(0.4, exact - 2 * per / Math.sqrt(10 ** k)))));
      g.closePath(); g.fill();
      hline(g, B, exact, PAL.ink, [], 1.4);
      if (means.length) line(g, B, means.map((_, i) => Math.log10(i + 1)), means.map(v => Math.max(0.4, Math.min(2.1, v))), PAL.dec, 2.2);
      g.save(); g.strokeStyle = '#9AA1AD'; g.lineWidth = 1; g.beginPath(); g.moveTo(B.L, B.Y(0.4)); g.lineTo(B.L + B.W, B.Y(0.4)); g.stroke();
      g.fillStyle = PAL.ink3; g.font = FONT; g.textAlign = 'center';
      for (let k = 0; k <= Math.floor(Math.log10(Smax)); k++) g.fillText((10 ** k).toLocaleString('en-US'), B.X(k), B.Y(0.4) + 18);
      g.textAlign = 'left'; g.fillText('number of draws S (log scale)', B.L, B.Y(0.4) + 36);
      g.fillStyle = PAL.ink; g.fillText('exact 1.25', B.L + B.W - 70, B.Y(exact) - 8);
      g.restore();
      set(root, '.mc-s', zs.length.toLocaleString('en-US'));
      set(root, '.mc-est', zs.length ? fx(means[means.length - 1], 3) : '—');
      set(root, '.mc-last', zs.length ? fx(zs[zs.length - 1], 2) : '—');
    }
    root.querySelectorAll('button').forEach(b => b.addEventListener('click', () => (b.dataset.a === 'reset' ? reset() : add(+b.dataset.a))));
    reset();
    if (PRINT) add(995);
  }

  // ---------------------------------------------------------------- Part 4: KL between two bell curves, as an area
  function klgauss(root) {
    const cv = root.querySelector('canvas');
    const [im, is] = root.querySelectorAll('input[type=range]');
    let dir = 'qp';
    const xs = linspace(-4.5, 4.5, 721), dx = xs[1] - xs[0];
    function draw() {
      const mu = +im.value, sd = +is.value;
      const p = xs.map(x => npdf(x, 0, 1)), q = xs.map(x => npdf(x, mu, sd));
      const f = dir === 'qp' ? q.map((v, i) => v * Math.log(v / p[i])) : p.map((v, i) => v * Math.log(v / q[i]));
      const areaNum = f.reduce((a, b) => a + b, 0) * dx;
      const [g, W, Hh] = crisp(cv);
      const A = axes([16, 8, W - 32, 150], [-4.5, 4.5], [0, Math.max(0.45, Math.max(...q)) * 1.08]);
      area(g, A, xs, p, PAL.zt, 0.7); line(g, A, xs, p, PAL.z, 2.2);
      line(g, A, xs, q, PAL.enc, 2.6);
      const fl = f.map(v => Math.max(-0.6, Math.min(1.6, v)));
      const B = axes([16, 178, W - 32, Hh - 220], [-4.5, 4.5], [Math.min(-0.15, Math.min(...fl)) * 1.1, Math.max(0.3, Math.max(...fl)) * 1.1]);
      g.save();
      for (const [sign, col] of [[1, '#F7B8C6'], [-1, '#E5E7EB']]) {
        g.fillStyle = col; g.beginPath(); g.moveTo(B.X(xs[0]), B.Y(0));
        xs.forEach((x, i) => g.lineTo(B.X(x), B.Y(sign * fl[i] > 0 ? fl[i] : 0)));
        g.lineTo(B.X(xs[xs.length - 1]), B.Y(0)); g.closePath(); g.fill();
      }
      g.restore();
      line(g, B, xs, fl, PAL.loss, 2);
      g.save(); g.strokeStyle = PAL.ink3; g.lineWidth = 0.8; g.beginPath(); g.moveTo(B.L, B.Y(0)); g.lineTo(B.L + B.W, B.Y(0)); g.stroke(); g.restore();
      const Ax = axes([16, 178, W - 32, Hh - 220], [-4.5, 4.5], [B.yr[0], B.yr[1]]);
      xaxis(g, Ax, [-4, -2, 0, 2, 4], 'z');
      const qp = 0.5 * (mu * mu + sd * sd - 1 - Math.log(sd * sd)), pq = Math.log(sd) + (1 + mu * mu) / (2 * sd * sd) - 0.5;
      set(root, '.kg-mu', fx(mu, 2)); set(root, '.kg-sd', fx(sd, 2));
      set(root, '.kg-qp', fx(qp, 3)); set(root, '.kg-pq', fx(pq, 3)); set(root, '.kg-area', fx(areaNum, 3));
      root.querySelectorAll('[data-dir]').forEach(b => b.classList.toggle('on', b.dataset.dir === dir));
    }
    [im, is].forEach(el => el.addEventListener('input', draw));
    root.querySelectorAll('[data-dir]').forEach(b => b.addEventListener('click', () => { dir = b.dataset.dir; draw(); }));
    draw();
  }

  // ---------------------------------------------------------------- Part 4: forward vs reverse KL on two bumps, by gradient descent
  function fwdrev(root) {
    const cvs = [...root.querySelectorAll('canvas')];
    const xs = linspace(-6, 6, 601), dx = xs[1] - xs[0];
    const p = xs.map(x => 0.5 * npdf(x, -2, 0.5) + 0.5 * npdf(x, 2, 0.5));
    const kl = (dir, m, ls) => {
      const s = Math.exp(ls); let a = 0;
      for (let i = 0; i < xs.length; i++) {
        const q = npdf(xs[i], m, s) + 1e-300, pp = p[i] + 1e-300;
        a += dir === 'fwd' ? pp * Math.log(pp / q) : q * Math.log(q / pp);
      }
      return a * dx;
    };
    let st, steps, timer = null;
    const reset = () => { st = ['fwd', 'rev'].map(d => ({ d, p: [-0.6, Math.log(0.6)], opt: adam(0.05) })); steps = 0; draw(); };
    const step = () => {
      for (const s of st) {
        const h = 1e-4, [m, ls] = s.p;
        const g = [-(kl(s.d, m + h, ls) - kl(s.d, m - h, ls)) / (2 * h), -(kl(s.d, m, ls + h) - kl(s.d, m, ls - h)) / (2 * h)];
        s.opt(s.p, g);
      }
      steps += 1;
    };
    function draw() {
      st.forEach((s, k) => {
        const [g, W, Hh] = crisp(cvs[k]);
        const A = axes([12, 8, W - 24, Hh - 50], [-5, 5], [0, 1.2]);
        area(g, A, xs, p, PAL.xt); line(g, A, xs, p, PAL.x, 1.6);
        const [m, ls] = s.p;
        line(g, A, xs, xs.map(x => npdf(x, m, Math.exp(ls))), PAL.enc, 2.8);
        xaxis(g, A, [-4, -2, 0, 2, 4], 'z');
        set(root, `.fr-${s.d}-m`, fx(m, 2)); set(root, `.fr-${s.d}-s`, fx(Math.exp(ls), 2)); set(root, `.fr-${s.d}-kl`, fx(kl(s.d, m, ls), 3));
      });
      set(root, '.fr-step', steps);
    }
    const play = () => {
      if (timer) return;
      const tick = () => {
        if (!onSlide(root) || steps >= 600) { timer = null; return; }
        for (let i = 0; i < 3; i++) step();
        draw(); timer = requestAnimationFrame(tick);
      };
      timer = requestAnimationFrame(tick);
    };
    root.querySelector('[data-a="play"]').addEventListener('click', () => { if (steps >= 600) reset(); play(); });
    root.querySelector('[data-a="reset"]').addEventListener('click', () => { if (timer) cancelAnimationFrame(timer); timer = null; reset(); });
    reset();
    if (PRINT) { while (steps < 600) step(); draw(); }
  }

  // ---------------------------------------------------------------- Part 5: fit q by hand: KL, ELBO + gap, or the two terms
  function vifit(root) {
    const M = coinModel(+(root.dataset.h || 8), +(root.dataset.t || 2));
    const mode = root.dataset.mode || 'kl';
    const cv = root.querySelector('canvas');
    const [im, is] = root.querySelectorAll('input[type=range]');
    const xs = linspace(-2.5, 3.8, 400), post = xs.map(M.post);
    const lo = -10, hi = -5.5, pct = v => `${(Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * 100}%`;
    function draw() {
      const m = +im.value, s = +is.value;
      const q = xs.map(x => npdf(x, m, s));
      const [g, W, Hh] = crisp(cv);
      const A = axes([12, 8, W - 24, Hh - 50], [-2.5, 3.8], [0, Math.max(1.0, Math.max(...q)) * 1.06]);
      area(g, A, xs, post, PAL.xt); line(g, A, xs, post, PAL.x, 2);
      area(g, A, xs, q, PAL.enc, 0.08); line(g, A, xs, q, PAL.enc, 2.8);
      xaxis(g, A, [-2, -1, 0, 1, 2, 3], 'z');
      const e = M.elbo(m, s), gap = M.logpx - e;
      set(root, '.vf-m', fx(m, 2)); set(root, '.vf-s', fx(s, 2)); set(root, '.vf-kl', fx(gap, 3));
      set(root, '.vf-elbo', fx(e, 3)); set(root, '.vf-lpx', fx(M.logpx, 3));
      set(root, '.vf-fit', fx(M.fit(m, s), 3)); set(root, '.vf-klp', fx(M.klp(m, s), 3));
      const nl = root.querySelector('.vf-line');
      if (nl) {
        nl.querySelector('.vf-mark-elbo').style.left = pct(e);
        nl.querySelector('.vf-mark-lpx').style.left = pct(M.logpx);
        const gp = nl.querySelector('.nl-gap'); gp.style.left = pct(e); gp.style.width = `calc(${pct(M.logpx)} - ${pct(e)})`;
      }
      root.querySelectorAll('.vf-bar').forEach(b => { b.style.width = `${Math.min(100, Math.abs(+b.dataset.k === 1 ? M.klp(m, s) : M.fit(m, s)) / 10 * 100)}%`; });
    }
    [im, is].forEach(el => el.addEventListener('input', draw));
    draw();
  }

  // ---------------------------------------------------------------- Part 6: two gradient estimators, the same draws
  function gradvar(root) {
    const cv = root.querySelector('canvas');
    const mu = 1, sd = 0.5;
    let R, sf, rp, jit;
    const push = e => { const z = mu + sd * e; rp.push(2 * z); sf.push(z * z * (z - mu) / (sd * sd)); jit.push(R.u()); };
    const reset = () => { R = rng(3); sf = []; rp = []; jit = []; push(-0.4); draw(); };
    const add = k => { for (let i = 0; i < k; i++) push(R.n()); draw(); };
    const stats = a => { const n = a.length, m = a.reduce((x, y) => x + y, 0) / n; return [m, Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / n)]; };
    function draw() {
      const [g, W, Hh] = crisp(cv);
      const rows = [[sf, PAL.ink2, 'score function'], [rp, PAL.enc, 'reparameterisation']];
      const rowH = (Hh - 50) / 2;
      rows.forEach(([a, col, lab], k) => {
        const A = axes([20, 10 + k * rowH, W - 40, rowH - 26], [-6, 12], [0, 1]);
        g.fillStyle = '#F6F7F9'; g.fillRect(A.L, A.T, A.W, A.H);
        g.fillStyle = col; g.globalAlpha = a.length > 200 ? 0.25 : 0.55;
        a.forEach((v, i) => { if (v < -6 || v > 12) return; g.beginPath(); g.arc(A.X(v), A.T + 8 + jit[i] * (A.H - 16), 3, 0, 7); g.fill(); });
        g.globalAlpha = 1;
        vline(g, A, 2, PAL.loss, [], 2);
        const [m] = stats(a);
        g.fillStyle = col; g.strokeStyle = '#fff'; g.lineWidth = 2;
        g.beginPath(); const X = A.X(Math.max(-6, Math.min(12, m))), Y = A.T + A.H / 2;
        g.moveTo(X, Y - 11); g.lineTo(X + 9, Y); g.lineTo(X, Y + 11); g.lineTo(X - 9, Y); g.closePath(); g.fill(); g.stroke();
        g.font = '600 14px "Avenir Next", Avenir, sans-serif'; g.textAlign = 'right'; g.fillText(lab, A.L + A.W - 6, A.T + 16);
        if (k === 1) xaxis(g, axes([20, 10 + k * rowH, W - 40, rowH - 26], [-6, 12], [0, 1]), [-6, -3, 0, 2, 3, 6, 9, 12], '');
      });
      const [m1, s1] = stats(sf), [m2, s2] = stats(rp);
      set(root, '.gv-n', sf.length.toLocaleString('en-US'));
      set(root, '.gv-sfm', fx(m1, 2)); set(root, '.gv-rpm', fx(m2, 2));
      set(root, '.gv-sfs', sf.length > 1 ? fx(s1, 2) : '—'); set(root, '.gv-rps', rp.length > 1 ? fx(s2, 2) : '—');
    }
    root.querySelectorAll('button').forEach(b => b.addEventListener('click', () => (b.dataset.a === 'reset' ? reset() : add(+b.dataset.a))));
    reset();
    if (PRINT) add(299);
  }

  // ---------------------------------------------------------------- Part 6: press play, VI climbs the ELBO
  function ascent(root) {
    const M = coinModel(8, 2);
    const [cq, ce] = root.querySelectorAll('canvas');
    const xs = linspace(-2.2, 3.6, 360), post = xs.map(M.post), STEPS = 400;
    let S = 10, P, opt, step, trace, trail, R, timer = null;
    const e0 = M.elbo(0, 1);
    function reset() {
      if (timer) cancelAnimationFrame(timer);
      timer = null; P = [0, 0]; opt = adam(0.05); step = 0; trace = [e0]; trail = []; R = rng(1); draw();
    }
    function once() {
      const s = Math.exp(P[1]); let gm = 0, gs = 0;
      for (let k = 0; k < S; k++) { const e = R.n(), z = P[0] + s * e, d = M.dlj(z); gm += d; gs += d * s * e; }
      opt(P, [gm / S, gs / S + 1]);                 // d ELBO / d log s gets +1 from the entropy of q
      step += 1;
      trace.push(M.elbo(P[0], Math.exp(P[1])));
      if (step % 8 === 0) trail.push([P[0], Math.exp(P[1])]);
    }
    function draw() {
      const s = Math.exp(P[1]);
      let [g, W, Hh] = crisp(cq);
      const A = axes([12, 8, W - 24, Hh - 50], [-2.2, 3.6], [0, 0.95]);
      area(g, A, xs, post, PAL.xt); line(g, A, xs, post, PAL.x, 1.8);
      trail.slice(-12).forEach(([m, sd], i, a) => line(g, A, xs, xs.map(x => npdf(x, m, sd)), PAL.enc, 1.2, null, 0.12 + 0.3 * i / a.length));
      line(g, A, xs, xs.map(x => npdf(x, P[0], s)), PAL.enc, 3);
      xaxis(g, A, [-2, -1, 0, 1, 2, 3], 'z');
      [g, W, Hh] = crisp(ce);
      const B = axes([54, 12, W - 70, Hh - 54], [0, STEPS], [e0 - 0.2, M.logpx + 0.25]);
      g.save(); g.strokeStyle = PAL.grid; g.lineWidth = 1;
      for (let v = -8; v <= -6; v += 0.5) { g.beginPath(); g.moveTo(B.L, B.Y(v)); g.lineTo(B.L + B.W, B.Y(v)); g.stroke(); }
      g.fillStyle = PAL.ink3; g.font = FONT; g.textAlign = 'right';
      for (let v = -8; v <= -6; v += 0.5) g.fillText(fx(v, 1), B.L - 8, B.Y(v) + 4);
      g.restore();
      hline(g, B, M.logpx, PAL.x);
      line(g, B, trace.map((_, i) => i), trace, PAL.dec, 2.4);
      xaxis(g, B, [0, 100, 200, 300, 400], 'step');
      g.fillStyle = PAL.x; g.font = '600 14px "Avenir Next", Avenir, sans-serif'; g.textAlign = 'right';
      g.fillText('log p(x)', B.L + B.W, B.Y(M.logpx) - 7);
      const e = trace[trace.length - 1];
      set(root, '.as-step', step); set(root, '.as-m', fx(P[0], 2)); set(root, '.as-s', fx(s, 2));
      set(root, '.as-elbo', fx(e, 3)); set(root, '.as-gap', fx(M.logpx - e, 3)); set(root, '.as-S', S);
      root.querySelectorAll('[data-s]').forEach(b => b.classList.toggle('on', +b.dataset.s === S));
      const pb = root.querySelector('[data-a="play"]'); if (pb) pb.textContent = timer ? 'pause' : (step >= STEPS ? 'play again' : 'play');
    }
    function tick() {
      if (!onSlide(root) || step >= STEPS) { timer = null; draw(); return; }
      once(); if (step < 60 || step % 2 === 0) draw();
      timer = requestAnimationFrame(tick);
    }
    root.querySelector('[data-a="play"]').addEventListener('click', () => {
      if (timer) { cancelAnimationFrame(timer); timer = null; draw(); return; }
      if (step >= STEPS) reset();
      timer = requestAnimationFrame(tick); draw();
    });
    root.querySelector('[data-a="reset"]').addEventListener('click', reset);
    root.querySelectorAll('[data-s]').forEach(b => b.addEventListener('click', () => { S = +b.dataset.s; reset(); }));
    reset();
  }

  // ---------------------------------------------------------------- Part 7: a Gaussian q in the plane, diagonal or full
  const BLUES = ['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'];
  const TARGETS = {
    tilted: (() => {
      const rho = 0.9, det = 1 - rho * rho;
      return { logp: (a, b) => -0.5 * (a * a - 2 * rho * a * b + b * b) / det - Math.log(2 * Math.PI) - 0.5 * Math.log(det),
        grad: (a, b) => [-(a - rho * b) / det, -(b - rho * a) / det], init: [1.6, -1.2] };
    })(),
    banana: { logp: (a, b) => { const m2 = 0.6 * a * a - 1.2; return -0.5 * (a / 1.2) ** 2 - Math.log(1.2) - 0.5 * ((b - m2) / 0.45) ** 2 - Math.log(0.45) - Math.log(2 * Math.PI); },
      grad: (a, b) => { const m2 = 0.6 * a * a - 1.2, r2 = (b - m2) / (0.45 * 0.45); return [-a / 1.44 + r2 * 1.2 * a, -r2]; }, init: [1.4, 1.6] },
    bumps: (() => {
      const s = 0.55, cs = [-1.5, 1.5];
      const comp = (a, b) => cs.map(c => -0.5 * ((a - c) ** 2 + (b - c) ** 2) / (s * s) - 2 * Math.log(s) - Math.log(2 * Math.PI) + Math.log(0.5));
      return { logp: (a, b) => { const [u, v] = comp(a, b), mx = Math.max(u, v); return mx + Math.log(Math.exp(u - mx) + Math.exp(v - mx)); },
        grad: (a, b) => { const [u, v] = comp(a, b), mx = Math.max(u, v), wu = Math.exp(u - mx), wv = Math.exp(v - mx), r0 = wu / (wu + wv), r1 = 1 - r0;
          return [-(r0 * (a - cs[0]) + r1 * (a - cs[1])) / (s * s), -(r0 * (b - cs[0]) + r1 * (b - cs[1])) / (s * s)]; }, init: [0.3, 0.2] };
    })(),
  };
  function vi2d(root) {
    const cv = root.querySelector('canvas');
    const lim = 3.2, STEPS = 1500;
    let tname = root.dataset.target || 'tilted', fam = root.dataset.family || 'diag';
    let P, opt, step, R, timer = null, heat = null, klv = null;
    const fixed = (() => { const r = rng(99), a = []; for (let i = 0; i < 3000; i++) a.push([r.n(), r.n()]); return a; })();
    function makeHeat() {
      const n = 230, img = document.createElement('canvas'); img.width = n; img.height = n;
      const g = img.getContext('2d'), id = g.createImageData(n, n), T = TARGETS[tname];
      const vals = new Float64Array(n * n); let mx = -Infinity;
      for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const a = -lim + (i + 0.5) / n * 2 * lim, b = lim - (j + 0.5) / n * 2 * lim, v = T.logp(a, b);
        vals[j * n + i] = v; mx = Math.max(mx, v);
      }
      for (let k = 0; k < n * n; k++) {
        const d = Math.exp(vals[k] - mx), lev = Math.floor(d * 9) - 1;
        if (d < 0.04 || lev < 0) { id.data[4 * k + 3] = 0; continue; }
        const c = BLUES[Math.min(BLUES.length - 1, lev)];
        id.data[4 * k] = parseInt(c.slice(1, 3), 16); id.data[4 * k + 1] = parseInt(c.slice(3, 5), 16); id.data[4 * k + 2] = parseInt(c.slice(5, 7), 16);
        id.data[4 * k + 3] = 235;
      }
      g.putImageData(id, 0, 0);
      heat = img;
    }
    const Lmat = () => [Math.exp(P[2]), fam === 'full' ? P[4] : 0, Math.exp(P[3])];   // L = [[l11, 0], [l21, l22]]
    function computeKL() {
      const T = TARGETS[tname], [l11, l21, l22] = Lmat(); let a = 0;
      for (const [e1, e2] of fixed) a += T.logp(P[0] + l11 * e1, P[1] + l21 * e1 + l22 * e2);
      klv = -(a / fixed.length) - (P[2] + P[3] + Math.log(2 * Math.PI) + 1);
    }
    function reset() {
      if (timer) cancelAnimationFrame(timer);
      timer = null; const T = TARGETS[tname];
      P = [T.init[0], T.init[1], Math.log(0.5), Math.log(0.5), 0]; opt = adam(0.03); step = 0; R = rng(5);
      makeHeat(); computeKL(); draw();
    }
    function once() {
      const T = TARGETS[tname], [l11, l21, l22] = Lmat(), S = 32, gr = [0, 0, 0, 0, 0];
      for (let k = 0; k < S; k++) {
        const e1 = R.n(), e2 = R.n(), [g1, g2] = T.grad(P[0] + l11 * e1, P[1] + l21 * e1 + l22 * e2);
        gr[0] += g1; gr[1] += g2; gr[2] += g1 * e1 * l11; gr[3] += g2 * e2 * l22; gr[4] += g2 * e1;
      }
      for (let i = 0; i < 5; i++) gr[i] /= S;
      gr[2] += 1; gr[3] += 1;                        // the entropy of q: log l11 + log l22 + const
      if (fam !== 'full') gr[4] = 0;
      opt(P, gr); step += 1;
    }
    function draw() {
      const [g, W] = crisp(cv);
      const X = v => (v + lim) / (2 * lim) * W, Y = v => W - (v + lim) / (2 * lim) * W;
      g.fillStyle = '#fff'; g.fillRect(0, 0, W, W);
      g.imageSmoothingEnabled = true; g.drawImage(heat, 0, 0, W, W);
      const [l11, l21, l22] = Lmat();
      const c11 = l11 * l11, c12 = l11 * l21, c22 = l21 * l21 + l22 * l22;
      const tr = c11 + c22, dt = c11 * c22 - c12 * c12, disc = Math.sqrt(Math.max(0, tr * tr / 4 - dt));
      const ev1 = tr / 2 + disc, ev2 = tr / 2 - disc;
      const ang = Math.abs(c12) < 1e-12 ? (c11 >= c22 ? 0 : Math.PI / 2) : Math.atan2(ev1 - c11, c12);
      g.strokeStyle = PAL.enc; g.lineWidth = 2.6;
      for (const k of [1, 2]) {
        g.beginPath();
        g.ellipse(X(P[0]), Y(P[1]), k * Math.sqrt(ev1) / (2 * lim) * W, k * Math.sqrt(Math.max(ev2, 1e-9)) / (2 * lim) * W, -ang, 0, 2 * Math.PI);
        g.stroke();
      }
      g.fillStyle = PAL.enc; g.beginPath(); g.arc(X(P[0]), Y(P[1]), 5, 0, 7); g.fill();
      set(root, '.v2-kl', klv === null ? '—' : fx(Math.max(0, klv), 3));
      set(root, '.v2-s1', fx(Math.sqrt(c11), 2)); set(root, '.v2-s2', fx(Math.sqrt(c22), 2));
      set(root, '.v2-rho', fx(c12 / Math.sqrt(c11 * c22), 2)); set(root, '.v2-step', step);
      root.querySelectorAll('[data-t]').forEach(b => b.classList.toggle('on', b.dataset.t === tname));
      root.querySelectorAll('[data-f]').forEach(b => b.classList.toggle('on', b.dataset.f === fam));
      const pb = root.querySelector('[data-a="play"]'); if (pb) pb.textContent = timer ? 'pause' : (step >= STEPS ? 'play again' : 'play');
    }
    function tick() {
      if (!onSlide(root) || step >= STEPS) { timer = null; computeKL(); draw(); return; }
      for (let i = 0; i < 4; i++) once();
      if (step % 20 === 0) computeKL();
      draw(); timer = requestAnimationFrame(tick);
    }
    root.querySelector('[data-a="play"]').addEventListener('click', () => {
      if (timer) { cancelAnimationFrame(timer); timer = null; draw(); return; }
      if (step >= STEPS) reset();
      timer = requestAnimationFrame(tick); draw();
    });
    root.querySelector('[data-a="reset"]').addEventListener('click', reset);
    root.querySelectorAll('[data-t]').forEach(b => b.addEventListener('click', () => { tname = b.dataset.t; reset(); }));
    root.querySelectorAll('[data-f]').forEach(b => b.addEventListener('click', () => { fam = b.dataset.f; reset(); }));
    reset();
  }

  const widgets = { kl, elbo, explorer, latent, gmmx, bayes, mc, klgauss, fwdrev, vifit, gradvar, ascent, vi2d };
  document.querySelectorAll('[data-widget]').forEach(el => {
    const w = widgets[el.dataset.widget];
    if (w) w(el);
  });
})();
