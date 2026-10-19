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

  const widgets = { kl, elbo, explorer };
  document.querySelectorAll('[data-widget]').forEach(el => {
    const w = widgets[el.dataset.widget];
    if (w) w(el);
  });
})();
