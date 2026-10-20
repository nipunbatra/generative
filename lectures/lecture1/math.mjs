/**
 * Analytic calculations for the two Lecture 1 prototype sequences.
 *
 * Normal parameters use standard deviation (sigma), never variance.
 * Coin data are an observed ORDERED sequence, so its likelihood has no
 * binomial coefficient. Displayed values evaluate analytic formulae in
 * JavaScript floating point; the fixture's provenance makes this explicit.
 *
 * This module has no dependencies and is shared by the browser and checks.
 */

const LOG_TWO_PI = Math.log(2 * Math.PI);

function finite(value, name) {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
  return value;
}

function positive(value, name) {
  finite(value, name);
  if (value <= 0) throw new RangeError(`${name} must be strictly positive`);
  return value;
}

function checkNormal(distribution, name) {
  finite(distribution.mu, `${name}.mu`);
  positive(distribution.sigma, `${name}.sigma`);
}

function checkBeta(distribution, name) {
  positive(distribution.alpha, `${name}.alpha`);
  positive(distribution.beta, `${name}.beta`);
}

function checkCoinData(data) {
  for (const key of ['heads', 'tails']) {
    if (!Number.isSafeInteger(data[key]) || data[key] < 0) {
      throw new RangeError(`${key} must be a nonnegative safe integer`);
    }
  }
}

export function normalLogPdf(x, distribution) {
  finite(x, 'x');
  checkNormal(distribution, 'Normal');
  const standardised = (x - distribution.mu) / distribution.sigma;
  return -0.5 * LOG_TWO_PI - Math.log(distribution.sigma) - 0.5 * standardised ** 2;
}

export function normalPdf(x, distribution) {
  return Math.exp(normalLogPdf(x, distribution));
}

/** KL(q || p), in nats. The expectation is under q. */
export function normalKL(q, p) {
  checkNormal(q, 'q');
  checkNormal(p, 'p');
  return Math.log(p.sigma / q.sigma)
    + (q.sigma ** 2 + (q.mu - p.mu) ** 2) / (2 * p.sigma ** 2) - 0.5;
}

/** Derivatives of KL(q || p) with respect to q's mu and sigma, with p fixed. */
export function normalKLGrad(q, p) {
  checkNormal(q, 'q');
  checkNormal(p, 'p');
  return {
    dMu: (q.mu - p.mu) / p.sigma ** 2,
    dSigma: q.sigma / p.sigma ** 2 - 1 / q.sigma,
  };
}

// Lanczos approximation to log Gamma, g=7; positive real arguments only.
// Reflection handles positive arguments below 1/2.
const LANCZOS = [
  0.99999999999980993,
  676.5203681218851,
  -1259.1392167224028,
  771.32342877765313,
  -176.61502916214059,
  12.507343278686905,
  -0.13857109526572012,
  9.9843695780195716e-6,
  1.5056327351493116e-7,
];

export function logGamma(value) {
  positive(value, 'Gamma argument');
  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }
  const z = value - 1;
  let sum = LANCZOS[0];
  for (let index = 1; index < LANCZOS.length; index += 1) {
    sum += LANCZOS[index] / (z + index);
  }
  const t = z + 7.5;
  return 0.5 * LOG_TWO_PI + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

/** Digamma for positive arguments, recurrence plus an asymptotic expansion. */
export function digamma(value) {
  positive(value, 'digamma argument');
  let x = value;
  let result = 0;
  while (x < 12) {
    result -= 1 / x;
    x += 1;
  }
  const inverseSquare = 1 / (x * x);
  result += Math.log(x) - 1 / (2 * x)
    - inverseSquare * (1 / 12
      - inverseSquare * (1 / 120
        - inverseSquare * (1 / 252
          - inverseSquare * (1 / 240
            - inverseSquare / 132))));
  return result;
}

export function logBeta(alpha, beta) {
  positive(alpha, 'alpha');
  positive(beta, 'beta');
  return logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
}

/** Log density. Out-of-support points have density zero. */
export function betaLogPdf(r, distribution) {
  finite(r, 'r');
  checkBeta(distribution, 'Beta');
  const { alpha, beta } = distribution;
  if (r < 0 || r > 1) return -Infinity;
  const logNormaliser = logBeta(alpha, beta);
  if (r === 0) return alpha === 1 ? -logNormaliser : (alpha > 1 ? -Infinity : Infinity);
  if (r === 1) return beta === 1 ? -logNormaliser : (beta > 1 ? -Infinity : Infinity);
  return (alpha - 1) * Math.log(r) + (beta - 1) * Math.log1p(-r) - logNormaliser;
}

export function betaPdf(r, distribution) {
  return Math.exp(betaLogPdf(r, distribution));
}

/** E_q[log r] and E_q[log(1-r)] for a Beta q. */
export function betaLogMoments(q) {
  checkBeta(q, 'q');
  const sumDigamma = digamma(q.alpha + q.beta);
  return {
    logR: digamma(q.alpha) - sumDigamma,
    logOneMinusR: digamma(q.beta) - sumDigamma,
  };
}

/** Analytic Beta-to-Beta KL(q || p), evaluated in floating point. */
export function betaKL(q, p) {
  checkBeta(q, 'q');
  checkBeta(p, 'p');
  if (q.alpha === p.alpha && q.beta === p.beta) return 0;
  const moments = betaLogMoments(q);
  const result = logBeta(p.alpha, p.beta) - logBeta(q.alpha, q.beta)
    + (q.alpha - p.alpha) * moments.logR
    + (q.beta - p.beta) * moments.logOneMinusR;
  // Remove negative round-off near a match, not a genuinely negative value.
  return result < 0 && result > -1e-12 ? 0 : result;
}

export function coinPosterior(prior, data) {
  checkBeta(prior, 'prior');
  checkCoinData(data);
  return { alpha: prior.alpha + data.heads, beta: prior.beta + data.tails };
}

/** Log evidence for one observed ordered sequence, without a choose(N,H) term. */
export function coinLogEvidence(prior, data) {
  const posterior = coinPosterior(prior, data);
  return logBeta(posterior.alpha, posterior.beta) - logBeta(prior.alpha, prior.beta);
}

/**
 * Direct ELBO calculation: E_q[log p(D|r)] + E_q[log p(r)] - E_q[log q(r)].
 * Deliberately does NOT compute log evidence minus KL, so the identity can be
 * checked independently. The Beta moments evaluate all three expectations.
 */
export function coinELBO(q, prior, data) {
  checkBeta(q, 'q');
  checkBeta(prior, 'prior');
  checkCoinData(data);
  const moments = betaLogMoments(q);
  const expectedLogLikelihood = data.heads * moments.logR + data.tails * moments.logOneMinusR;
  const expectedLogPrior = (prior.alpha - 1) * moments.logR
    + (prior.beta - 1) * moments.logOneMinusR - logBeta(prior.alpha, prior.beta);
  const expectedLogQ = (q.alpha - 1) * moments.logR
    + (q.beta - 1) * moments.logOneMinusR - logBeta(q.alpha, q.beta);
  return {
    expectedLogLikelihood,
    expectedLogPrior,
    expectedLogJoint: expectedLogLikelihood + expectedLogPrior,
    expectedLogQ,
    entropy: -expectedLogQ,
    elbo: expectedLogLikelihood + expectedLogPrior - expectedLogQ,
  };
}

function linspace(start, end, points) {
  return Array.from({ length: points }, (_, index) => start + (end - start) * index / (points - 1));
}

/** All plotted and displayed prototype numbers originate here. */
export function createFixtures() {
  const target = { mu: 1, sigma: 4 };
  const candidates = [
    { id: 'q1', label: 'q₁', mu: 0, sigma: 2 },
    { id: 'q2', label: 'q₂', mu: 1, sigma: 3 },
  ].map(q => ({
    ...q,
    klToTarget: normalKL(q, target),
    reverseKL: normalKL(target, q),
  }));
  const before = { mu: 0, sigma: 2 };
  const gradient = normalKLGrad(before, target);
  const learningRate = 4;
  const after = { mu: before.mu - learningRate * gradient.dMu, sigma: before.sigma };
  const x = linspace(-11, 13, 481);
  const prior = { alpha: 5, beta: 3 };
  const data = { heads: 17, tails: 13, sequenceLength: 30 };
  const posterior = coinPosterior(prior, data);
  const logEvidence = coinLogEvidence(prior, data);
  const coinCandidates = [
    { id: 'q10-10', alpha: 10, beta: 10 },
    { id: 'q30-20', alpha: 30, beta: 20 },
    { id: 'q22-16', alpha: 22, beta: 16 },
  ].map(q => ({
    ...q,
    ...coinELBO(q, prior, data),
    klToPosterior: betaKL(q, posterior),
    reverseKL: betaKL(posterior, q),
  }));
  const r = linspace(0, 1, 501);
  return {
    schemaVersion: 1,
    provenance: {
      source: 'math.mjs:createFixtures',
      evidenceType: 'analytic formula evaluated in floating point',
      units: 'nats',
      normalConvention: 'Normal(mu, sigma), where sigma is standard deviation',
      coinEvidenceConvention: 'one observed ordered sequence; no binomial coefficient',
      elboMethod: 'direct expected log joint minus expected log q using analytic Beta moments',
      independentCheck: 'data/math-golden.json from PyTorch float64; numerical quadrature checks are separate approximations',
      notation: {
        genericUnknown: 'u',
        coinProbability: 'r',
        variationalParameters: 'ψ',
        futureNeuralModelWeights: 'θ',
      },
    },
    normal: {
      target,
      candidates,
      update: {
        before,
        gradientMu: gradient.dMu,
        gradientSigma: gradient.dSigma,
        heldFixed: ['target.mu', 'target.sigma', 'q.sigma'],
        learningRate,
        after,
        klBefore: normalKL(before, target),
        klAfter: normalKL(after, target),
        decrease: normalKL(before, target) - normalKL(after, target),
        optimalMean: target.mu,
        klAtOptimalMean: normalKL({ mu: target.mu, sigma: before.sigma }, target),
        fixedScaleLimitation: 'Moving only the mean cannot match the target standard deviation.',
      },
      axes: { xMin: -11, xMax: 13, yMin: 0, yMax: 0.23 },
      curves: {
        x,
        target: x.map(value => normalPdf(value, target)),
        q1: x.map(value => normalPdf(value, candidates[0])),
        q2: x.map(value => normalPdf(value, candidates[1])),
        updated: x.map(value => normalPdf(value, after)),
      },
    },
    coin: {
      prior,
      data,
      posterior,
      logEvidence,
      evidence: Math.exp(logEvidence),
      candidates: coinCandidates,
      axes: { rMin: 0, rMax: 1, yMin: 0, yMax: 6.5 },
      curves: {
        r,
        prior: r.map(value => betaPdf(value, prior)),
        posterior: r.map(value => betaPdf(value, posterior)),
        candidates: Object.fromEntries(coinCandidates.map(q => [
          q.id, r.map(value => betaPdf(value, q)),
        ])),
      },
    },
  };
}
