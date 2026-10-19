"""Probabilistic ELBO and GAN detours, with contextual reading links."""
from story import D, P, M, R, slider, btn
from resource_catalog import RESOURCES, SLIDE_READINGS

def extend(slides):
    by = {s['id']: s for s in slides}
    additions = {}
    def add(id, title, kind, builds, seconds, after, notes, answer, controls='', source='gan-paper'):
        s = dict(by['three-objects'])
        s.update(id=id, title=title, question=title, body=D(kind, controls), builds=builds,
                 planned_seconds=seconds, detour=True, notes=notes, answer=answer, idea='', appendix=False,
                 section=by[after]['section'], source=RESOURCES[source]['url'],
                 provenance='ANALYTICAL / EXACT · original probability example')
        additions.setdefault(after, []).append(s)
        by[id] = s
    add('elbo-reweight', 'Can we average over a distribution we can sample?', 'prob-reweight', 3, 90, 'kl-modes',
        'Return to the exact three-state model. Define w(z)=p(x,z)/q(z|x), with q positive wherever the joint contributes. Multiplying each weight by q recovers the same joint contributions .02,.40,.06. Thus E_q[w]=p(x)=.48. We have changed the averaging distribution, not the model or its evidence. For continuous z the sum becomes an integral. This identity alone does not make the logarithm of that expectation easy to optimize.',
        'Multiply and divide by q: p(x)=E_q[p(x,z)/q(z|x)]. The expectation is over the approximate posterior.', source='vae-paper')
    add('elbo-jensen', 'What changes when we move the log inside?', 'prob-jensen', 3, 120, 'elbo-reweight',
        'The weights are .05, 4/3 and .2 with averaging probabilities .4,.3,.3. First show log applied after averaging: log(.48)=-.733969. Then average the three log weights: -1.594820. The logarithm is concave, so E[log w] <= log E[w]. The lower point is the q-weighted average of the three points on the log curve; it is below the curve at the average weight. This is Jensen’s inequality. The lower expression is the Evidence Lower BOund. Equality occurs when w is constant q-almost surely, as when q is the true posterior in this toy.',
        'The expectation of log weights is a lower bound on the log of their expectation: ELBO <= log p(x).', source='vae-paper')
    add('elbo-learning', 'Which parts can we compute and differentiate?', 'prob-learning', 2, 90, 'elbo',
        'For each observed x, sample z from the encoder distribution, evaluate the decoder log likelihood and subtract KL(q||prior). Approximate the reconstruction expectation with K samples; K=1 is common. Our Gaussian KL is analytic. Average the negative bound across the minibatch and update both theta and phi. The exact true-posterior KL gap is generally unavailable. Holding theta fixed tightens the bound by changing q; updating theta also changes the evidence. Improving a bound does not guarantee every stochastic update improves true marginal likelihood. Beta=1 gives this ELBO; other beta weights define a different objective. This creates the need for the following reparameterization graph.',
        'Estimate the reconstruction expectation, compute the Gaussian prior KL, and differentiate the negative bound with respect to both networks.', source='vae-paper')
    add('gan-source', 'What probability does the discriminator predict?', 'prob-source', 3, 90, 'another-signal',
        'Define the training experiment before introducing its loss. Choose source S=1 or S=0 with equal probability. If S=1, draw x from the data distribution; otherwise draw z from the noise prior and set x=G(z). This mapping induces p_G over x even when its density is unavailable. Hide S and ask D to estimate P(S=1|x). This is source classification in a constructed mixture, not a general probability that an image is real or a density p_data(x). Equal source priors matter for the later D* formula.',
        'D(x) estimates the probability that x came from the data source in this balanced classification experiment.')
    add('gan-logloss', 'Where does the discriminator loss come from?', 'prob-logloss', 3, 90, 'discriminator',
        'Use the Bernoulli conditional model P(S|x)=D(x)^S(1-D(x))^(1-S). Its log likelihood is S log D+(1-S)log(1-D). Average over the balanced source experiment. The result is V/2, where V is the sum of real and generated expectations in the original GAN objective. Multiplying by two preserves the optimum, so the implementation minimizes L_D=-V. Keep this factor explicit: an average across all real and fake examples differs by two from a sum of the two class means.',
        'The discriminator objective is ordinary binary classification log likelihood on a balanced mixture of data and generated examples.')
    add('gan-bayes', 'What would the best discriminator say?', 'prob-bayes', 3, 120, 'gan-logloss',
        'Use exact illustrative categorical probabilities p_data=(.6,.3,.1), p_G=(.1,.3,.6). Bayes gives D*(x)=pi p_data(x)/(pi p_data(x)+(1-pi)p_G(x)); pi is the data-source prior. At pi=.5, D*=(6/7,.5,1/7). The middle bin has score .5 even though the full distributions differ. Change pi to show why an unbalanced sampling experiment changes the posterior. Under equal priors the odds D*/(1-D*) recover the density ratio p_data/p_G. In actual GANs those densities are unavailable; D learns from samples. The formula describes the population optimum, not every trained neural discriminator.',
        'At equal source priors, D*(x)=p_data(x)/(p_data(x)+p_G(x)); it compares the two densities at the same x.',
        slider('sourcePrior', 'data-source prior π', .1, .9, .5, .1))
    add('gan-density', 'How does the discriminator change as the model moves?', 'prob-density', 4, 120, 'gan-bayes',
        'Original distribution diagram inspired by Goodfellow Figure 1 and Lilian Weng’s probability discussion. Here p_data=N(0,1), z~N(0,1), G_mu(z)=z+mu, so p_G=N(mu,1) is analytically known. Reveal p_G, then the exact Bayes discriminator on a separate probability axis. D*(x)=sigmoid(mu^2/2-mu*x). Move mu toward zero and inspect a fixed x. At mu=0 the distributions agree and optimal D is .5 everywhere. A trained network constantly outputting .5 could instead be underfit, so its output alone is not proof of convergence. These are controlled distributions and an oracle D*, not neural training snapshots.',
        'The optimal discriminator follows the density ratio. When both distributions match, it cannot infer the source better than the source prior.',
        slider('modelMean', 'generator mean μ', -3, 3, 2, .1)+slider('probe', 'inspect x', -4, 4, 0, .1)+btn('match-density', 'Match the distributions'))
    add('gan-gradient', 'Why change the generator’s loss in practice?', 'prob-gradient', 3, 90, 'train-generator',
        'Write D(G(z))=sigmoid(a), where a is the discriminator logit. For minimax generator loss log(1-D), the derivative with respect to a is -D. For non-saturating loss -log D, it is D-1. At a=-4, D=.017986, so the two derivatives are -.017986 and -.982014. Both updates try to increase the data-source logit, but the latter avoids the small loss-to-logit derivative when D confidently rejects the sample. The complete generator gradient also includes the derivative of the discriminator logit with respect to its input and of G with respect to its parameters; the new objective does not guarantee useful gradients in every setting. D parameters stay fixed during this update.',
        'The non-saturating loss provides a larger loss-to-logit derivative when D strongly rejects a generated example.',
        slider('logit', 'discriminator logit a', -6, 6, -4, .25))
    add('gan-js', 'How is the ideal game connected to KL?', 'prob-js', 3, 90, 'gan-numbers',
        'Return to the exact three-bin distributions. Form m=(p_data+p_G)/2 and average KL(p_data||m) and KL(p_G||m): this is Jensen–Shannon divergence. Substitute the optimal discriminator into the classical minimax value to obtain V(G,D*)=-log 4+2 JS. Change the categorical model along a chosen interpolation toward the data distribution. JS reaches zero and V reaches -log 4. This is an analytical distribution path, not a learned G update. The result assumes optimal D and the original balanced minimax objective. Finite alternating updates with the non-saturating generator loss are not literally gradient descent on this JS expression.',
        'Under an optimal balanced discriminator, the classical minimax value differs from twice JS only by -log 4.',
        slider('match', 'move model toward data', 0, 1, 0, .05))

    # Insert complete detours without compressing the previously allocated demos.
    def append_with_children(s, output):
        output.append(s)
        for child in additions.get(s['id'], []):
            append_with_children(child, output)
    result=[]
    for s in slides: append_with_children(s, result)
    for s in result:
        s['readings'] = [dict(id=k, **RESOURCES[k]) for k in SLIDE_READINGS.get(s['id'], [])]
        if s['id']=='chapter-4':
            s['notes']='Chapter plan: 20 minutes. From the generative model to posterior inference, KL, Jensen’s inequality and a computable evidence bound.'
        if s['id']=='chapter-7':
            s['title']='Learning a distribution through classification'
            s['question']='Can source classification teach a generator a distribution?'
            s['body']='<div class="chapter-number">07</div>'+P(s['question'],'chapter-question')+P('From randomness to data','chapter-thread')
            s['notes']='Chapter plan: 20 minutes. Construct the source-classification experiment, derive its objective and Bayes optimum, then connect alternating updates, gradient choices and distribution coverage.'
        if s['id']=='backup-sources':
            s['body']='<div class="resource-columns">'+''.join('<div><h3>'+name+'</h3><ul>'+''.join(f'<li><a href="{RESOURCES[k]["url"]}">{RESOURCES[k]["title"]}</a></li>' for k in keys)+'</ul></div>' for name,keys in [('Explanations and course material',['lil-vae','lil-gan','jaan','nipun-vi','nipun-vae','nipun-notebook','mit']),('Papers and worked implementations',['vae-paper','gan-paper','flow-paper','tf-cvae','torch-sampling'])])+'</div>'+P('<a href="resources.html">All resources, descriptions and links back to lecture concepts</a>','caption')
            s['provenance']='Primary papers and teaching resources · full attribution in resources.html and SOURCES.md'
        if s['id']=='start':s['question']='What must a useful generator learn?'
        if s['id']=='beta-experiment':s['notes']='Train with reconstruction + β × prior KL. β=1 is the negative ELBO; β≠1 changes the objective. '+s['notes']
    assert sum(s['planned_seconds'] for s in result if not s['appendix'])==5700
    return result
