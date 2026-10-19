/* Original diagrams for the modelling question, posterior inference and variational bound. */
const Pedagogy = (() => {
  const {C, text:T, path:P, arrow:A, circle:Q, box:B, wrap:W} = Fig;
  const f = (x,n=3) => Number(x).toFixed(n).replace('-', '−');
  const svg = (body,label) => W(body,label,1120,440);
  const muted = (body,on) => `<g opacity="${on?1:0.12}" aria-hidden="${!on}">${body}</g>`;
  const rect=(x,y,w,h,c='line',fill=false) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${fill?C[c]+'18':'none'}" stroke="${C[c]}" stroke-width="2"/>`;
  function digit(v,x,y,size=100){
    let s=''; for(let i=0;i<784;i++){const g=Math.round(255*(1-v[i]));s+=`<rect x="${x+i%28*size/28}" y="${y+Math.floor(i/28)*size/28}" width="${size/28+.1}" height="${size/28+.1}" fill="rgb(${g},${g},${g})"/>`;}
    return s;
  }
  const strip=(values,x,y,size=92,gap=23)=>values.map((v,i)=>digit(v,x+i*(size+gap),y,size)).join('');
  const curve=(mu,sigma,x,y,w,h,color='purple',lo=-4,hi=4)=>P('M'+Array.from({length:161},(_,i)=>{const z=lo+(hi-lo)*i/160;return [x+w*i/160,y-h*Fig.normal(z,mu,sigma)].join(',')}).join(' L'),color,false,3);
  function cover(){
    let o=T(60,54,'From randomness to data','gray',32,'start');
    o+=digit(DATA.evidence.observed_seven,60,127,190)+T(155,365,'an observed digit','blue',24);
    o+=A(295,220,500,220,'gray')+T(397,180,'learn its structure','gray',22);
    o+=rect(555,127,190,190,'purple')+T(650,253,'?','purple',96)+T(650,365,'another plausible example','purple',24);
    o+=T(1080,172,'What must','ink',31,'end')+T(1080,218,'the model','ink',31,'end')+T(1080,264,'learn?','ink',31,'end');
    return svg(o,'Observed digit beside an empty frame: the task is to generate another plausible example');
  }
  function recognize(b){
    let o=T(270,45,'Given an image','blue',28)+digit(DATA.evidence.observed_seven,65,115,155)+A(245,195,345,195,'blue')+T(425,216,'7','blue',70)+T(270,360,'Predict a label','blue',28);
    o+=P('M560,70 V370','line',false,2);
    o+=muted(T(850,45,'Given a collection of examples','purple',28)+T(680,205,'learn','purple',30)+A(745,195,810,195,'purple')+rect(840,115,155,155,'purple')+T(918,218,'?','purple',67)+T(850,360,'Produce a possible image','purple',28),b>0);
    return svg(o,'Recognition maps a supplied image to a label; generation produces a possible image');
  }
  function completion(b){
    const top=(x,y)=>P(`M${x-44},${y+36} C${x-52},${y-25} ${x+55},${y-25} ${x+44},${y+36}`,'blue',false,10);
    let o=T(180,45,'Only part is visible','blue',27)+top(180,140)+`<rect x="94" y="177" width="172" height="140" fill="${C.line}"/>`+T(180,361,'What could be below?','gray',24);
    o+=A(300,215,415,215,'gray');
    if(b){
      [550,765,980].forEach((x,i)=>{
        o+=top(x,140);
        const paths=[`M${x-44},176 C${x-65},340 ${x+65},340 ${x+44},176`,`M${x-44},176 C${x-20},220 ${x+25},220 ${x+44},176 M${x-20},208 C${x-95},323 ${x+95},323 ${x+20},208`,`M${x-44},176 C${x-18},225 ${x+25},220 ${x+44},176 L${x+30},307`];
        o+=P(paths[i],'purple',false,10);
      });
      o+=T(765,361,'Several plausible completions','purple',27);
    }else o+=T(765,225,'Predict more than one answer.','purple',31);
    o+=T(560,425,'A distribution can represent several possible answers.','ink',27);
    return svg(o,'Original partial digit sketch with alternative zero, eight and nine completions');
  }
  function baselines(b){
    const e=DATA.pedagogy.opening;
    let o=T(35,29,'Replay training images','blue',27,'start')+strip(e.replay,35,55,88,44)+T(1080,29,'A sampler that remembers','gray',23,'end');
    o+=T(35,220,'Sample each pixel independently','purple',27,'start');
    if(b>=1)o+=strip(e.independent,35,245,88,44);else o+=T(560,305,'Predict what the second generator will draw.','purple',30);
    if(b>=2)o+=T(560,410,'Matching pixel frequencies does not capture how strokes fit together.','ink',27);
    return svg(o,'Measured training-image replay and independent-pixel samples fitted to the same MNIST training set');
  }
  function success(b){
    const vals=DATA.evidence.models.vae16.prior;
    let o=strip(vals,35,85,112,22)+T(560,40,'Actual saved VAE-16 decoder probabilities','purple',26);
    if(b)o+=T(180,303,'Plausibility','teal',29)+T(560,303,'Coverage','blue',29)+T(940,303,'Generalization','purple',29)+T(180,350,'Does each example fit?','gray',22)+T(560,350,'Do we capture variety?','gray',22)+T(940,350,'Did we learn beyond replay?','gray',22);
    else o+=T(560,320,'What works? What still looks wrong?','ink',33);
    o+=T(560,420,'A small sample panel cannot answer all three questions.','gray',24);
    return svg(o,'Actual VAE samples and three distinct criteria for evaluating a generator');
  }
  function directions(b){
    let o=Q(200,205,54,'purple',.09)+`<circle cx="200" cy="205" r="54" fill="none" stroke="${C.purple}" stroke-width="3"/>`+T(200,217,'z','purple',40)+digit(DATA.evidence.observed_seven,860,145,120)+T(920,305,'observed x','blue',28);
    o+=T(200,305,b>0?'Which z explains x?':'Draw from p(z)','purple',27);
    o+=A(280,160,830,160,'teal')+T(560,120,'Generation: pθ(x | z)','teal',30);
    if(b>=1)o+=A(830,250,280,250,'purple',true)+T(560,300,'Inference: pθ(z | x)','purple',30);
    if(b>=2)o+=T(560,383,'The reverse question needs a distribution over explanations.','ink',27)+T(560,425,'An inference network is not an inverse decoder.','gray',23);
    return svg(o,'One latent-data model with distinct generation and posterior-inference directions');
  }
  function evidence(b){
    const t=DATA.pedagogy.toy;let o='';const xs=[470,710,950];
    o+=T(40,45,'A tiny model with three latent choices','gray',25,'start');
    ['z = A','z = B','z = C'].forEach((s,i)=>o+=T(xs[i],100,s,'purple',30));
    [['prior p(z)',t.prior,'amber'],['likelihood p(x | z)',t.likelihood,'teal']].forEach(([name,vals,c],r)=>{const y=157+r*62;o+=T(40,y,name,c,26,'start');vals.forEach((v,i)=>o+=T(xs[i],y,f(v,2),c,29));});
    if(b>=1){o+=P('M40,248 H1060','line',false,2)+T(40,292,'joint contribution', 'ink',26,'start');xs.forEach((x,i)=>o+=T(x,292,f(t.prior[i]*t.likelihood[i],2),'ink',29));o+=T(560,357,'p(x) = 0.02 + 0.40 + 0.06 = 0.48','blue',31);}
    if(b>=2)o+=T(560,417,'Continuous z: replace the sum with ∫ pθ(x | z) p(z) dz.','purple',27);
    return svg(o,'Exact marginalization over three latent explanations; continuous latent variables require integration');
  }
  function amortized(b){
    const obs=DATA.evidence.observed_seven;let o='';
    if(b===0){o+=T(560,60,'The model defines the true posterior','purple',30)+T(560,151,'pθ(z | x) = pθ(x | z) p(z) / pθ(x)','ink',37)+T(560,245,'The denominator integrates over all latent explanations.','gray',27)+T(560,332,'For a neural decoder, exact inference is generally intractable.','coral',26);}
    else{
      [0,1,2].forEach(i=>{const y=45+i*118;const v=i===0?obs:DATA.pedagogy.opening.replay[i];o+=digit(v,35,y,85)+A(135,y+42,290,205,'blue');});
      o+=B(300,155,265,100,'shared encoder φ','teal',30)+A(580,205,650,205,'teal');
      o+=T(850,60,'qφ(z | x)','purple',35)+T(850,105,'A tractable approximation','gray',24);
      [0,1,2].forEach(i=>{const y=195+i*82;o+=P(`M680,${y} H1060`,'line',false,1.5)+curve(-1+i,.65,680,y,380,65,'purple')+T(655,y-25,String(i+1),'blue',20);});
      o+=T(430,365,'New image → new μ and σ','amber',25);
      if(b>=2)o+=T(560,429,'Amortized inference: learn one function instead of fitting each image from scratch.','ink',24);
    }
    return svg(o,'A shared encoder predicts a separate approximate posterior for each observed image');
  }
  function kl(s,b){
    const q=s.qmass??.8,p=s.pmass??.5,qs=[q,1-q],ps=[p,1-p];const terms=qs.map((v,i)=>v*Math.log(v/ps[i]));
    let o=T(280,35,'Outcomes are drawn from q','purple',27)+T(850,35,'Score them using p','amber',27);
    for(let j=0;j<2;j++){
      const x=185+j*270;
      o+=`<rect x="${x}" y="${255-130*qs[j]}" width="65" height="${130*qs[j]}" fill="${C.purple}"/><rect x="${x+76}" y="${255-130*ps[j]}" width="65" height="${130*ps[j]}" fill="${C.amber}"/>`;
      o+=T(x+32,287,`q=${f(qs[j],2)}`,'purple',21)+T(x+108,315,`p=${f(ps[j],2)}`,'amber',21)+T(x+70,99,j?'B':'A','ink',28);
    }
    if(b>=1){o+=T(850,116,'extra log loss at each outcome','gray',23)+T(850,171,`A: log(q/p) = ${f(Math.log(q/p))}`,'ink',26)+T(850,227,`B: log(q/p) = ${f(Math.log((1-q)/(1-p)))}`,'ink',26);}
    if(b>=2)o+=T(560,367,`${f(q,2)} × (${f(Math.log(q/p))}) + ${f(1-q,2)} × (${f(Math.log((1-q)/(1-p)))})`,'ink',28)+T(560,413,`KL(q ∥ p) = ${f(terms[0]+terms[1])} nats`,'teal',32);
    else o+=T(560,405,b?'Average these values under q.':'Compare the two probability assignments.','ink',29);
    return svg(o,'Discrete KL as expected extra log loss, with editable probabilities and a worked sum');
  }
  function direction(b){
    let o=T(560,50,'q = (0.8, 0.2)        p = (0.5, 0.5)','ink',31);
    o+=T(270,140,'Draw from q','purple',30)+T(270,210,'0.8 log(0.8/0.5)','ink',27)+T(270,253,'+ 0.2 log(0.2/0.5)','ink',27)+T(270,320,'0.193 nats','purple',39);
    o+=P('M560,105 V340','line',false,2);
    if(b)o+=T(850,140,'Draw from p','amber',30)+T(850,210,'0.5 log(0.5/0.8)','ink',27)+T(850,253,'+ 0.5 log(0.5/0.2)','ink',27)+T(850,320,'0.223 nats','amber',39);
    else o+=T(850,210,'What happens if','gray',30)+T(850,258,'we swap p and q?','gray',30);
    o+=T(560,415,'The first distribution determines the averaging weights. KL is not symmetric.','ink',25);
    return svg(o,'KL asymmetry demonstrated by reversing the averaging distribution');
  }
  function klModes(s,b){
    const wide=s.klShape===undefined?b===0:s.klShape==='wide',mu=wide?0:-2,sigma=wide?2.1:.65;
    const p=x=>.5*Fig.normal(x,-2,.55)+.5*Fig.normal(x,2,.55),q=x=>Fig.normal(x,mu,sigma);
    let forward=0,reverse=0;const step=.005;
    for(let x=-12+step/2;x<12;x+=step){let a=p(x),c=q(x);forward+=a*Math.log(a/c)*step;reverse+=c*Math.log(c/a)*step;}
    let o=T(35,32,'One Gaussian approximation to two separated groups','ink',28,'start');
    const plot=(fn,c)=>P('M'+Array.from({length:241},(_,i)=>{let x=-5+10*i/240;return [70+620*i/240,310-270*fn(x)].join(',')}).join(' L'),c,false,4);
    o+=P('M70,310 H690','gray',false,2)+plot(p,'blue')+plot(q,'purple');
    [-4,-2,0,2,4].forEach(x=>o+=T(380+62*x,342,x,'gray',21));
    o+=T(190,80,'target p','blue',25)+T(505,80,'approximation q','purple',25);
    o+=T(915,130,'Average under p','blue',26)+T(915,180,`KL(p ∥ q) = ${f(forward)}`,'blue',30)+T(915,255,'Average under q','purple',26)+T(915,305,`KL(q ∥ p) = ${f(reverse)}`,'purple',30);
    o+=T(560,413,wide?'Covering both groups also puts q mass in the low-density gap.':'Concentrating on one group leaves the other group poorly covered.','ink',25);
    return svg(o,'Two candidate Gaussian approximations evaluated in both KL directions by numerical integration');
  }

  function bound(s,b){
    const t=DATA.pedagogy.toy,idx=s.manual?s.fit:b,k=t.states[idx],logp=t.log_evidence;let o='';
    o+=T(270,30,'Same fixed generative model','gray',26)+T(270,76,'p(x) = 0.48','blue',31);
    [0,1,2].forEach(i=>{let x=82+i*132;o+=`<rect x="${x}" y="${305-200*t.posterior[i]}" width="42" height="${200*t.posterior[i]}" fill="${C.blue}70"/><rect x="${x+49}" y="${305-200*k.q[i]}" width="42" height="${200*k.q[i]}" fill="${C.purple}"/>`+T(x+45,337,['A','B','C'][i],'gray',23);});
    o+=T(270,384,'blue: true posterior     purple: q','gray',21);
    const x0=635,x1=800,yE=108,yB=108+k.gap*180;
    o+=P(`M${x0},${yE} H${x1}`,'blue',false,4)+T(1080,yE+8,`log p(x) = ${f(logp)}`,'blue',25,'end');
    o+=P(`M${x0},${yB} H${x1}`,'purple',false,4);
    if(k.gap>.00001){o+=P(`M610,${yE} V${yB}`,'coral',false,3)+T(955,280,`gap = ${f(k.gap)} nats`,'coral',27)+T(955,321,'KL(q ∥ posterior)','coral',24);}
    else o+=T(850,272,'q equals the true posterior','teal',27)+T(850,315,'The gap is zero.','teal',27);
    o+=T(850,365,`ELBO = ${f(k.elbo)}`,'purple',30);
    o+=T(560,431,'log p(x) = ELBO + KL(q ∥ true posterior)','ink',29);
    return svg(o,'Exact evidence lower bound for a three-state latent model; improving q closes the posterior KL gap');
  }
  function twoKL(b){
    let o=T(40,45,'The gap: a comparison we cannot usually compute','gray',27,'start');
    o+=T(560,115,'log pθ(x) − ELBO = KL(qφ(z | x) ∥ pθ(z | x))','purple',32);
    o+=T(920,163,'true posterior','blue',24);
    if(b){o+=P('M40,214 H1080','line',false,2)+T(40,261,'The objective: terms we can estimate or compute','gray',27,'start')+T(560,331,'ELBO = E q [log pθ(x | z)] − KL(qφ(z | x) ∥ p(z))','ink',31)+T(385,386,'reconstruction likelihood','teal',24)+T(993,386,'prior','amber',24);}
    return svg(o,'Posterior KL is the gap; prior KL is part of the computable ELBO');
  }
  function autograd(b){
    const rows=[['Normal(mu, std).sample()','no pathwise autograd connection','gray'],['mu + std * eps','explicit differentiable path','purple'],['Normal(mu, std).rsample()','the reparameterized interface','teal']];
    let o=T(40,38,'Same distribution. Different autograd behavior.','ink',29,'start');
    rows.forEach(([code,label,c],i)=>{if(i<=b){const y=100+i*100;o+=`<text x="40" y="${y}" fill="${C[c]}" font-size="28" font-family="Menlo,monospace">${code}</text>`+T(1080,y,label,c,25,'end')+P(`M40,${y+30} H1080`,'line',false,1);}});
    o+=T(560,421,'For z = μ + σε:  ∂z/∂μ = 1,  ∂z/∂σ = ε. Keep ε fixed in this backward pass.','ink',24);
    return svg(o,'Verified PyTorch sampling and reparameterized-sampling interfaces tied to the chain rule');
  }
  function beta(s,b){
    const idx=s.betaIndex??b,row=DATA.pedagogy.beta[idx],v=row.beta;let o='';
    o+=T(35,32,`β = ${v} · separately trained 2D VAE`,'purple',28,'start');
    o+=T(1080,32,`Test recon ${f(row.test_reconstruction_nats,1)} · KL ${f(row.test_kl_nats,2)} nats`,'teal',22,'end');
    o+=T(35,86,'Held-out reconstructions','gray',23,'start')+strip(row.recon,35,110,76,10);
    o+=T(35,252,'Same prior vectors → decoder means','gray',23,'start')+strip(row.prior,35,278,76,10);
    const range=26,scale=125/range,cx=935,cy=235;
    o+=T(cx,86,'600 posterior means','purple',23)+P(`M810,${cy} H1060 M${cx},110 V360`,'line',false,1.5);
    o+=`<circle cx="${cx}" cy="${cy}" r="${2*scale}" fill="${C.amber}12" stroke="${C.amber}" stroke-width="2"/>`;
    row.mu.forEach(p=>o+=Q(cx+p[0]*scale,cy-p[1]*scale,2,'purple',.45));
    o+=T(810,386,'−26','gray',18)+T(1060,386,'26','gray',18)+T(cx,386,'z₁','gray',20)+T(cx+20,132,'z₂','gray',20)+T(cx,420,'amber: prior radius 2','amber',20);
    o+=T(365,420,'Same inputs, seeds, data order and training budget.','ink',23);
    return svg(o,'Controlled beta sweep with measured reconstruction, prior outputs and posterior means on identical axes');
  }

  function draw(el,s,b){
    const kind=el.dataset.kind;if(!kind.startsWith('ped-'))return false;
    let html='',read='';
    switch(kind){
      case 'ped-cover':html=cover();break;
      case 'ped-recognize':html=recognize(b);break;
      case 'ped-completion':html=completion(b);read='Application preview: these are authored sketches, not model-generated completions.';break;
      case 'ped-baselines':html=baselines(b);read=DATA.pedagogy.opening.protocol;break;
      case 'ped-success':html=success(b);read='Saved VAE-16 probabilities; fixed seed, no sample selection.';break;
      case 'ped-directions':html=directions(b);break;
      case 'ped-evidence':html=evidence(b);read='Exact finite example; neural latent models generally need approximate inference.';break;
      case 'ped-amortized':html=amortized(b);read='Approximate posterior qφ(z | x), not the true posterior.';break;
      case 'ped-kl':html=kl(s,b);read='Natural logarithms → nats. Individual contributions can be negative; total KL ≥ 0.';break;
      case 'ped-kl-direction':html=direction(b);read='If p assigns zero probability where q has positive mass, KL(q ∥ p) is infinite.';break;
      case 'ped-kl-modes':html=klModes(s,b);read='Two illustrative candidate fits, not optimized solutions. Both scores use numerical quadrature.';break;
      case 'ped-bound':if(!s.manual){el.querySelector('input').value=b;el.querySelector('output').value=b;}html=bound(s,b);read='Exact toy calculation. θ is fixed here; changing θ changes the evidence too.';break;
      case 'ped-two-kls':html=twoKL(b);read='The prior p(z) and the posterior pθ(z | x) are different distributions.';break;
      case 'ped-autograd':html=autograd(b);read='PyTorch behavior verified locally. Other stochastic-gradient estimators also exist.';break;
      case 'ped-beta':html=beta(s,b);read='12k training images · 12 epochs · one seed. β = 0 still samples a stochastic latent.';el.querySelectorAll('button').forEach((e,i)=>e.setAttribute('aria-pressed',String(i===(s.betaIndex??b))));break;
      default:return false;
    }
    el.querySelector('.plot').innerHTML=html;el.querySelector('.readout').textContent=read;return true;
  }
  function action(el,s,a){if(a==='kl-wide')s.klShape='wide';if(a==='kl-narrow')s.klShape='narrow';if(el.dataset.kind==='ped-beta'&&a.startsWith('beta-'))s.betaIndex={'beta-0':0,'beta-1':1,'beta-4':2}[a];}
  return {draw,action};
})();
