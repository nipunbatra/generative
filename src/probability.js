/* Original probability diagrams. Analytical distributions are distinct from saved training runs. */
const Probability = (() => {
 const {C,text:T,path:P,arrow:A,circle:Q,box:B,wrap:W}=Fig;
 const f=(x,n=3)=>(Math.abs(x)<.5*10**(-n)?0:x).toFixed(n).replace('-','−');
 const svg=(s,label)=>W(s.replace(/\b(p_data|p_G|E_q|E_data|E_G|L_D|z_K)\b/g, token=>{const [base,sub]=token.split('_');return `${base}<tspan baseline-shift="sub" font-size="0.72em">${sub}</tspan>`;}),label,1120,440);
 const normal=x=>Math.exp(-x*x/2)/Math.sqrt(2*Math.PI);
 const pData=[.6,.3,.1],pInitial=[.1,.3,.6];
 const kl=(p,q)=>p.reduce((v,a,i)=>v+(a? a*Math.log(a/q[i]):0),0);
 const line=(fn,x0,x1,y0,ys,n=220)=>'M'+Array.from({length:n+1},(_,i)=>`${x0+(x1-x0)*i/n},${y0-ys*fn(i/n)}`).join(' L');
 function reweight(b){
  const toy=DATA.pedagogy.toy,q=toy.states[0].q,joint=toy.prior.map((v,i)=>v*toy.likelihood[i]),w=joint.map((v,i)=>v/q[i]);
  let o=T(40,35,'Same fixed model. Choose q = (0.4, 0.3, 0.3).','gray',27,'start');
  [450,700,950].forEach((x,i)=>o+=T(x,95,['A','B','C'][i],'purple',30));
  [['joint p(x,z)',joint,'blue'],['proposal q(z | x)',q,'purple']].forEach(([name,values,c],r)=>{let y=150+r*62;o+=T(40,y,name,c,26,'start');values.forEach((v,i)=>o+=T(450+250*i,y,f(v,2),c,29));});
  if(b>=1){o+=P('M40,241 H1070','line',false,2)+T(40,286,'weight w = joint / q','ink',25,'start');w.forEach((v,i)=>o+=T(450+250*i,286,f(v),'ink',29));}
  if(b>=2)o+=T(560,357,'E_q[w] = 0.4 × 0.05 + 0.3 × 1.333… + 0.3 × 0.2','ink',27)+T(560,410,'E_q[w] = 0.48 = p(x)','teal',33);
  return svg(o,'Reweight three joint probabilities under a sampleable approximate posterior without changing the evidence');
 }
 function jensen(b){
  const toy=DATA.pedagogy.toy,q=toy.states[0].q,weights=toy.prior.map((v,i)=>v*toy.likelihood[i]/q[i]);
  const X=w=>80+w/1.45*530,Y=v=>160-63*v,avg=toy.evidence,lower=toy.states[0].elbo;
  let o=T(68,42,'log w','gray',25,'start')+P('M70,70 V400 H630','line',false,2);
  [-3,-2,-1,0].forEach(v=>o+=T(53,Y(v)+7,String(v).replace('-','−'),'gray',20,'end'));
  [.5,1].forEach(v=>o+=T(X(v),427,String(v),'gray',20));o+=T(648,410,'w','gray',25);
  o+=P(line(t=>Math.log(.025+1.4*t),X(.025),X(1.425),160,63),'gray',false,3);
  weights.forEach(w=>o+=Q(X(w),Y(Math.log(w)),6,'purple'));
  o+=Q(X(avg),Y(Math.log(avg)),8,'blue')+T(875,60,'Log after averaging','blue',29)+T(875,113,'log E_q[w] = log 0.48','blue',28)+T(875,170,'= −0.734','blue',36);
  if(b>=1){o+=P(`M${X(avg)},${Y(Math.log(avg))} V${Y(lower)}`,'coral',true,3)+Q(X(avg),Y(lower),8,'purple');o+=T(875,237,'Average after taking logs','purple',27)+T(875,286,'E_q[log w] = −1.595','purple',32);}
  if(b>=2)o+=T(875,365,'ELBO ≤ log p(x)','teal',35)+T(875,412,'Jensen’s inequality','ink',23);
  return svg(o,'Jensen inequality on the concave log curve: the log of the average lies above the average of logs');
 }
 function learning(b){
  let o=T(560,35,'A computable objective for each observed x','ink',30);
  o+=B(30,107,130,76,'x','blue',33)+A(170,145,245,145,'blue')+B(255,107,250,76,'qφ(z | x)','purple',31)+A(515,145,580,145,'purple')+B(590,107,150,76,'z₁, …, z_K','purple',27)+A(750,145,815,145,'purple')+B(825,107,265,76,'log pθ(x | z)','teal',28);
  o+=T(380,225,'encoder proposes','purple',23)+T(665,225,'K draws','purple',23)+T(958,225,'decoder scores x','teal',23);
  o+=T(560,310,'ELBO estimate = average log likelihood − KL(qφ(z | x) ∥ p(z))','ink',28);
  if(b)o+=T(560,378,'Minimise the negative bound over examples; update both φ and θ.','teal',27)+T(560,423,'Gaussian prior KL is exact here. The expectation uses samples.','gray',23);
  return svg(o,'Sample latent explanations from the encoder, score the observed image with the decoder, and subtract prior KL');
 }
 function source(b){
  let o=T(560,35,'Construct a balanced source-classification experiment','ink',30);
  o+=Q(140,205,52,'amber',.12)+T(140,217,'S','amber',38)+T(140,306,'choose source','gray',24);
  o+=A(182,169,333,115,'blue')+T(280,72,'½: data','blue',25)+B(345,80,265,75,'x ~ p_data','blue',30);
  o+=A(182,241,333,278,'purple')+T(280,340,'½: generator','purple',25)+B(345,240,265,75,'z ~ p(z), x = G(z)','purple',25);
  if(b>=1)o+=A(620,117,700,190,'blue')+A(620,277,700,220,'purple')+B(710,167,100,76,'x','ink',32)+T(765,299,'hide S','gray',25);
  if(b>=2)o+=A(820,205,905,205,'ink')+B(915,167,170,76,'D(x)','teal',33)+T(1000,299,'P(S = 1 | x)','teal',26);
  o+=T(560,415,'The generator induces p_G over x through its map G(z).','ink',27);
  return svg(o,'A fair source choice produces either a data observation or a generator sample; D predicts the hidden source');
 }
 function logloss(b){
  let o=T(560,48,'A Bernoulli model for the hidden source S','ink',30);
  o+=B(70,105,440,80,'S = 1: probability D(x)','blue',29)+B(610,105,440,80,'S = 0: probability 1 − D(x)','purple',27);
  if(b>=1)o+=T(560,258,'log P(S | x) = S log D(x) + (1 − S) log(1 − D(x))','ink',29);
  if(b>=2)o+=T(560,335,'V(G,D) = E_data[log D(x)] + E_G[log(1 − D(x))]','teal',29)+T(560,405,'Balanced average = V / 2. Train D by minimising L_D = −V.','ink',26);
  return svg(o,'Binary classification log likelihood averaged over the two sources yields the discriminator objective');
 }
 function bayes(s,b){
  const pi=s.sourcePrior??.5,ds=pData.map((v,i)=>pi*v/(pi*v+(1-pi)*pInitial[i]));
  let o=T(560,35,`Data-source prior π = ${f(pi,2)}; generator-source prior = ${f(1-pi,2)}`,'gray',26);
  [450,700,950].forEach((x,i)=>o+=T(x,94,['A','B','C'][i],'ink',28));
  [['p_data(x)',pData,'blue'],['p_G(x)',pInitial,'purple']].forEach(([label,vs,c],r)=>{let y=150+r*60;o+=T(45,y,label,c,28,'start');vs.forEach((v,i)=>o+=T(450+250*i,y,f(v,2),c,28));});
  if(b>=1){o+=P('M45,240 H1075','line',false,2)+T(45,287,'D*(x)', 'teal',30,'start');ds.forEach((v,i)=>o+=T(450+250*i,287,f(v),'teal',33));}
  if(b>=2)o+=T(560,357,'D*(x) = π p_data(x) / [π p_data(x) + (1 − π) p_G(x)]','ink',27)+T(560,418,Math.abs(pi-.5)<1e-8?'Equal priors: D* / (1 − D*) = p_data / p_G':'Changing source priors changes the posterior, even at the same x.','teal',25);
  return svg(o,'Exact Bayes discriminator for three categorical outcomes with adjustable source prior');
 }
 function density(s,b,el){
  const mu=s.manual?(s.modelMean??2):(b>=3?0:2),x=s.probe??0,X=v=>70+(v+5)*65,dp=normal(x),dg=normal(x-mu),d=dp/(dp+dg);
  if(!s.manual){const input=el.querySelector('[data-param=modelMean]');input.value=mu;el.querySelector('[data-value=modelMean]').value=f(mu,1);}
  let o=T(235,32,'p_data = N(0,1)','blue',26)+T(600,32,`p_G = N(${f(mu,1)},1)`,'purple',26);
  o+=P('M70,195 H720','line',false,2)+P(line(t=>normal(-5+10*t),70,720,195,285),'blue',false,3);
  if(b>=1)o+=P(line(t=>normal(-5+10*t-mu),70,720,195,285),'purple',false,3);
  o+=T(70,67,'density','gray',20,'start')+P(`M${X(x)},72 V195`,'gray',true,1.5)+Q(X(x),195-285*dp,5,'blue');
  if(b>=1)o+=Q(X(x),195-285*dg,5,'purple');
  o+=T(925,87,`Inspect x = ${f(x,1)}`,'ink',29)+T(925,138,`p_data(x) = ${f(dp)}`,'blue',27);
  if(b>=1)o+=T(925,187,`p_G(x) = ${f(dg)}`,'purple',27);
  if(b>=2){
   o+=T(395,246,'Optimal source probability D*(x)','teal',25)+P('M70,265 V400 H720','line',false,2)+P('M70,332.5 H720','line',true,1.5);
   [0,.5,1].forEach(v=>o+=T(53,400-135*v+7,String(v),'gray',19,'end'));
   o+=P(line(t=>1/(1+Math.exp(mu*(-5+10*t)-mu*mu/2)),70,720,400,135),'teal',false,3)+Q(X(x),400-135*d,7,'teal');
   [-4,-2,0,2,4].forEach(v=>o+=T(X(v),430,String(v).replace('-','−'),'gray',19));
   o+=T(925,292,`D*(x) = ${f(d)}`,'teal',37)+T(925,346,mu===0?'Same distributions':'Compare both densities','ink',25)+T(925,391,mu===0?'D* = 0.5 everywhere':'at the same x.','ink',25);
  }
  return svg(o,'Data and model densities on a shared axis above the Bayes-optimal discriminator; move the generator mean to match them');
 }
 function gradient(s,b){
  const a=s.logit??-4,d=1/(1+Math.exp(-a));
  let o=T(560,35,'Hold D’s weights fixed; keep its input-gradient path.','ink',28);
  o+=B(55,105,150,75,'G(z)','purple',30)+A(215,142,300,142,'purple')+B(310,105,150,75,'x','purple',30)+A(470,142,555,142,'purple')+B(565,105,205,75,`logit a = ${f(a,2)}`,'gray',25)+A(780,142,860,142,'gray')+B(870,105,225,75,`D = ${f(d)}`,'teal',29);
  o+=T(570,222,'D = sigmoid(a)','teal',25);
  if(b>=1){o+=T(55,295,'minimax: log(1 − D)','gray',27,'start')+T(600,295,'−D','gray',25)+P(`M935,286 H${935-250*d}`,'gray',false,12)+T(1080,295,f(-d),'gray',30,'end');}
  if(b>=2){o+=T(55,365,'non-saturating: −log D','teal',27,'start')+T(600,365,'D − 1','teal',25)+P(`M935,356 H${935-250*(1-d)}`,'teal',false,12)+T(1080,365,f(d-1),'teal',30,'end');o+=T(560,430,'These are ∂L/∂a. The full gradient also passes through D and G.','ink',25);}
  if(b>=1)o+=T(865,245,'∂L / ∂a','gray',22);
  return svg(o,'Generator loss derivatives with respect to the discriminator logit, comparing minimax and non-saturating objectives');
 }
 function js(s,b){
  const t=s.match??0,g=pInitial.map((v,i)=>(1-t)*v+t*pData[i]),m=pData.map((v,i)=>(v+g[i])/2),a=kl(pData,m),c=kl(g,m),j=(a+c)/2;
  let o=P('M80,105 V300 H660','line',false,2)+T(77,83,'mass','gray',20)+[0,.3,.6].map(v=>T(63,307-280*v,String(v),'gray',19,'end')).join('')+T(165,37,'p_data','blue',26)+T(370,37,'p_G','purple',26)+T(575,37,'mixture m','amber',26);
  [0,1,2].forEach(i=>{const x=110+i*200;[pData,g,m].forEach((v,k)=>{let cx=x+35*k;o+=`<rect x="${cx}" y="${300-280*v[i]}" width="28" height="${280*v[i]}" fill="${C[['blue','purple','amber'][k]]}"/>`;});o+=T(x+48,338,['A','B','C'][i],'ink',28);});
  o+=T(900,92,'m = (p_data + p_G) / 2','amber',27);
  if(b>=1)o+=T(900,169,`KL(p_data ∥ m) = ${f(a)}`,'blue',27)+T(900,223,`KL(p_G ∥ m) = ${f(c)}`,'purple',27);
  if(b>=2)o+=T(900,293,`JS = ½ (${f(a)} + ${f(c)})`,'teal',26)+T(900,342,`JS = ${f(j)} nats`,'teal',32)+T(560,418,`V(G,D*) = −log 4 + 2 JS = ${f(-Math.log(4)+2*j)}`,'ink',29);
  return svg(o,'Jensen-Shannon divergence averages two KL comparisons with the mixture; the optimal minimax value follows exactly');
 }
 function draw(el,s,b){
  const kind=el.dataset.kind;if(!kind.startsWith('prob-'))return false;
  let html='',read='';
  switch(kind){
   case 'prob-reweight':html=reweight(b);read='q must be positive wherever p(x,z) contributes to the evidence.';break;
   case 'prob-jensen':html=jensen(b);read='ELBO: evidence lower bound. Small dots: log weights; large purple dot: their q-weighted average.';break;
   case 'prob-learning':html=learning(b);read='K = 1 is common. β = 1 gives the ELBO; changing β changes the objective.';break;
   case 'prob-source':html=source(b);read='D predicts the source in this experiment. Its output is not the data density.';break;
   case 'prob-logloss':html=logloss(b);read='Expectations use data samples and generated samples. No evaluation of p_G(x) is required.';break;
   case 'prob-bayes':html=bayes(s,b);read='Known toy probabilities; a neural discriminator learns the comparison from samples.';break;
   case 'prob-density':html=density(s,b,el);read='Analytical Gμ(z) = z + μ and optimal D*. This is not a recorded neural-training run.';break;
   case 'prob-gradient':html=gradient(s,b);read='A larger loss-to-logit derivative does not guarantee a useful full generator gradient.';break;
   case 'prob-js':html=js(s,b);read='Optimal D and the classical minimax objective. Practical non-saturating updates differ.';break;
  }
  el.querySelector('.plot').innerHTML=html;el.querySelector('.readout').textContent=read;return true;
 }
 function action(el,s,a){if(a==='match-density'){s.modelMean=0;s.manual=true;el.querySelector('[data-param=modelMean]').value=0;el.querySelector('[data-value=modelMean]').value='0.00';}}
 return {draw,action};
})();
