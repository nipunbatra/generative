// Independent checks: the browser's formulas (math.mjs) against scripts/pixel.py, and the browser's smiley
// network (widgets.js) against PyTorch's outputs on fixed inputs. Writes qa/science-checks.json.
//   node scripts/check-science.mjs        (from lecture5/)
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dir=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
globalThis.matchMedia=()=>({matches:false});
const M=await import(path.join(dir,'math.mjs'));
const {toyNet}=await import(path.join(dir,'widgets.js'));
const px=JSON.parse(fs.readFileSync(path.join(dir,'data/pixel.json'))),W=JSON.parse(fs.readFileSync(path.join(dir,'data/toy-weights.json')));
const r=px.run,checks=[];
const check=(name,got,want,tol=6e-4)=>{const ok=Math.abs(got-want)<=tol*Math.max(1,Math.abs(want));checks.push({name,got:+got.toFixed(6),want,ok});};
const {LIN,TOY}=M,t=r.t;
check('alpha_bar[250]',LIN.ab[t],r.alpha_bar);
check('x_250 from the shortcut',Math.sqrt(LIN.ab[t])*r.x0+Math.sqrt(1-LIN.ab[t])*r.eps,r.xt);
check('P(ink | x_250)',M.pInk(r.xt,t,px.pi),r.post_ink);
check('E[x0 | x_250]',M.x0Hat(r.xt,t,px.pi),r.x0_hat);
check('E[eps | x_250]',M.epsStar(r.xt,t,px.pi),r.eps_star);
check('score at x_250',M.score(r.xt,t,px.pi),r.score);
check('mu tilde with x0 = 1',M.muTilde(r.xt,1,t),r.mu_true);
const R=M.reverseStep(r.xt,t,M.epsStar(r.xt,t,px.pi),r.z,LIN);
check('reverse mean from eps_hat',R.mu,r.mu_from_eps);
for(const row of px.reverse){const K=M.reverseKernel(0,500,row.s,px.pi);check(`reverse kernel 500→${row.s}: fit sd`,K.fitSd,row.fit_sd);check(`reverse kernel 500→${row.s}: bump width`,K.sd,row.sd);}
for(const [tt,v] of px.irreducible.filter(x=>x[0]>=100&&x[0]<=900))check(`irreducible loss t=${tt}`,M.irreducible(tt,px.pi),v,2e-3);
// the toy schedule: products by hand
[0.9,0.72,0.504,0.252,0.0504].forEach((v,k)=>check(`toy alpha_bar[${k+1}]`,TOY.ab[k+1],v,1e-9));
let x=1;for(let k=1;k<=5;k++)x=M.forwardStep(x,k,M.TOY_EPS[k],TOY);check('toy x_5',x,0.35948,1e-4);
check('toy x_2 equals the two-step example',M.forwardStep(M.forwardStep(1,1,.3,TOY),2,-.5,TOY),px.two_steps.x2);
// the browser network against PyTorch
for(const [name,spec,c] of [['uncond',W.uncond,null],['cond_mouth',W.cond,2],['cond_none',W.cond,3]]){
 const net=toyNet(spec);W.check.x.forEach(([a,b],i)=>{const out=net(a,b,W.check.t[i],c);check(`${name} network x=(${a},${b}) t=${W.check.t[i]}: eps_x`,out[0],W.check[name][i][0],1e-4);check(`${name} network x=(${a},${b}) t=${W.check.t[i]}: eps_y`,out[1],W.check[name][i][1],1e-4);});
}
const failed=checks.filter(c=>!c.ok);
fs.mkdirSync(path.join(dir,'qa'),{recursive:true});
fs.writeFileSync(path.join(dir,'qa/science-checks.json'),JSON.stringify({passed:!failed.length,count:checks.length,failed,checks},null,1)+'\n');
console.log(`${checks.length} checks, ${failed.length} failed`);for(const f of failed)console.log('FAILED',f);
if(failed.length)process.exit(1);
