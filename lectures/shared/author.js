export const palette={data:'#245EDB',q:'#8B2CDE',p:'#0F766E',noise:'#AA4E08',loss:'#BE123C',d:'#147737',ink:'#14171F',paper:'#FAFAFB'};
export const esc=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
export const m=tex=>`<span class="math" data-tex="${esc(tex)}">${esc(tex)}</span>`;
export const eq=tex=>`<div class="formula-scroll"><div class="math display" data-display="true" data-tex="${esc(tex)}">${esc(tex)}</div></div>`;
export const at=(build,html)=>`<div data-at="${build}">${html}</div>`;
export const only=(builds,html)=>`<div data-only="${builds}">${html}</div>`;
export const cols=(left,right,ratio='1fr 1fr')=>`<div class="columns" style="grid-template-columns:${esc(ratio)}"><div>${left}</div><div>${right}</div></div>`;
export const code=text=>`<pre class="code"><code>${esc(text)}</code></pre>`;
export const image=(src,alt,caption='')=>`<figure class="evidence"><img src="${esc(src)}" alt="${esc(alt)}" loading="eager">${caption?`<figcaption>${caption}</figcaption>`:''}</figure>`;
export const table=(headers,rows)=>`<div class="table-scroll"><table><thead><tr>${headers.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(x=>`<td>${x}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
let figureId=0;
function col(role){return palette[role]||role||palette.ink;}
export function flow(nodes,edges,options={}){
  const width=options.width||1000,height=options.height||330,id=`flow-${++figureId}`;
  const map=Object.fromEntries(nodes.map(n=>[n.id,{w:150,h:75,...n}]));
  const point=(n,side)=>({left:[n.x,n.y+n.h/2],right:[n.x+n.w,n.y+n.h/2],top:[n.x+n.w/2,n.y],bottom:[n.x+n.w/2,n.y+n.h]})[side];
  const lines=edges.map((e,i)=>{
    const a=map[e.from],b=map[e.to];if(!a||!b)throw new Error(`Unknown flow node ${e.from}→${e.to}`);
    const [x1,y1]=point(a,e.fromSide||'right'),[x2,y2]=point(b,e.toSide||'left');
    const stroke=col(e.role||'ink'),marker=`${id}-arrow-${i}`;
    const vertical=['top','bottom'].includes(e.fromSide);
    const outside=e.fromSide==='top'?Math.min(y1,y2)-70:Math.max(y1,y2)+70;
    const sameSide=vertical&&e.fromSide===e.toSide;
    const path=sameSide?`M${x1},${y1} C${x1},${outside} ${x2},${outside} ${x2},${y2}`:vertical?`M${x1},${y1} C${x1},${(y1+y2)/2} ${x2},${(y1+y2)/2} ${x2},${y2}`:`M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`;
    const labelY=sameSide?(e.fromSide==='top'?outside+6:outside+10):(y1+y2)/2-13;
    return `<defs><marker id="${marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10Z" fill="${stroke}"/></marker></defs><path d="${path}" fill="none" stroke="${stroke}" stroke-width="2.5" ${e.dashed?'stroke-dasharray="7 5"':''} marker-end="url(#${marker})"/>${e.label?`<text x="${(x1+x2)/2}" y="${labelY}" text-anchor="middle" fill="${stroke}" font-size="23" style="paint-order:stroke;stroke:${palette.paper};stroke-width:8px;stroke-linejoin:round">${esc(e.label)}</text>`:''}`;
  }).join('');
  const shapes=Object.values(map).map(n=>{
    const stroke=col(n.role||'ink');
    const shape=n.shape==='circle'?`<ellipse cx="${n.x+n.w/2}" cy="${n.y+n.h/2}" rx="${n.w/2}" ry="${n.h/2}" fill="${palette.paper}" stroke="${stroke}" stroke-width="2.5"/>`:`<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="4" fill="${palette.paper}" stroke="${stroke}" stroke-width="2.5"/>`;
    const limit=Math.max(7,Math.floor((n.w-20)/13.5));
    const labelLines=String(n.label??'').split('\n').flatMap(line=>{
      const lines=[''];for(const word of line.split(' ')){const last=lines.length-1;if(lines[last]&&(lines[last]+' '+word).length>limit)lines.push(word);else lines[last]+=(lines[last]?' ':'')+word;}return lines;
    });
    const textHeight=labelLines.length*29+(n.sub?26:0),firstY=n.y+(n.h-textHeight)/2+23;
    return `${shape}${labelLines.map((s,j)=>`<text x="${n.x+n.w/2}" y="${firstY+j*29}" text-anchor="middle" fill="${stroke}" font-size="27">${esc(s)}</text>`).join('')}${n.sub?`<text x="${n.x+n.w/2}" y="${firstY+labelLines.length*29-3}" text-anchor="middle" fill="#68717B" font-size="22">${esc(n.sub)}</text>`:''}`;
  }).join('');
  return `<div class="diagram-scroll"><svg class="diagram" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(options.label||'Computation diagram')}">${lines}${shapes}</svg></div>`;
}
export function plot(series,options={}){
  const W=options.width||1000,H=options.height||360,L=75,R=25,T=58,B=60;
  const allX=series.flatMap(s=>s.x),allY=series.flatMap(s=>s.y);
  const xmin=options.xMin??Math.min(...allX),xmax=options.xMax??Math.max(...allX);
  const ymin=options.yMin??0,ymax=options.yMax??Math.max(...allY)*1.12;
  if(![xmin,xmax,ymin,ymax].every(Number.isFinite)||xmin===xmax||ymin===ymax)throw new Error('Invalid plot bounds');
  const x=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),y=v=>H-B-(v-ymin)/(ymax-ymin)*(H-T-B);
  const tx=Array.isArray(options.ticksX)?options.ticksX:Array.from({length:5},(_,i)=>xmin+(xmax-xmin)*i/4);
  const ty=Array.isArray(options.ticksY)?options.ticksY:[ymin,ymax/2,ymax];
  const fmt=v=>Math.abs(v)>=100?String(Math.round(v)):String(Number(v.toFixed(2)));
  let svg=`<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(options.label||'Calculated values on fixed axes')}">`;
  svg+=`<line x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}" stroke="#AAB3BC"/><line x1="${L}" y1="${T}" x2="${L}" y2="${H-B}" stroke="#AAB3BC"/>`;
  for(const v of tx)svg+=`<text x="${x(v)}" y="${H-B+27}" text-anchor="middle" fill="#68717B" font-size="22">${fmt(v)}</text>`;
  for(const v of ty)svg+=`<text x="${L-10}" y="${y(v)+7}" text-anchor="end" fill="#68717B" font-size="22">${fmt(v)}</text>`;
  let legendX=L;
  for(const s of series){
    if(s.x.length!==s.y.length)throw new Error('Plot coordinate lengths differ');
    const stroke=col(s.role||'p');let path='',started=false;
    s.x.forEach((v,i)=>{if(!Number.isFinite(v)||!Number.isFinite(s.y[i])){started=false;return;}path+=`${started?'L':'M'}${x(v).toFixed(2)},${y(s.y[i]).toFixed(2)} `;started=true;});
    svg+=`<path d="${path}" fill="none" stroke="${stroke}" stroke-width="3" ${s.dashed?'stroke-dasharray="8 5"':''}/>`;
    svg+=`<line x1="${legendX}" y1="22" x2="${legendX+28}" y2="22" stroke="${stroke}" stroke-width="3" ${s.dashed?'stroke-dasharray="7 4"':''}/><text x="${legendX+37}" y="29" fill="${stroke}" font-size="23">${esc(s.label||'')}</text>`;
    legendX+=50+String(s.label||'').length*13;
  }
  svg+=`<text x="${W-R}" y="${H-12}" text-anchor="end" fill="#68717B" font-size="23">${esc(options.xLabel||'')}</text><text x="${L}" y="${T-10}" fill="#68717B" font-size="22">${esc(options.yLabel||'')}</text></svg>`;
  return `<div class="diagram-scroll">${svg}</div>`;
}
