// Stamp each lecture page with content-hashed asset URLs, so a browser never runs a stale
// stylesheet or module after an edit: deck.css gets ?v=<hash>, and an import map sends every
// module import (deck.js, author.js, content.js, widgets.js, …) to its hashed URL.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const hash=file=>crypto.createHash('sha1').update(fs.readFileSync(file)).digest('hex').slice(0,10);
const shared=['deck.js','author.js'].map(f=>[`../shared/${f}`,hash(path.join(root,'shared',f))]);
const css=hash(path.join(root,'shared','deck.css'));
for(let n=1;n<=5;n++){
 const dir=path.join(root,`lecture${n}`),page=path.join(dir,'index.html');
 const local=fs.readdirSync(dir).filter(f=>/\.(m?js)$/.test(f)).map(f=>[`./${f}`,hash(path.join(dir,f))]);
 const imports=Object.fromEntries([...shared,...local].map(([k,h])=>[k,`${k}?v=${h}`]));
 let html=fs.readFileSync(page,'utf8');
 html=html.replace(/<script type="importmap">[\s\S]*?<\/script>/,'');
 html=html.replace(/href="\.\.\/shared\/deck\.css(\?v=[0-9a-f]+)?"/,`href="../shared/deck.css?v=${css}"`);
 html=html.replace(/<script type="module" src="\.\.\/shared\/deck\.js(\?v=[0-9a-f]+)?"><\/script>/,
  `<script type="importmap">${JSON.stringify({imports})}</script><script type="module" src="${imports['../shared/deck.js']}"></script>`);
 if(!html.includes('type="importmap"')||!html.includes(`deck.css?v=${css}`))throw new Error(`Could not stamp lecture${n}/index.html`);
 fs.writeFileSync(page,html);
}
console.log(`Stamped asset versions: deck.css ${css}, ${shared.map(([k,h])=>`${path.basename(k)} ${h}`).join(', ')}`);
