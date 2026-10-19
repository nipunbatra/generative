import fs from 'node:fs';
import {content,titles} from '../src/content.mjs';
import {teaching} from '../src/teaching-cues.mjs';
const file='src/scene-manifest.json',manifest=JSON.parse(fs.readFileSync(file));
for(const s of manifest){s.title=titles[s.number];s.builds=content[s.number].map(([caption,latex],i)=>({order:i+1,caption,latex}));Object.assign(s,{narration:teaching[s.number].narration,question:teaching[s.number].question,answer:teaching[s.number].answer,speaking_cue:teaching[s.number].say,takeaway:content[s.number].at(-1)[0]});}
fs.writeFileSync(file,JSON.stringify(manifest,null,2));
