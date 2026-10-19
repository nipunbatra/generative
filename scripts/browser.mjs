import puppeteer from 'puppeteer';import fs from 'node:fs';
export async function browser(){const executablePath=process.env.PUPPETEER_EXECUTABLE_PATH||['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser'].find(p=>fs.existsSync(p))||puppeteer.executablePath();return puppeteer.launch({headless:true,executablePath,args:['--no-sandbox']});}
export const url=process.env.LECTURE_URL||'http://127.0.0.1:8766';
