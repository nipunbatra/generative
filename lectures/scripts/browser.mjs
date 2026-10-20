import puppeteer from 'puppeteer';
import fs from 'node:fs';
export const base=process.env.SERIES_URL||'http://127.0.0.1:8767';
export async function browser(){
 const executablePath=process.env.PUPPETEER_EXECUTABLE_PATH||['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome','/usr/bin/google-chrome','/usr/bin/chromium'].find(p=>fs.existsSync(p))||puppeteer.executablePath();
 return puppeteer.launch({headless:true,executablePath,args:['--no-sandbox']});
}
