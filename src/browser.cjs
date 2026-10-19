/* Use an installed browser when this checkout reuses node_modules without its browser cache. */
const fs=require('fs');
exports.launch=(puppeteer)=>{
 let bundled;try{bundled=puppeteer.executablePath()}catch{}
 const executablePath=[process.env.PUPPETEER_EXECUTABLE_PATH,bundled,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'].find(p=>p&&fs.existsSync(p));
 return puppeteer.launch({headless:true,...(executablePath?{executablePath}:{})});
};
