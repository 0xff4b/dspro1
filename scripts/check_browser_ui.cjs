/* Run against a local preview. Requires Playwright + installed Chrome, Edge, WebKit.
 * NODE_PATH may point to an existing Playwright installation.
 */
const {chromium, webkit, devices} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const url = process.env.DEMO_URL || 'http://127.0.0.1:18501';
const testAddress = process.env.TEST_ADDRESS;
if (!testAddress) throw new Error('Set TEST_ADDRESS to a full Swiss address for the live lookup check.');
const output = path.resolve(process.env.UI_CHECK_OUTPUT || 'tmp/ui-check');
fs.mkdirSync(output, {recursive:true});
const cases = [
 {name:'chrome-desktop',engine:chromium,launch:{channel:'chrome'},context:{viewport:{width:1440,height:1000},colorScheme:'light'}},
 {name:'edge-desktop-dark-system',engine:chromium,launch:{channel:'msedge'},context:{viewport:{width:1366,height:900},colorScheme:'dark'}},
 {name:'android-chrome',engine:chromium,launch:{channel:'chrome'},context:{...devices['Pixel 7'],colorScheme:'dark'}},
 {name:'iphone-webkit',engine:webkit,launch:{},context:{...devices['iPhone 13'],colorScheme:'light',reducedMotion:'reduce'}},
];
async function imagesLoaded(page) {
 await page.waitForFunction(()=>Array.from(document.images).every(i=>i.complete && i.naturalWidth>0),{},{timeout:60000});
}
async function checkLayout(page) {
 const dimensions=await page.locator('[data-testid="stMain"]').evaluate(e=>({scroll:e.scrollWidth,width:e.clientWidth}));
 assert.ok(dimensions.scroll <= dimensions.width+2, 'Horizontal overflow: '+JSON.stringify(dimensions));
 const colors=await page.locator('.stApp').evaluate(e=>({bg:getComputedStyle(e).backgroundColor,text:getComputedStyle(e).color}));
 assert.equal(colors.bg,'rgb(255, 255, 255)','Poster theme must stay consistent with native controls');
}
(async()=>{
 await Promise.all(cases.filter(c=>!process.env.BROWSER_CASE || c.name===process.env.BROWSER_CASE).map(async(item)=> {
  const browser=await item.engine.launch({...item.launch,headless:true});
  const context=await browser.newContext(item.context);
  const page=await context.newPage();
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  page.setDefaultTimeout(60000);
  try {
   await page.goto(url,{waitUntil:'domcontentloaded'});
   const main=page.locator('[data-testid="stMain"]');
   await main.getByRole('combobox',{name:'Adresse',exact:true}).waitFor({timeout:120000});
   assert.ok(await main.getByRole('radio',{name:'Adresse suchen',exact:true}).isChecked(),'Address search must be the default');
   await page.locator('.demo-footer').waitFor();
   await page.locator('.stApp[data-test-script-state=notRunning]').waitFor();
   await page.evaluate(()=>document.fonts.ready);
   await imagesLoaded(page);
   await checkLayout(page);
   const touchButton=await main.getByRole('combobox',{name:'Adresse',exact:true}).boundingBox();
   assert.ok(touchButton.height>=44,'Touch target is too small');
   if(item.context.reducedMotion==='reduce'){
    assert.equal(await page.locator('.hero').evaluate(e=>getComputedStyle(e).animationName),'none');
   }
   await page.screenshot({path:path.join(output,item.name+'.jpg'),fullPage:true,type:'jpeg',quality:80,scale:'css'});
   await page.getByRole('link',{name:'Demo entdecken',exact:false}).click();
   const address=main.getByRole('combobox',{name:'Adresse',exact:true});
   assert.equal(await address.count(),1,'Exactly one address field');
   await address.fill(testAddress);
   const suggestion=main.getByRole('option').first();
   await suggestion.waitFor();
   await page.screenshot({path:path.join(output,item.name+'-address.jpg'),fullPage:true,type:'jpeg',quality:85,scale:'css'});
   if(['chrome-desktop','edge-desktop-dark-system','android-chrome','iphone-webkit'].includes(item.name)) {
    const label=await suggestion.innerText();
    if(item.context.isMobile) await suggestion.tap();
    else if(item.name==='chrome-desktop') {await address.press('ArrowDown'); await address.press('Enter');}
    else await suggestion.click();
    await main.getByRole('combobox',{name:'🏠 Wohnung auswählen',exact:true}).waitFor({timeout:120000});
    await main.locator('.kpi-value').first().waitFor({timeout:120000});
    assert.equal(await address.inputValue(),label,'Selected canonical address must persist');
    await address.fill('xy');
    await main.getByRole('combobox',{name:'🏠 Wohnung auswählen',exact:true}).waitFor({state:'detached'});
    assert.equal(await address.inputValue(),'xy','Editing must not restore the previous selection');
   } else { await address.press('Escape'); await address.fill(''); }
   await main.locator('[data-testid=stRadioOption]').filter({hasText:'Beispielwohnung'}).click();
   await main.getByRole('button',{name:'Miete schätzen',exact:true}).waitFor();
   const initial=await page.locator('.kpi-value').first().innerText();
   const cityButton=main.locator('.st-key-demo_city').getByRole('button',{name:'Open',exact:true});
   await cityButton.scrollIntoViewIfNeeded();
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
   await cityButton.click();
   await page.getByRole('option',{name:'Luzern',exact:true}).click();
   await main.getByRole('spinbutton',{name:'Wohnfläche (m²)',exact:true}).fill('100');
   await main.getByRole('button',{name:'Miete schätzen',exact:true}).click();
   await page.waitForFunction(old=>{const result=document.querySelector('.kpi-value');return result && result.textContent.trim()!==old.trim();},initial,{timeout:60000});
   await checkLayout(page);
   await page.screenshot({path:path.join(output,item.name+'-form.jpg'),fullPage:true,type:'jpeg',quality:85,scale:'css'});
   await main.locator('[data-testid=stRadioOption]').filter({hasText:'Modellgüte'}).click();
   await main.locator('[data-testid="stImage"]').nth(1).waitFor();
   await imagesLoaded(page);
   assert.ok(await main.locator('[data-testid="stImage"]').count()>=2,'Both analysis plots must render');
   await checkLayout(page);
   await main.locator('[data-testid=stRadioOption]').filter({hasText:'Projekt & Daten'}).click();
   await main.locator('.process-grid').waitFor();
   await page.locator('.stApp[data-test-script-state=notRunning]').waitFor();
   await main.locator('[data-testid="stImage"]').first().waitFor();
   await imagesLoaded(page);
   await checkLayout(page);
   await main.locator('[data-testid=stRadioOption]').filter({hasText:'Live-Demo'}).click();
   await main.locator('[data-testid=stRadioOption]').filter({hasText:'Adresse suchen'}).click();
   await main.getByRole('combobox',{name:'Adresse',exact:true}).waitFor();
   assert.equal(await main.getByRole('button',{name:'Adresse suchen',exact:true}).count(),0,'No separate search button');
   assert.equal(errors.length,0,errors.join('\n'));
   console.log('PASS '+item.name+' — images, forms, estimate, navigation, plots, width, touch targets');
  } catch(error) {
   await page.screenshot({path:path.join(output,item.name+'-failure.jpg'),fullPage:true,type:'jpeg',quality:80,scale:'css'}).catch(()=>{});
   console.error('FAIL '+item.name+' '+error.stack);
   process.exitCode=1;
  } finally {
   await context.close(); await browser.close();
  }
 }));
})();
