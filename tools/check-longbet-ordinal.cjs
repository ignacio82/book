// Serve docs/ locally first. Requires puppeteer-core and a Chrome executable.
// PUPPETEER_MODULE, CHROME_PATH, and BOOK_REVIEW_URL can override local defaults.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const puppeteer = require(process.env.PUPPETEER_MODULE || 'puppeteer-core');

(async () => {
  const root = path.resolve(__dirname, '..');
  const target = process.env.BOOK_REVIEW_URL || 'http://127.0.0.1:8769/longbet_ordinal.html';
  const output = path.join(root, 'cache/ordinal-review/browser');
  fs.mkdirSync(output, {recursive: true});
  const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  const report = {checkedAt: new Date().toISOString(),
    htmlSha256: sha(path.join(root, 'docs/longbet_ordinal.html')),
    sourceSha256: sha(path.join(root, 'longbet_ordinal.qmd')),
    thirdPartyWidgetsStubbed: ['umami.martinez.fyi', 'utteranc.es'], views: []};
  const browser = await puppeteer.launch({executablePath: process.env.CHROME_PATH,
    headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage']});
  try {
    for (const width of [1440, 768, 390, 320]) {
      const page = await browser.newPage();
      const errors = [];
      await page.setRequestInterception(true);
      page.on('request', r => {
        const host = new URL(r.url()).hostname;
        if (['umami.martinez.fyi', 'utteranc.es'].includes(host)) {
          r.respond({status: 200, contentType: 'application/javascript',
            headers: {'Access-Control-Allow-Origin': '*'}, body: ''});
        } else r.continue();
      });
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
      page.on('requestfailed', r => errors.push(`${r.url()}: ${r.failure()?.errorText}`));
      await page.setViewport({width, height: 1050, deviceScaleFactor: 1});
      await page.goto(target, {waitUntil: 'networkidle0', timeout: 60000});
      await page.evaluate(async () => {
        if (window.MathJax?.startup?.promise) await window.MathJax.startup.promise;
        await document.fonts.ready;
        document.querySelectorAll('main details').forEach(e => { e.open = true; });
      });
      const view = await page.evaluate(() => {
        const main = document.querySelector('main');
        const text = main.innerText.replace(/\s+/g, ' ');
        const ids = [...document.querySelectorAll('[id]')].map(e => e.id);
        const refs = [...document.querySelectorAll('a[href^="#"]')].map(e => e.getAttribute('href'));
        const imgs = [...main.querySelectorAll('img')];
        const localLinks = [...document.querySelectorAll('a[href]')]
          .map(e => e.href).filter(u => u.startsWith(location.origin + '/'));
        const mainRect = main.getBoundingClientRect();
        return {
          title: document.title, width: innerWidth, pageWidth: document.documentElement.scrollWidth,
          mathCount: main.querySelectorAll('mjx-container').length,
          mathErrors: [...main.querySelectorAll('mjx-merror,[data-mjx-error]')].map(e => e.textContent),
          missingFragments: refs.filter(h => h.length > 1 && !document.getElementById(decodeURIComponent(h.slice(1)))),
          duplicateIds: [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))],
          missingClaims: ['72.4%', '17 of 20', '93.9%', '20.35', '4.39', '3.95']
            .filter(s => !text.includes(s)),
          missingFigure: imgs.length !== 1 || imgs.some(e => !e.complete || !e.naturalWidth || !e.alt.trim()),
          cellErrors: [...main.querySelectorAll('.cell-output-error,.cell-output-stderr')].map(e => e.innerText),
          nonfiniteOutput: [...main.querySelectorAll('table,.cell-output-stdout')]
            .filter(e => /\b(NaN|Inf|NA)\b/.test(e.innerText)).map(e => e.innerText),
          unresolved: text.match(/\[@[^\]]+\]|`r [^`]+`/g) || [],
          overflow: [...main.querySelectorAll('p,table,img,h1,h2,h3,.math.inline')].filter(e => {
            const r = e.getBoundingClientRect();
            return r.width > 0 && (r.left < -2 || r.right > innerWidth + 2);
          }).map(e => ({tag: e.tagName, text: e.textContent.slice(0, 100)})),
          columnOverflow: [...main.querySelectorAll('table,img')].filter(e => {
            const r = e.getBoundingClientRect();
            return r.width > 0 && (r.left < mainRect.left - 2 || r.right > mainRect.right + 2);
          }).map(e => e.tagName),
          sidebarHasChapter: !!document.querySelector('#quarto-sidebar a[href$="longbet_ordinal.html"]'),
          localLinks: [...new Set(localLinks)],
          tables: [...main.querySelectorAll('table')].map(e => e.innerText)
        };
      });
      view.missingLocalFiles = view.localLinks.filter(u => {
        const file = decodeURIComponent(new URL(u).pathname).replace(/^\//, '') || 'index.html';
        return !fs.existsSync(path.join(root, 'docs', file));
      });
      delete view.localLinks;
      view.browserErrors = errors;
      report.views.push(view);
      await page.screenshot({path: path.join(output, `chapter-${width}.png`)});
      const fig = await page.$('main .quarto-figure');
      await fig.screenshot({path: path.join(output, `profiles-${width}.png`)});
      if (width === 1440) {
        fs.writeFileSync(path.join(output, 'rendered-text.txt'), await page.$eval('main', e => e.innerText));
        await page.click('#toc-from-rating-effects-to-a-rollout-probability');
        await page.waitForFunction(() => location.hash === '#from-rating-effects-to-a-rollout-probability');
        await page.screenshot({path: path.join(output, 'decision-section.png')});
        view.tocNavigation = true;
      }
      await page.close();
    }
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  }
  const failed = report.views.some(v => v.pageWidth > v.width + 2 || v.mathCount < 15 ||
    v.mathErrors.length || v.missingFragments.length || v.duplicateIds.length || v.missingClaims.length ||
    v.missingFigure || v.cellErrors.length || v.nonfiniteOutput.length || v.unresolved.length ||
    v.overflow.length || v.columnOverflow.length || !v.sidebarHasChapter || v.missingLocalFiles.length ||
    v.browserErrors.length);
  console.log(JSON.stringify(report, null, 2));
  if (report.htmlSha256 !== sha(path.join(root, 'docs/longbet_ordinal.html'))) {
    throw new Error('Rendered chapter changed during browser check');
  }
  process.exitCode = failed ? 1 : 0;
})().catch(e => { console.error(e); process.exitCode = 1; });
