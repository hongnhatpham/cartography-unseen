import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright';
import { uiMachine } from './ui-fixture.ts';

// Real browser focus is the contract here. DOM string assertions cannot detect
// replacement of the active link during a background refresh.
const server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', 'preview', '--host', '127.0.0.1', '--port', '4179', '--strictPort'], { stdio: 'pipe', windowsHide: true });
let browser;
try {
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('UI preview did not start')), 10_000);
    server.stdout.on('data', chunk => {
      if (chunk.toString().includes('127.0.0.1:4179')) { clearTimeout(timeout); resolve(); }
    });
    server.once('error', reject);
    server.once('exit', code => { clearTimeout(timeout); reject(new Error(`UI preview exited ${code}`)); });
  });
  browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {}) });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.clock.install();
  let state = 'auth';
  let requests = 0;
  let seriesRequests = 0;
  await page.route('**/api/v1/**', route => {
    requests++;
    if (state !== 'healthy') return route.fulfill({ status: state === 'auth' ? 401 : 503, json: { error: 'Test response' } });
    const now = Date.now();
    const machine = uiMachine(now);
    const url = new URL(route.request().url());
    if (url.pathname.endsWith('/machines')) return route.fulfill({ json: { serverTime: now, machines: [machine], nextCursor: null } });
    if (url.pathname.endsWith('/series')) {
      seriesRequests++;
      return route.fulfill({ json: { machineId: machine.id, resolution: '5m', from: now - 86400_000, to: now, points: [] } });
    }
    return route.fulfill({ json: { serverTime: now, alerts: [], nextCursor: null } });
  });
  await page.goto('http://127.0.0.1:4179/');
  await page.locator('#main a').waitFor();
  for (const selector of ['#notice a', '#main a']) {
    await page.locator(selector).focus();
    const original = await page.locator(selector).elementHandle();
    for (let tick = 0; tick < 4; tick++) {
      await page.clock.runFor(5001);
      assert.equal(await original.evaluate(node => node.isConnected && document.activeElement === node), true, `${selector} lost focus during auth refresh`);
    }
  }
  assert.ok(requests >= 3, 'The focus check must cross automatic API retries');

  state = 'healthy';
  await page.locator('#refresh').click();
  await page.locator('.healthline').waitFor();
  await page.waitForFunction(() => document.querySelector('#refresh').getAttribute('aria-busy') === 'false');
  await page.clock.runFor(60_001);
  assert.equal(seriesRequests, 1, 'Live polling repeatedly fetched the full history');
  await page.waitForFunction(() => document.querySelector('#refresh').getAttribute('aria-busy') === 'false');
  await page.evaluate(() => Object.defineProperty(document, 'hidden', { configurable: true, get: () => true }));
  const requestsBeforeHidden = requests;
  await page.clock.runFor(60_001);
  assert.equal(requests, requestsBeforeHidden, 'Hidden dashboard continued polling');
  await page.evaluate(() => { delete document.hidden; });
  state = 'auth';
  await page.locator('#refresh').click();
  await page.locator('#notice a').waitFor();
  await page.locator('#notice a').focus();
  const retainedLink = await page.locator('#notice a').elementHandle();
  await page.clock.runFor(16001);
  assert.equal(await retainedLink.evaluate(node => node.isConnected && document.activeElement === node), true, 'Auth notice lost focus with retained machine readings');

  state = 'error';
  await page.locator('#refresh').click();
  await page.waitForFunction(() => document.querySelector('#notice').textContent.includes('could not refresh'));
  await page.locator('#refresh').focus();
  await page.clock.runFor(16001);
  assert.equal(await page.locator('#refresh').evaluate(node => document.activeElement === node), true, 'Refresh action lost keyboard focus during error retries');
  assert.deepEqual(errors, []);
  console.log('Browser regression passed: auth links and refresh action retain focus across render ticks and API retries.');
} finally {
  await browser?.close();
  server.kill();
}
