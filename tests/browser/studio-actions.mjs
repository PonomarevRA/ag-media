/** Check compact actions with the real Studio frontend and synthetic, request-isolated data. */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';

const { chromium } = await import(process.env.VOICE_PLAYWRIGHT_MODULE || 'playwright');
const browser = await chromium.launch({ executablePath: process.env.VOICE_CHROMIUM || undefined });
const origin = 'http://studio-design.test';
const calls = [];
const recording = { id: '10000000000000000000000000000000', text: 'Тестовая реплика', seconds: 2 };
const node = (id) => ({ id, label: id, editable: true, enabled: true, dependsOn: [] });
const results = {
  storage: {
    items: [
      {
        id: 'row-a',
        label: 'Long-file-name-for-existing-archive-0123456789',
        kind: 'exports',
        bytes: 1337,
        lastUsed: 1700000000,
        candidate: true,
      },
    ],
    groups: [{ kind: 'exports', bytes: 1337 }],
    freeBytes: 8096,
    policy: { cacheDays: 7, temporaryDays: 2, exportDays: 31 },
    totalItems: 1,
  },
  'transcript-materials': { items: [], nextOffset: null },
  mods: { mods: [], directory: '/test/mods' },
  'mods/pipeline': { nodes: [node('mod.a'), node('mod.b')], dependencies: {}, revision: 'test' },
  state: {
    engines: {},
    checkpoints: {},
    profiles: [],
    recordings: [recording],
    jobs: [],
    trainers: [],
    testText: 'test',
    heldOut: [],
    examples: [],
  },
};
const kinds = {
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.html': 'text/html',
};
try {
  for (const [design, scheme, width, scale, locale] of [
    ['retro', 'dark', 1080, 1, 'ru'],
    ['modern', 'light', 320, 1, 'en'],
    ['editor', 'dark', 1080, 2, 'en'],
    ['retro', 'light', 320, 2, 'ru'],
  ]) {
    const page = await browser.newPage({ viewport: { width, height: 960 } });
    await page.addInitScript(
      ({ design, scheme }) => {
        localStorage.setItem('voice-studio:design-profile', design);
        localStorage.setItem('voice-studio:color-mode', scheme);
      },
      { design, scheme },
    );
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.route(origin + '/**', async (route) => {
      const { pathname } = new URL(route.request().url());
      if (pathname.startsWith('/api/')) {
        const key = pathname.slice(5);
        calls.push(key);
        return route.fulfill({ json: results[key] ?? {} });
      }
      const part = extname(pathname) ? pathname.slice(1) : 'index.html';
      assert(!part.includes('..'));
      try {
        const path = join(process.cwd(), 'web', part);
        return route.fulfill({
          body: await readFile(path),
          contentType: kinds[extname(part)] ?? 'application/octet-stream',
        });
      } catch {
        return route.fulfill({ status: 404, body: '' });
      }
    });
    const goto = async (name) => {
      await page.goto(origin + name + '?ui=' + locale);
      if (scale === 2) await page.addStyleTag({ content: ':root { font-size: 200% !important; }' });
      assert.equal(
        await page.locator('html').evaluate((e) => parseFloat(getComputedStyle(e).fontSize)),
        16 * scale,
      );
    };
    const report = { design, scheme, width, scale, locale, controls: {} };
    /** Verify the action slot, accessible name, and pointer/keyboard tooltip lifecycle. */
    async function control(id, name, { disabled = false } = {}) {
      const button = page.locator(id);
      await button.waitFor({ state: 'visible' });
      await button.scrollIntoViewIfNeeded();
      const box = await button.boundingBox();
      report.controls[id] = {
        width: box.width,
        height: box.height,
        caption: (await button.textContent()).trim(),
      };
      assert.equal(await button.locator('svg[aria-hidden=true]').count(), 1, id);
      assert.equal(await button.getAttribute('aria-label'), name, id);
      assert(box.width >= 44 && box.height >= 44, id);
      if (disabled) {
        assert(await button.isDisabled());
        return;
      }
      await button.hover();
      const hint = await button.evaluate((element) => {
        const style = getComputedStyle(element, '::after');
        const box = element.getBoundingClientRect();
        const pixels = (key) => parseFloat(style[key]) || 0;
        const width =
          pixels('width') +
          (style.boxSizing === 'border-box'
            ? 0
            : pixels('paddingLeft') +
              pixels('paddingRight') +
              pixels('borderLeftWidth') +
              pixels('borderRightWidth'));
        const left =
          style.left === 'auto'
            ? box.right - element.clientLeft - pixels('right') - width
            : box.left + element.clientLeft + pixels('left');
        return { display: style.display, left, right: left + width };
      });
      assert.equal(hint.display, 'block', id);
      assert(hint.left >= -1 && hint.right <= width + 1, id + ' tooltip overflows: ' + JSON.stringify(hint));
      await button.focus();
      await page.keyboard.press('Escape');
      assert.equal(await button.evaluate((e) => getComputedStyle(e, '::after').display), 'none', id);
      await page.mouse.move(1, 1);
      await page.keyboard.press('Shift+Tab');
      await page.keyboard.press('Tab');
      assert(
        await button.evaluate(
          (e) => e === document.activeElement && getComputedStyle(e, '::after').display === 'block',
        ),
        id,
      );
    }
    const en = locale === 'en';
    await goto('/storage');
    await control('#storage-refresh', en ? 'Refresh list' : 'Обновить список');
    await control(
      '.storage-item button',
      en ? 'Delete ' + results.storage.items[0].label : 'Удалить ' + results.storage.items[0].label,
    );
    {
      await page.keyboard.press('Enter');
      assert(await page.locator('#storage-dialog').isVisible(), 'Storage confirmation opens');
      assert(await page.locator('#storage-confirm').isVisible(), 'Final delete is visible');
      const deletions = calls.filter((x) => x === 'storage-delete').length;
      await page.locator('#storage-cancel').click();
      assert.equal(calls.filter((x) => x === 'storage-delete').length, deletions, 'Cancel does not delete');
      const refreshes = calls.filter((x) => x === 'storage').length;
      await page.locator('#storage-refresh').click();
      await page.waitForFunction(() => !document.querySelector('#storage-refresh').disabled);
      assert.equal(calls.filter((x) => x === 'storage').length, refreshes + 1);
    }
    await goto('/settings');
    await control('#mods-rescan', en ? 'Rescan catalog' : 'Обновить каталог');
    await control('#pipeline-refresh', en ? 'Reload plan' : 'Обновить план');
    await control('#pipeline-undo', en ? 'Undo edit' : 'Отменить правку', { disabled: true });
    await control('#pipeline-redo', en ? 'Redo edit' : 'Вернуть правку', { disabled: true });
    {
      await page.locator('[data-node-id="mod.a"]').click();
      const dependency = page.locator('[data-parent-id="mod.b"]');
      await dependency.check();
      assert(await page.locator('#pipeline-save').isEnabled());
      await control('#pipeline-undo', en ? 'Undo edit' : 'Отменить правку');
      await page.keyboard.press('Enter');
      assert.equal(await dependency.isChecked(), false);
      await control('#pipeline-redo', en ? 'Redo edit' : 'Вернуть правку');
      await page.keyboard.press('Enter');
      assert(await dependency.isChecked());
      assert.equal(await page.locator('#pipeline-save').textContent(), en ? 'Save plan' : 'Сохранить план');
    }
    await goto('/tools');
    await page.locator('#transcript-materials summary').click();
    await control('#transcript-material-refresh', en ? 'Refresh list' : 'Обновить список');
    const materialRefreshes = calls.filter((key) => key === 'transcript-materials').length;
    await page.keyboard.press('Enter');
    await page.waitForFunction(() => !document.querySelector('#transcript-material-refresh').disabled);
    assert.equal(calls.filter((key) => key === 'transcript-materials').length, materialRefreshes + 1);

    await goto('/history');
    await page.locator('.history-use').click();
    await control('#clear-source', en ? 'Clear source' : 'Сбросить исходник');
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('#clear-source').isHidden(), true);
    assert.equal(await page.locator('#test-text').evaluate((e) => e === document.activeElement), true);
    assert.deepEqual(errors, []);
    report.pageErrors = errors;
    console.log(JSON.stringify(report));
    await page.close();
  }
} finally {
  await browser.close();
}
