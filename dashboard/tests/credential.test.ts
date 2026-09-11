import { afterEach, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const directories: string[] = [];
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

it.each([false, true])('generates a private, non-overwriting credential with reader mode=%s', alertReader => {
  const directory = mkdtempSync(join(tmpdir(), 'monitor-credential-'));
  directories.push(directory);
  const script = fileURLToPath(new URL('../scripts/credential.mjs', import.meta.url));
  const args = [script, ...(alertReader ? ['--alert-reader'] : []), 'test', "Test's label"];
  const stdout = execFileSync(process.execPath, args, { cwd: directory, encoding: 'utf8' });
  const filename = alertReader ? 'alert-reader-test' : 'test';
  const tokenPath = join(directory, 'credentials', `${filename}.json`);
  const credential = JSON.parse(readFileSync(tokenPath, 'utf8'));
  const sql = readFileSync(join(directory, 'credentials', `${filename}.sql`), 'utf8');
  expect(credential.token).toMatch(/^[A-Za-z0-9_-]{43}$/);
  expect(credential).toMatchObject(alertReader ? { readerId: 'test', role: 'alert-reader' } : { machineId: 'test' });
  expect(sql).toContain(`INSERT INTO ${alertReader ? 'alert_readers' : 'machines'}(`);
  expect(sql).toContain("Test''s label");
  expect(sql).toContain(createHash('sha256').update(credential.token).digest('hex'));
  expect(sql + stdout).not.toContain(credential.token);
  expect(() => execFileSync(process.execPath, args, { cwd: directory, stdio: 'pipe' })).toThrow();
  expect(JSON.parse(readFileSync(tokenPath, 'utf8'))).toEqual(credential);
});
