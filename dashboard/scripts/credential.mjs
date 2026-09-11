import { randomBytes, createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const [id, label] = process.argv.slice(2);
if (!id || !/^[A-Za-z0-9_-]{1,64}$/.test(id) || !label || label.length > 100 || /[\r\n\0]/.test(label)) {
  console.error('Usage: pnpm credential <machine-id> "Machine label"');
  process.exit(1);
}
const token = randomBytes(32).toString('base64url');
const hash = createHash('sha256').update(token).digest('hex');
const directory = resolve('credentials');
await mkdir(directory, { recursive: true, mode: 0o700 });
const tokenPath = resolve(directory, `${id}.json`);
const sqlPath = resolve(directory, `${id}.sql`);
const quote = value => `'${value.replaceAll("'", "''")}'`;
const sql = `INSERT INTO machines(id, label, token_hash, created_at) VALUES(${quote(id)}, ${quote(label)}, ${quote(hash)}, ${Date.now()});\n`;
await writeFile(tokenPath, `${JSON.stringify({ machineId: id, token }, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
await writeFile(sqlPath, sql, { flag: 'wx', mode: 0o600 });
console.log(`Credential saved to ${tokenPath}`);
console.log(`Bootstrap SQL saved to ${sqlPath}. It contains the hash, not the token.`);
console.log('Restrict the credential file to the collector account before use. Do not commit or share it.');
