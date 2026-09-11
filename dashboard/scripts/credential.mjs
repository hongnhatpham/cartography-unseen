import { randomBytes, createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const args = process.argv.slice(2);
const alertReader = args[0] === '--alert-reader';
const [id, label] = alertReader ? args.slice(1) : args;
if (args.length !== (alertReader ? 3 : 2) || !id || !/^[A-Za-z0-9_-]{1,64}$/.test(id) || !label || label.length > 100 || /[\r\n\0]/.test(label)) {
  console.error('Usage: pnpm credential [--alert-reader] <id> "Label"');
  process.exit(1);
}
const token = randomBytes(32).toString('base64url');
const hash = createHash('sha256').update(token).digest('hex');
const directory = resolve('credentials');
await mkdir(directory, { recursive: true, mode: 0o700 });
const filename = alertReader ? `alert-reader-${id}` : id;
const tokenPath = resolve(directory, `${filename}.json`);
const sqlPath = resolve(directory, `${filename}.sql`);
const quote = value => `'${value.replaceAll("'", "''")}'`;
const sql = `INSERT INTO ${alertReader ? 'alert_readers' : 'machines'}(id, label, token_hash, created_at) VALUES(${quote(id)}, ${quote(label)}, ${quote(hash)}, ${Date.now()});\n`;
await writeFile(tokenPath, `${JSON.stringify(alertReader ? { readerId: id, role: 'alert-reader', token } : { machineId: id, token }, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
await writeFile(sqlPath, sql, { flag: 'wx', mode: 0o600 });
console.log(`Credential saved to ${tokenPath}`);
console.log(`Bootstrap SQL saved to ${sqlPath}. It contains the hash, not the token.`);
console.log(`Restrict the credential file to the ${alertReader ? 'ARIA/Hermes' : 'collector'} account before use. Do not commit or share it.`);
