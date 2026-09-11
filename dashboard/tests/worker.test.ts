import { beforeEach, afterEach, describe, expect, it } from 'vitest';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import { createWorker, maintain, type Env } from '../worker/index';
import { tokenHash } from '../worker/auth';
import { connectionStatus, RAW_RETENTION_MS, type Sample } from '../worker/schema';

// Execute the real migration and SQL against SQLite. The small adapter mirrors
// D1's prepared statements and transactional batch, including trigger behavior.
function database(sqlite: DatabaseSync): D1Database {
  function statement(sql: string, args: unknown[] = []): D1PreparedStatement {
    const execute = () => {
      const stmt = sqlite.prepare(sql);
      if (stmt.columns().length) return { success: true, results: stmt.all(...args as []), meta: { changes: 0 } };
      const result = stmt.run(...args as []);
      return { success: true, results: [], meta: { changes: Number(result.changes) } };
    };
    return {
      bind: (...values: unknown[]) => statement(sql, values),
      first: async () => sqlite.prepare(sql).get(...args as []) ?? null,
      all: async () => execute(), run: async () => execute(),
    } as unknown as D1PreparedStatement;
  }
  return {
    prepare: statement,
    batch: async (statements: D1PreparedStatement[]) => {
      sqlite.exec('BEGIN');
      try {
        const results = [];
        for (const stmt of statements) results.push(await stmt.run());
        sqlite.exec('COMMIT');
        return results;
      } catch (error) { sqlite.exec('ROLLBACK'); throw error; }
    },
  } as unknown as D1Database;
}

const TOKEN_A = 'a'.repeat(43);
const TOKEN_B = 'b'.repeat(43);
let now: number;
let sqlite: DatabaseSync;
let env: Env;
const worker = createWorker({
  now: () => now,
  humanVerifier: async request => {
    if (request.headers.get('Cf-Access-Jwt-Assertion') !== 'test-human') throw new Error('Unauthorized');
    return 'human-id';
  },
});
function sample(sequence = 0): Sample {
  return {
    schemaVersion: 1, bootId: '2e100384-24de-4b48-aa51-b91d5b7a2b52', sequence,
    sampledAt: new Date(now).toISOString(), agentVersion: '1.0.0',
    system: {
      cpuPercent: 20, ramUsedBytes: 1024, ramTotalBytes: 4096,
      diskFreeBytes: 20e9, diskTotalBytes: 100e9, diskReadBytesPerSecond: null,
      diskWriteBytesPerSecond: null, networkRxBytesPerSecond: 4, networkTxBytesPerSecond: 2,
      gpuPercent: 60, vramUsedBytes: 1000, vramTotalBytes: 2000, gpuTemperatureC: 55,
    },
    app: { state: 'running', processRunning: true, statusAgeSeconds: 0, displayFps: 60,
      generationFps: 20, lastFrameAgeSeconds: 0, errorCode: null },
    archiveSync: { enabled: false, processRunning: null, state: 'disabled', statusAgeSeconds: null,
      pendingArchives: null, pendingBytes: null, verifiedLocalArchives: null, incompleteArchives: null,
      invalidArchives: null, localBytes: null, currentArchiveId: null, completedPayloadBytes: null,
      verifiedBytes: null, lastVerifiedAt: null, errorCode: null },
  };
}
function post(body: unknown, options: { token?: string; path?: string; headers?: Record<string, string>; raw?: boolean } = {}) {
  return worker.fetch(new Request(`https://monitor.test${options.path ?? '/api/v1/heartbeat'}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${options.token ?? TOKEN_A}`, ...options.headers },
    body: options.raw ? body as string : JSON.stringify(body),
  }), env);
}
async function machines() {
  return (await worker.fetch(new Request('https://monitor.test/api/v1/machines', { headers: { 'Cf-Access-Jwt-Assertion': 'test-human' } }), env)).json() as Promise<{ machines: { id: string; lastSeenAt: number | null; connection: string; latest: Sample | null }[] }>;
}

beforeEach(async () => {
  now = Date.parse('2026-09-11T06:00:00Z');
  sqlite = new DatabaseSync(':memory:');
  sqlite.exec(readFileSync(new URL('../migrations/0001_monitor.sql', import.meta.url), 'utf8'));
  sqlite.prepare('INSERT INTO machines(id,label,token_hash,created_at) VALUES(?,?,?,?)').run('a', 'Machine A', await tokenHash(TOKEN_A), now);
  sqlite.prepare('INSERT INTO machines(id,label,token_hash,created_at) VALUES(?,?,?,?)').run('b', 'Machine B', await tokenHash(TOKEN_B), now);
  const limiter = { limit: async () => ({ success: true }) } as RateLimit;
  env = { DB: database(sqlite), PUBLIC_ORIGIN: 'https://monitor.test',
    ACCESS_TEAM_DOMAIN: 'test.cloudflareaccess.com', ACCESS_AUD: 'test-aud',
    REQUEST_LIMIT: limiter, INGEST_LIMIT: limiter, HISTORY_LIMIT: limiter,
    ASSETS: { fetch: async () => new Response('placeholder') } as unknown as Fetcher };
});
afterEach(() => sqlite.close());

describe('authentication and request boundary', () => {
  it('rejects absent, malformed, wrong and revoked machine tokens', async () => {
    for (const token of ['', 'wrong', 'z'.repeat(43)]) expect((await post(sample(), { token })).status).toBe(401);
    sqlite.prepare('UPDATE machines SET revoked_at = ? WHERE id = ?').run(now, 'a');
    expect((await post(sample())).status).toBe(401);
    expect(sqlite.prepare('SELECT count(*) AS n FROM samples').get()?.n).toBe(0);
  });
  it('requires human authentication for all reads and static assets', async () => {
    for (const path of ['/', '/api/v1/machines', '/api/v1/series', '/api/v1/alerts']) {
      const response = await worker.fetch(new Request(`https://monitor.test${path}`, { headers: { Authorization: `Bearer ${TOKEN_A}`, 'Cf-Access-Authenticated-User-Email': 'spoof@example.test' } }), env);
      expect(response.status).toBe(401);
    }
    const realVerifier = createWorker();
    const response = await realVerifier.fetch(new Request('https://monitor.test/api/v1/machines', { headers: { 'Cf-Access-Jwt-Assertion': 'forged' } }), env);
    expect(response.status).toBe(401);
  });
  it('rejects wrong host and cross-origin traffic', async () => {
    expect((await worker.fetch(new Request('https://other.test/api/v1/machines'), env)).status).toBe(403);
    expect((await post(sample(), { headers: { Origin: 'https://evil.test' } })).status).toBe(403);
    expect((await post(sample(), { headers: { 'Sec-Fetch-Site': 'cross-site' } })).status).toBe(403);
  });
  it('rejects rate excess before writing', async () => {
    env.INGEST_LIMIT = { limit: async () => ({ success: false }) } as RateLimit;
    const response = await post(sample());
    expect(response.status).toBe(429);
    expect(response.headers.get('Retry-After')).toBe('60');
    expect(sqlite.prepare('SELECT count(*) AS n FROM samples').get()?.n).toBe(0);
  });
});

describe('strict bounded schema', () => {
  it('rejects invalid JSON, unknown fields at every level, missing fields and invalid numbers', async () => {
    const valid = sample();
    for (const body of [
      { ...valid, machineId: 'b' }, { ...valid, system: { ...valid.system, hostname: 'private' } },
      { ...valid, app: { ...valid.app, command: 'restart' } },
      { ...valid, archiveSync: { ...valid.archiveSync, path: 'private' } },
      { ...valid, system: {} }, { ...valid, schemaVersion: 2 },
      { ...valid, sequence: -1 }, { ...valid, sequence: 1.5 },
      { ...valid, system: { ...valid.system, cpuPercent: 101 } },
      { ...valid, app: { ...valid.app, errorCode: 'stack trace with private text' } },
    ]) expect((await post(body)).status).toBe(400);
    expect((await post('{', { raw: true })).status).toBe(400);
    expect((await post(sample(), { headers: { 'Content-Type': 'text/plain' } })).status).toBe(415);
  });
  it('rejects oversized declared and streamed bodies and excessive history batches', async () => {
    expect((await post(sample(), { headers: { 'Content-Length': '16385' } })).status).toBe(413);
    expect((await post(' '.repeat(16385), { raw: true })).status).toBe(413);
    expect((await post(' '.repeat(262145), { raw: true, path: '/api/v1/history' })).status).toBe(413);
    expect((await post({ schemaVersion: 1, samples: Array.from({ length: 121 }, () => sample()) }, { path: '/api/v1/history' })).status).toBe(400);
  });
  it('rejects future and expired records atomically', async () => {
    const future = { ...sample(), sampledAt: new Date(now + 31_000).toISOString() };
    expect((await post(future)).status).toBe(400);
    const expired = { ...sample(1), sampledAt: new Date(now - RAW_RETENTION_MS - 1).toISOString() };
    expect((await post({ schemaVersion: 1, samples: [sample(), expired] }, { path: '/api/v1/history' })).status).toBe(400);
    expect(sqlite.prepare('SELECT count(*) AS n FROM samples').get()?.n).toBe(0);
  });
  it('accepts a full bounded history batch without changing liveness', async () => {
    const samples = Array.from({ length: 120 }, (_, sequence) => ({
      ...sample(sequence), sampledAt: new Date(now - sequence * 30_000).toISOString(),
    }));
    expect((await post({ schemaVersion: 1, samples }, { path: '/api/v1/history' })).status).toBe(202);
    expect(sqlite.prepare('SELECT count(*) AS n FROM samples').get()?.n).toBe(120);
    expect((await machines()).machines[0].lastSeenAt).toBeNull();
  });
});

describe('identity, freshness and replay', () => {
  it('derives isolated machine identity from the token', async () => {
    expect((await post(sample())).status).toBe(202);
    expect((await post(sample(), { token: TOKEN_B })).status).toBe(202);
    expect(sqlite.prepare('SELECT count(*) AS n FROM samples').get()?.n).toBe(2);
    const result = await machines();
    expect(result.machines.map(row => row.connection)).toEqual(['online', 'online']);
    expect(JSON.stringify(result)).not.toContain('token_hash');
  });
  it('does not refresh liveness or overwrite payload on duplicate replay', async () => {
    const first = sample(10);
    await post(first);
    const seen = now;
    now += 120_000;
    await post({ ...first, sampledAt: new Date(now).toISOString(), app: { ...first.app, displayFps: 1 } });
    const result = (await machines()).machines[0];
    expect(result.lastSeenAt).toBe(seen);
    expect(result.connection).toBe('stale');
    expect(result.latest?.app.displayFps).toBe(60);
    expect(sqlite.prepare('SELECT count(*) AS n FROM samples').get()?.n).toBe(1);
  });
  it('does not promote old history, out-of-order sequence, or a delayed old boot', async () => {
    await post(sample(10));
    const seen = now;
    now += 60_000;
    await post(sample(9));
    expect((await machines()).machines[0].lastSeenAt).toBe(seen);
    await post({ schemaVersion: 1, samples: [sample(11)] }, { path: '/api/v1/history' });
    expect((await machines()).machines[0].lastSeenAt).toBe(seen);
    const delayed = { ...sample(12), sampledAt: new Date(now - 120_000).toISOString() };
    await post(delayed);
    expect((await machines()).machines[0].lastSeenAt).toBe(seen);
    const nextBoot = { ...sample(), bootId: 'fdc6a7e0-c5c5-4fc7-9e5b-47d2540f8250' };
    await post(nextBoot);
    now += 30_000;
    await post({ ...sample(13), sampledAt: new Date(seen + 10_000).toISOString() });
    expect((await machines()).machines[0].latest?.bootId).toBe(nextBoot.bootId);
  });
  it('never marks a history-only machine live, even after heartbeat replay', async () => {
    const record = sample();
    await post({ schemaVersion: 1, samples: [record] }, { path: '/api/v1/history' });
    await post(record);
    expect((await machines()).machines[0].connection).toBe('unknown');
  });
  it('uses exact receipt age boundaries', () => {
    expect(connectionStatus(null, now)).toBe('unknown');
    expect(connectionStatus(now - 90_000, now)).toBe('online');
    expect(connectionStatus(now - 90_001, now)).toBe('stale');
    expect(connectionStatus(now - 180_000, now)).toBe('stale');
    expect(connectionStatus(now - 180_001, now)).toBe('offline');
  });
});

describe('series and retention', () => {
  it('reads only the selected machine and computes five-minute means', async () => {
    await post(sample());
    await post({ ...sample(1), system: { ...sample().system, cpuPercent: 40 } });
    await post({ ...sample(), system: { ...sample().system, cpuPercent: 90 } }, { token: TOKEN_B });
    now += 1000;
    const response = await worker.fetch(new Request(`https://monitor.test/api/v1/series?machineId=a`, { headers: { 'Cf-Access-Jwt-Assertion': 'test-human' } }), env);
    const result = await response.json() as { points: { cpuPercent: number; sampleCount: number }[] };
    expect(response.status).toBe(200);
    expect(result.points).toHaveLength(1);
    expect(result.points[0].cpuPercent).toBe(30);
    expect(result.points[0].sampleCount).toBe(2);
  });
  it('opens and resolves alerts and expires raw samples', async () => {
    await post(sample());
    now += 180_001;
    await maintain(env, now);
    await maintain(env, now);
    expect(sqlite.prepare("SELECT count(*) AS n FROM alerts WHERE code='offline' AND resolved_at IS NULL").get()?.n).toBe(1);
    await post(sample(1));
    await maintain(env, now);
    expect(sqlite.prepare("SELECT count(*) AS n FROM alerts WHERE resolved_at IS NULL").get()?.n).toBe(0);
    now += RAW_RETENTION_MS + 1;
    await maintain(env, now);
    expect(sqlite.prepare('SELECT count(*) AS n FROM samples').get()?.n).toBe(0);
  });
});
