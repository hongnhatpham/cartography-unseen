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
const READER_TOKEN = 'r'.repeat(43);
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
  sqlite.exec(readFileSync(new URL('../migrations/0002_alert_feed.sql', import.meta.url), 'utf8'));
  sqlite.exec(readFileSync(new URL('../migrations/0003_incremental_series.sql', import.meta.url), 'utf8'));
  sqlite.prepare('INSERT INTO machines(id,label,token_hash,created_at) VALUES(?,?,?,?)').run('a', 'Machine A', await tokenHash(TOKEN_A), now);
  sqlite.prepare('INSERT INTO machines(id,label,token_hash,created_at) VALUES(?,?,?,?)').run('b', 'Machine B', await tokenHash(TOKEN_B), now);
  sqlite.prepare('INSERT INTO alert_readers(id,label,token_hash,created_at) VALUES(?,?,?,?)').run('aria', 'ARIA', await tokenHash(READER_TOKEN), now);
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
  it('allows the authenticated document navigation back from Access login', async () => {
    const headers = { 'Sec-Fetch-Site': 'cross-site', 'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Dest': 'document', 'Cf-Access-Jwt-Assertion': 'test-human' };
    const response = await worker.fetch(new Request('https://monitor.test/', { headers }), env);
    expect(response.status).toBe(200);
    expect(await response.text()).toBe('placeholder');
    const anonymous = { ...headers, 'Cf-Access-Jwt-Assertion': '' };
    expect((await worker.fetch(new Request('https://monitor.test/', { headers: anonymous }), env)).status).toBe(401);
    expect((await createWorker().fetch(new Request('https://monitor.test/', { headers }), env)).status).toBe(401);
  });
  it('keeps cross-site APIs, subresource loads, and embedded documents blocked', async () => {
    const headers = { 'Sec-Fetch-Site': 'cross-site', 'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Dest': 'document', 'Cf-Access-Jwt-Assertion': 'test-human' };
    for (const path of ['/api/v1/machines', '/api/v1/alerts', '/api/v1/alert-feed']) {
      expect((await worker.fetch(new Request(`https://monitor.test${path}`, { headers }), env)).status).toBe(403);
    }
    expect((await post(sample(), { headers })).status).toBe(403);
    const blockedHeaders: Record<string, string>[] = [
      { 'Sec-Fetch-Mode': 'cors' }, { 'Sec-Fetch-Mode': 'no-cors' },
      { 'Sec-Fetch-Dest': 'iframe' }, { 'Sec-Fetch-Dest': 'script' },
      { Origin: 'https://evil.test' },
    ];
    for (const overrides of blockedHeaders) {
      expect((await worker.fetch(new Request('https://monitor.test/', { headers: { ...headers, ...overrides } }), env)).status).toBe(403);
    }
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
  it('follows receipt cursors, replaces late buckets, and uses bounded indexes', async () => {
    const plans: string[] = [];
    const prepare = env.DB.prepare.bind(env.DB);
    env.DB.prepare = (sql: string) => {
      if (sql.includes('FROM samples INDEXED')) {
        plans.push(JSON.stringify(sqlite.prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...Array((sql.match(/\?/g) ?? []).length).fill(0))));
      }
      return prepare(sql);
    };
    const read = async (since?: number) => {
      const response = await worker.fetch(new Request(`https://monitor.test/api/v1/series?machineId=a${since === undefined ? '' : `&since=${since}`}`, { headers: { 'Cf-Access-Jwt-Assertion': 'test-human' } }), env);
      expect(response.status).toBe(200);
      return response.json() as Promise<{ cursor: number; points: { bucketAt: number; sampleCount: number; generationFps: number }[] }>;
    };
    now -= 3600_000;
    const first = sample(1);
    await post(first);
    now += 30_001;
    const initial = await read();
    expect(initial.points).toHaveLength(1);
    expect(plans.at(-1)).toContain('samples_machine_time (machine_id=? AND sampled_at>? AND sampled_at<?)');
    now += 3600_000;
    const empty = await read(initial.cursor);
    expect(empty.points).toEqual([]);
    const delayed = { ...first, sequence: 2, app: { ...first.app, generationFps: 40 } };
    await post({ schemaVersion: 1, samples: [delayed] }, { path: '/api/v1/history' });
    // A write in the cursor's exact millisecond must still be picked up.
    const delta = await read(empty.cursor);
    expect(delta.points).toHaveLength(1);
    expect(delta.points[0]).toMatchObject({ sampleCount: 2, generationFps: 30 });
    expect(plans.at(-2)).toContain('samples_machine_received (machine_id=? AND received_at>? AND received_at<?)');
    expect(plans.at(-1)).toContain('samples_rollup (machine_id=? AND bucket_at=?)');
    now += 30_001;
    const replay = await read(delta.cursor);
    expect(replay.points).toEqual(delta.points);
    now += 1;
    expect((await read(replay.cursor)).points).toEqual([]);
  });
  it('does not lose samples received with an allowed clock lead', async () => {
    const future = sample(1);
    future.sampledAt = new Date(now + 20_000).toISOString();
    await post(future);
    const read = async (since?: number) => (await worker.fetch(new Request(`https://monitor.test/api/v1/series?machineId=a${since === undefined ? '' : `&since=${since}`}`, { headers: { 'Cf-Access-Jwt-Assertion': 'test-human' } }), env)).json() as Promise<{ cursor: number; points: unknown[] }>;
    const initial = await read();
    expect(initial.points).toEqual([]);
    now += 300_000;
    expect((await read(initial.cursor)).points).toHaveLength(1);
  });
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

type Feed = {
  nextCursor: number; hasMore: boolean;
  events: { id: number; alertId: number; machineId: string; machineLabel: string; code: string;
    transition: string; reason: string | null; evidence: Record<string, unknown> | null }[];
};
function feedRequest(query = '', token = READER_TOKEN, method = 'GET') {
  return worker.fetch(new Request(`https://monitor.test/api/v1/alert-feed${query}`, {
    method, headers: { Authorization: `Bearer ${token}` },
  }), env);
}
async function feed(after = 0) {
  const response = await feedRequest(`?after=${after}`);
  expect(response.status).toBe(200);
  return await response.json() as Feed;
}

describe('ARIA alert feed', () => {
  it('isolates reader, collector and human privileges and supports immediate revocation', async () => {
    expect((await feedRequest()).status).toBe(200);
    for (const token of ['', 'wrong', TOKEN_A, 'z'.repeat(43)]) {
      expect((await feedRequest('', token)).status).toBe(401);
    }
    expect((await worker.fetch(new Request('https://monitor.test/api/v1/alert-feed', {
      headers: { 'Cf-Access-Jwt-Assertion': 'test-human' },
    }), env)).status).toBe(401);
    for (const path of ['/', '/api/v1/machines', '/api/v1/series', '/api/v1/alerts']) {
      expect((await worker.fetch(new Request(`https://monitor.test${path}`, {
        headers: { Authorization: `Bearer ${READER_TOKEN}` },
      }), env)).status).toBe(401);
    }
    expect((await post(sample(), { token: READER_TOKEN })).status).toBe(401);
    expect((await post({ schemaVersion: 1, samples: [sample()] }, { token: READER_TOKEN, path: '/api/v1/history' })).status).toBe(401);
    expect((await feedRequest('', READER_TOKEN, 'POST')).status).toBe(405);
    const response = await feedRequest();
    expect(response.headers.get('Cache-Control')).toBe('no-store');
    expect(JSON.stringify(await response.json())).not.toContain('token');
    sqlite.prepare('UPDATE alert_readers SET revoked_at = ? WHERE id = ?').run(now, 'aria');
    expect((await feedRequest()).status).toBe(401);
  });

  it('preserves both transitions between polls and replays stable IDs without duplicates', async () => {
    await post({ ...sample(), system: { ...sample().system, gpuTemperatureC: 90 } });
    await maintain(env, now);
    await maintain(env, now);
    now += 30_000;
    await post(sample(1));
    await maintain(env, now);
    await maintain(env, now);
    const result = await feed();
    expect(result.events.map(event => [event.code, event.transition, event.reason])).toEqual([
      ['gpu_hot', 'opened', null], ['gpu_hot', 'resolved', 'recovered'],
    ]);
    expect(result.events[0].alertId).toBe(result.events[1].alertId);
    expect(result.events[0].evidence?.gpuTemperatureC).toBe(90);
    expect(result.events[1].evidence?.gpuTemperatureC).toBe(55);
    expect(result.events[0].machineLabel).toBe('Machine A');
    expect((await feed()).events).toEqual(result.events);
    expect(await feed(result.nextCursor)).toMatchObject({ events: [], nextCursor: result.nextCursor, hasMore: false });
    now += 30_000;
    await post({ ...sample(2), system: { ...sample().system, gpuTemperatureC: 90 } });
    await maintain(env, now);
    const recurrence = await feed(result.nextCursor);
    expect(recurrence.events).toHaveLength(1);
    expect(recurrence.events[0].alertId).not.toBe(result.events[0].alertId);
  });

  it('pages an exclusive cursor in ID order and rejects ambiguous cursors', async () => {
    const insert = sqlite.prepare('INSERT INTO alerts(machine_id,code,opened_at,resolved_at) VALUES(?,?,?,?)');
    for (let i = 0; i < 205; i++) insert.run('a', `test_${i}`, now, null);
    const first = await feed();
    expect(first.events).toHaveLength(200);
    expect(first.hasMore).toBe(true);
    const second = await feed(first.nextCursor);
    expect(second.events).toHaveLength(5);
    expect(second.hasMore).toBe(false);
    expect(second.events[0].id).toBeGreaterThan(first.nextCursor);
    expect(new Set([...first.events, ...second.events].map(event => event.id)).size).toBe(205);
    for (const query of ['?after=-1', '?after=1.2', '?after=', '?after=01', '?after=1e2', '?after=9007199254740992', '?after=1&after=2', '?limit=1']) {
      expect((await feedRequest(query)).status).toBe(400);
    }
  });

  it('retains recovery events after the dashboard alert is pruned', async () => {
    await post(sample());
    now += 180_001;
    await maintain(env, now);
    await post(sample(1));
    await maintain(env, now);
    const original = await feed();
    now += 31 * 86400_000;
    await maintain(env, now);
    expect(sqlite.prepare('SELECT id FROM alerts WHERE id = ?').get(original.events[0].alertId)).toBeUndefined();
    expect((await feed()).events.slice(0, 2)).toEqual(original.events);
  });

  it('holds incidents through silence and unknown observations until fresh positive recovery', async () => {
    const bad = sample();
    bad.app.state = 'stopped';
    bad.system.diskFreeBytes = 1;
    bad.system.gpuTemperatureC = 90;
    bad.archiveSync = { ...bad.archiveSync, enabled: true, processRunning: true, state: 'retrying', statusAgeSeconds: 0 };
    await post(bad);
    await maintain(env, now);
    now += 180_001;
    await maintain(env, now);
    expect((await feed()).events.map(event => event.transition)).toEqual(Array(5).fill('opened'));
    const unknown = sample(1);
    unknown.app.state = 'unknown';
    unknown.system.diskFreeBytes = null;
    unknown.system.gpuTemperatureC = null;
    unknown.archiveSync.state = 'unknown';
    await post(unknown);
    await maintain(env, now);
    expect((await feed()).events.filter(event => event.transition === 'resolved').map(event => event.code)).toEqual(['offline']);
    now += 30_000;
    const healthy = sample(2);
    healthy.archiveSync = { ...healthy.archiveSync, enabled: true, processRunning: false, state: 'idle', statusAgeSeconds: 0 };
    await post(healthy);
    await maintain(env, now);
    const result = await feed();
    expect(result.events.filter(event => event.transition === 'resolved')).toHaveLength(5);
    expect(sqlite.prepare('SELECT count(*) AS n FROM alerts WHERE resolved_at IS NULL').get()?.n).toBe(0);
  });

  it.each(['error', 'stalled', 'stopped'] as const)('opens an artwork alert for %s and does not recover during startup', async state => {
    await post({ ...sample(), app: { ...sample().app, state } });
    await maintain(env, now);
    await post({ ...sample(1), app: { ...sample().app, state: 'starting' } });
    await maintain(env, now);
    expect((await feed()).events.map(event => [event.code, event.transition])).toEqual([['app_error', 'opened']]);
  });

  it('classifies revoked-machine closures separately from healthy recovery', async () => {
    await post({ ...sample(), app: { ...sample().app, state: 'error' } });
    await maintain(env, now);
    sqlite.prepare('UPDATE machines SET revoked_at = ? WHERE id = ?').run(now, 'a');
    await maintain(env, now);
    expect((await feed()).events.map(event => [event.transition, event.reason])).toEqual([
      ['opened', null], ['resolved', 'revoked'],
    ]);
  });

  it('does not clear artwork or uploader errors from stale observations or disabled uploads', async () => {
    const bad = sample();
    bad.app.state = 'error';
    bad.archiveSync = { ...bad.archiveSync, enabled: true, processRunning: true, state: 'error', statusAgeSeconds: 0 };
    await post(bad);
    await maintain(env, now);
    const stale = sample(1);
    stale.app.statusAgeSeconds = 21;
    stale.archiveSync = { ...bad.archiveSync, state: 'uploading', statusAgeSeconds: 1801 };
    await post(stale);
    await maintain(env, now);
    const disabled = sample(2);
    disabled.app.displayFps = 0;
    await post(disabled);
    await maintain(env, now);
    expect((await feed()).events.map(event => [event.code, event.transition])).toEqual([
      ['app_error', 'opened'], ['archive_error', 'opened'],
    ]);
  });

  it('backfills existing incidents without labelling legacy resolutions as recovery', () => {
    const legacy = new DatabaseSync(':memory:');
    try {
      legacy.exec(readFileSync(new URL('../migrations/0001_monitor.sql', import.meta.url), 'utf8'));
      legacy.prepare('INSERT INTO machines(id,label,token_hash,created_at) VALUES(?,?,?,?)').run('a', 'Machine A', 'a'.repeat(64), now);
      legacy.prepare('INSERT INTO alerts(machine_id,code,opened_at,resolved_at) VALUES(?,?,?,?)').run('a', 'offline', now - 1000, now);
      legacy.prepare('INSERT INTO alerts(machine_id,code,opened_at) VALUES(?,?,?)').run('a', 'app_error', now - 500);
      legacy.exec(readFileSync(new URL('../migrations/0002_alert_feed.sql', import.meta.url), 'utf8'));
      expect(legacy.prepare('SELECT code, transition, reason, evidence_json FROM alert_events ORDER BY id').all()).toEqual([
        { code: 'offline', transition: 'opened', reason: null, evidence_json: null },
        { code: 'app_error', transition: 'opened', reason: null, evidence_json: null },
        { code: 'offline', transition: 'resolved', reason: 'unknown', evidence_json: null },
      ]);
    } finally { legacy.close(); }
  });
});
