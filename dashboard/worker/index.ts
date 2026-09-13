import { verifyHuman, tokenHash, type HumanVerifier } from './auth';
import { connectionStatus, HEARTBEAT_BYTES, HISTORY_BYTES, historySchema, ONLINE_MS, RAW_RETENTION_MS, sampleSchema, type Sample } from './schema';

export interface Env {
  DB: D1Database;
  ASSETS: Fetcher;
  PUBLIC_ORIGIN: string;
  ACCESS_TEAM_DOMAIN: string;
  ACCESS_AUD: string;
  REQUEST_LIMIT: RateLimit;
  INGEST_LIMIT: RateLimit;
  HISTORY_LIMIT: RateLimit;
}

type Machine = {
  id: string; label: string; created_at: number; revoked_at: number | null;
  last_seen_at: number | null; latest_json: string | null;
};
class HttpError extends Error {
  constructor(public status: number, message: string) { super(message); }
}
const fail = (status: number, message: string): never => { throw new HttpError(status, message); };
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json; charset=utf-8' },
});
function secure(response: Response) {
  const result = new Response(response.body, response);
  result.headers.set('Cache-Control', 'no-store');
  result.headers.set('X-Content-Type-Options', 'nosniff');
  result.headers.set('Referrer-Policy', 'no-referrer');
  result.headers.set('X-Frame-Options', 'DENY');
  result.headers.set('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'");
  result.headers.set('Strict-Transport-Security', 'max-age=31536000');
  return result;
}

async function limitedBody(request: Request, limit: number): Promise<unknown> {
  if (request.headers.get('content-type')?.split(';')[0].trim() !== 'application/json') fail(415, 'Expected application/json');
  if (request.headers.has('content-encoding') && request.headers.get('content-encoding') !== 'identity') fail(415, 'Compressed requests are not accepted');
  const length = request.headers.get('content-length');
  if (length && (!/^\d+$/.test(length) || Number(length) > limit)) fail(413, 'Request too large');
  if (!request.body) fail(400, 'JSON body required');
  const reader = request.body!.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) {
        await reader.cancel();
        fail(413, 'Request too large');
      }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch { return fail(400, 'Invalid JSON'); }
}

async function rate(binding: RateLimit, key: string) {
  if (!(await binding.limit({ key })).success) fail(429, 'Request rate exceeded');
}

async function ingest(request: Request, env: Env, history: boolean, now: number) {
  const authorization = request.headers.get('Authorization') ?? '';
  if (!/^Bearer [A-Za-z0-9_-]{43}$/.test(authorization)) fail(401, 'Invalid machine credential');
  const hash = await tokenHash(authorization.slice(7));
  const machine = await env.DB.prepare('SELECT id FROM machines WHERE token_hash = ? AND revoked_at IS NULL').bind(hash).first<{ id: string }>();
  if (!machine) fail(401, 'Invalid machine credential');
  await rate(env.INGEST_LIMIT, machine!.id);
  if (history) await rate(env.HISTORY_LIMIT, machine!.id);
  const body = await limitedBody(request, history ? HISTORY_BYTES : HEARTBEAT_BYTES);
  const result = history ? historySchema.safeParse(body) : sampleSchema.safeParse(body);
  if (!result.success) fail(400, 'Invalid schema v1 payload');
  const parsed = result.data!;
  const samples = 'samples' in parsed ? parsed.samples : [parsed];
  for (const sample of samples) {
    const sampled = Date.parse(sample.sampledAt);
    if (sampled < now - RAW_RETENTION_MS || sampled > now + 30_000) fail(400, 'Sample time outside accepted window');
  }
  const records = samples.map(sample => {
    const sampled = Date.parse(sample.sampledAt);
    return { sample, sampled, bucket: Math.floor(sampled / 300_000) * 300_000,
      live: !history && now - sampled <= ONLINE_MS ? 1 : 0 };
  });
  // A single statement keeps a full history batch atomic and below D1's
  // per-invocation query limit even on the free plan.
  const inserted = await env.DB.prepare(`INSERT OR IGNORE INTO samples
    (machine_id, boot_id, sequence, sampled_at, received_at, bucket_at, source, live_eligible, payload_json)
    SELECT machines.id, json_extract(item.value, '$.sample.bootId'),
      json_extract(item.value, '$.sample.sequence'), json_extract(item.value, '$.sampled'),
      ?, json_extract(item.value, '$.bucket'), ?, json_extract(item.value, '$.live'),
      json_extract(item.value, '$.sample')
    FROM machines, json_each(?) AS item
    WHERE machines.id = ? AND token_hash = ? AND revoked_at IS NULL`)
    .bind(now, history ? 'history' : 'heartbeat', JSON.stringify(records), machine!.id, hash).run();
  // Changes can include trigger writes. Deduplication is reported as an accepted
  // idempotent request; callers do not need to infer insert counts from D1.
  if (!inserted.success) fail(503, 'Storage unavailable');
  return json({ accepted: samples.length, receivedAt: new Date(now).toISOString(), history }, 202);
}

function queryParams(url: URL, allowed: string[]) {
  for (const key of url.searchParams.keys()) {
    if (!allowed.includes(key) || url.searchParams.getAll(key).length !== 1) fail(400, 'Invalid query parameter');
  }
}
function machineId(url: URL) {
  const id = url.searchParams.get('machineId');
  if (!id || !/^[A-Za-z0-9_-]{1,64}$/.test(id)) fail(400, 'Valid machineId required');
  return id!;
}

async function readApi(url: URL, env: Env, now: number) {
  if (url.pathname === '/api/v1/machines') {
    queryParams(url, ['after']);
    const after = url.searchParams.get('after') ?? '';
    if (after.length > 64) fail(400, 'Invalid cursor');
    const { results } = await env.DB.prepare(`SELECT id, label, created_at, revoked_at, last_seen_at, latest_json
      FROM machines WHERE id > ? ORDER BY id LIMIT 201`).bind(after).all<Machine>();
    const machines = results.slice(0, 200).map(row => ({
      id: row.id, label: row.label, createdAt: row.created_at,
      revoked: row.revoked_at !== null,
      connection: connectionStatus(row.last_seen_at, now),
      lastSeenAt: row.last_seen_at,
      ageSeconds: row.last_seen_at === null ? null : Math.max(0, (now - row.last_seen_at) / 1000),
      latest: row.latest_json ? JSON.parse(row.latest_json) as Sample : null,
    }));
    return json({ serverTime: now, machines, nextCursor: results.length > 200 ? machines.at(-1)!.id : null });
  }
  if (url.pathname === '/api/v1/series') {
    queryParams(url, ['machineId', 'from', 'to', 'resolution', 'since']);
    const id = machineId(url);
    const to = url.searchParams.has('to') ? Number(url.searchParams.get('to')) : now;
    const from = url.searchParams.has('from') ? Number(url.searchParams.get('from')) : to - 24 * 3600_000;
    const resolution = url.searchParams.get('resolution') ?? '5m';
    const since = url.searchParams.has('since') ? Number(url.searchParams.get('since')) : null;
    // Ingest permits 30 seconds of device clock skew. Keep that overlap so a
    // sample initially ahead of the chart's end is picked up when time catches up.
    const cursor = now - 30_000;
    if (since !== null && (!Number.isSafeInteger(since) || since < now - RAW_RETENTION_MS || since > now || resolution !== '5m')) fail(400, 'Invalid series cursor');
    if (!Number.isSafeInteger(from) || !Number.isSafeInteger(to) || from >= to || to > now + 30_000 || from < now - RAW_RETENTION_MS || !['raw', '5m'].includes(resolution)) fail(400, 'Invalid series window or resolution');
    const exists = await env.DB.prepare('SELECT id FROM machines WHERE id = ?').bind(id).first();
    if (!exists) fail(404, 'Machine not found');
    if (resolution === 'raw') {
      const { results } = await env.DB.prepare(`SELECT sampled_at AS sampledAt, received_at AS receivedAt, source, payload_json
        FROM samples WHERE machine_id = ? AND sampled_at >= ? AND sampled_at < ? ORDER BY sampled_at LIMIT 2001`).bind(id, from, to).all<{ sampledAt: number; receivedAt: number; source: string; payload_json: string }>();
      return json({ machineId: id, from, to, resolution, truncated: results.length > 2000,
        points: results.slice(0, 2000).map(({ payload_json, ...row }) => ({ ...row, sample: JSON.parse(payload_json) })) });
    }
    const metrics = {
      cpuPercent: '$.system.cpuPercent', ramUsedBytes: '$.system.ramUsedBytes',
      ramTotalBytes: '$.system.ramTotalBytes', diskFreeBytes: '$.system.diskFreeBytes',
      diskTotalBytes: '$.system.diskTotalBytes', diskReadBytesPerSecond: '$.system.diskReadBytesPerSecond',
      diskWriteBytesPerSecond: '$.system.diskWriteBytesPerSecond', networkRxBytesPerSecond: '$.system.networkRxBytesPerSecond',
      networkTxBytesPerSecond: '$.system.networkTxBytesPerSecond', gpuPercent: '$.system.gpuPercent',
      vramUsedBytes: '$.system.vramUsedBytes', vramTotalBytes: '$.system.vramTotalBytes', gpuTemperatureC: '$.system.gpuTemperatureC',
      displayFps: '$.app.displayFps', generationFps: '$.app.generationFps', pendingArchives: '$.archiveSync.pendingArchives',
      pendingBytes: '$.archiveSync.pendingBytes', verifiedBytes: '$.archiveSync.verifiedBytes',
    };
    const columns = Object.entries(metrics).map(([key, path]) => `AVG(json_extract(payload_json, '${path}')) AS ${key}`).join(', ');
    // Use receipt time for deltas: a late replay can change an older bucket.
    // Inclusive cursor boundaries intentionally replay equal-millisecond writes.
    let changed: number[] | null = null;
    if (since !== null) {
      const delta = await env.DB.prepare(`SELECT DISTINCT bucket_at AS bucketAt
        FROM samples INDEXED BY samples_machine_received
        WHERE machine_id = ? AND received_at >= ? AND received_at <= ?
          AND bucket_at >= ? AND bucket_at < ?`).bind(id, since, now, Math.floor(from / 300_000) * 300_000, to).all<{ bucketAt: number }>();
      changed = delta.results.map(row => row.bucketAt);
      if (!changed.length) return json({ machineId: id, from, to, resolution, cursor, points: [] });
    }
    const query = changed === null
      ? env.DB.prepare(`SELECT bucket_at AS bucketAt, COUNT(*) AS sampleCount, ${columns}
          FROM samples INDEXED BY samples_machine_time
          WHERE machine_id = ? AND sampled_at >= ? AND sampled_at < ?
          GROUP BY bucket_at ORDER BY bucket_at LIMIT 2017`).bind(id, from, to)
      : env.DB.prepare(`SELECT bucket_at AS bucketAt, COUNT(*) AS sampleCount, ${columns}
          FROM samples INDEXED BY samples_rollup
          WHERE machine_id = ? AND bucket_at IN (SELECT value FROM json_each(?))
            AND sampled_at >= ? AND sampled_at < ?
          GROUP BY bucket_at ORDER BY bucket_at LIMIT 2017`).bind(id, JSON.stringify(changed), from, to);
    const { results } = await query.all();
    return json({ machineId: id, from, to, resolution, cursor, points: results });
  }
  if (url.pathname === '/api/v1/alerts') {
    queryParams(url, ['machineId', 'before']);
    const id = url.searchParams.has('machineId') ? machineId(url) : null;
    const before = url.searchParams.has('before') ? Number(url.searchParams.get('before')) : Number.MAX_SAFE_INTEGER;
    if (!Number.isSafeInteger(before) || before < 0) fail(400, 'Invalid alert cursor');
    const { results } = await env.DB.prepare(`SELECT id, machine_id AS machineId, code, opened_at AS openedAt, resolved_at AS resolvedAt
      FROM alerts WHERE (? IS NULL OR machine_id = ?) AND id < ? ORDER BY id DESC LIMIT 201`).bind(id, id, before).all<{ id: number }>();
    return json({ serverTime: now, alerts: results.slice(0, 200), nextCursor: results.length > 200 ? results[199].id : null });
  }
  return fail(404, 'Not found');
}

async function alertFeed(request: Request, url: URL, env: Env, now: number) {
  const authorization = request.headers.get('Authorization') ?? '';
  if (!/^Bearer [A-Za-z0-9_-]{43}$/.test(authorization)) fail(401, 'Invalid alert reader credential');
  const reader = await env.DB.prepare('SELECT id FROM alert_readers WHERE token_hash = ? AND revoked_at IS NULL')
    .bind(await tokenHash(authorization.slice(7))).first<{ id: string }>();
  if (!reader) fail(401, 'Invalid alert reader credential');
  await rate(env.REQUEST_LIMIT, `alert-reader:${reader!.id}`);
  queryParams(url, ['after']);
  const cursor = url.searchParams.get('after') ?? '0';
  const after = Number(cursor);
  if (!/^(0|[1-9][0-9]*)$/.test(cursor) || !Number.isSafeInteger(after)) fail(400, 'Invalid alert feed cursor');
  const { results } = await env.DB.prepare(`SELECT id, alert_id AS alertId, machine_id AS machineId,
    machine_label AS machineLabel, code, transition, occurred_at AS occurredAt, reason, evidence_json
    FROM alert_events WHERE id > ? ORDER BY id LIMIT 201`).bind(after).all<{
      id: number; alertId: number; machineId: string; machineLabel: string; code: string;
      transition: 'opened' | 'resolved'; occurredAt: number; reason: string | null; evidence_json: string | null;
    }>();
  const events = results.slice(0, 200).map(({ evidence_json, ...event }) => ({
    ...event, evidence: evidence_json ? JSON.parse(evidence_json) : null,
  }));
  return json({ schemaVersion: 1, serverTime: now, events,
    nextCursor: events.at(-1)?.id ?? after, hasMore: results.length > 200 });
}

const alertConditions: Record<string, { open: string; recover: string }> = {
  offline: {
    open: 'last_seen_at IS NOT NULL AND last_seen_at < ? - 180000',
    recover: 'last_seen_at >= ? - 90000',
  },
  app_error: {
    open: "last_seen_at >= ? - 90000 AND json_extract(latest_json, '$.app.state') IN ('error','stalled','stopped')",
    recover: `last_seen_at >= ? - 90000 AND json_extract(latest_json, '$.app.state') = 'running'
      AND json_extract(latest_json, '$.app.processRunning') = 1
      AND json_extract(latest_json, '$.app.statusAgeSeconds') <= 20
      AND json_extract(latest_json, '$.app.displayFps') > 0
      AND json_extract(latest_json, '$.app.lastFrameAgeSeconds') <= 20`,
  },
  archive_error: {
    open: "last_seen_at >= ? - 90000 AND json_extract(latest_json, '$.archiveSync.state') IN ('error','retrying')",
    recover: `last_seen_at >= ? - 90000 AND json_extract(latest_json, '$.archiveSync.enabled') = 1
      AND json_extract(latest_json, '$.archiveSync.statusAgeSeconds') <= 1800
      AND json_extract(latest_json, '$.archiveSync.state') IN ('idle','scanning','packing','uploading','verifying','pruning')
      AND (json_extract(latest_json, '$.archiveSync.processRunning') = 1 OR json_extract(latest_json, '$.archiveSync.state') = 'idle')`,
  },
  disk_low: {
    open: "last_seen_at >= ? - 90000 AND json_extract(latest_json, '$.system.diskFreeBytes') < 10737418240",
    recover: "last_seen_at >= ? - 90000 AND json_extract(latest_json, '$.system.diskFreeBytes') >= 10737418240",
  },
  gpu_hot: {
    open: "last_seen_at >= ? - 90000 AND json_extract(latest_json, '$.system.gpuTemperatureC') >= 85",
    recover: "last_seen_at >= ? - 90000 AND json_extract(latest_json, '$.system.gpuTemperatureC') < 85",
  },
};

export async function maintain(env: Env, now: number) {
  const statements: D1PreparedStatement[] = [];
  for (const [code, condition] of Object.entries(alertConditions)) {
    statements.push(env.DB.prepare(`INSERT OR IGNORE INTO alerts(machine_id, code, opened_at)
      SELECT id, ?, ? FROM machines WHERE revoked_at IS NULL AND (${condition.open})`).bind(code, now, now));
    // Silence and null readings are not evidence of recovery.
    statements.push(env.DB.prepare(`UPDATE alerts SET resolved_at = ?, resolution_reason = 'recovered'
      WHERE code = ? AND resolved_at IS NULL AND machine_id IN
      (SELECT id FROM machines WHERE revoked_at IS NULL AND (${condition.recover}))`).bind(now, code, now));
  }
  statements.push(env.DB.prepare(`UPDATE alerts SET resolved_at = ?, resolution_reason = 'revoked'
    WHERE resolved_at IS NULL AND machine_id IN (SELECT id FROM machines WHERE revoked_at IS NOT NULL)`).bind(now));
  statements.push(env.DB.prepare('DELETE FROM samples WHERE sampled_at < ?').bind(now - RAW_RETENTION_MS));
  statements.push(env.DB.prepare('DELETE FROM machine_boots WHERE last_received_at < ?').bind(now - RAW_RETENTION_MS));
  statements.push(env.DB.prepare('DELETE FROM alerts WHERE resolved_at IS NOT NULL AND resolved_at < ?').bind(now - 30 * 86400_000));
  await env.DB.batch(statements);
}

export function createWorker(options: { humanVerifier?: HumanVerifier; now?: () => number } = {}) {
  const humanVerifier = options.humanVerifier ?? verifyHuman;
  return {
    async fetch(request: Request, env: Env): Promise<Response> {
      try {
        const url = new URL(request.url);
        if (url.origin !== env.PUBLIC_ORIGIN || url.username || url.password) fail(403, 'Host not allowed');
        if (url.protocol !== 'https:' && !(url.protocol === 'http:' && ['127.0.0.1', 'localhost'].includes(url.hostname))) fail(403, 'HTTPS required');
        const origin = request.headers.get('Origin');
        if (origin && origin !== env.PUBLIC_ORIGIN) fail(403, 'Origin not allowed');
        // Access returns from its separate login origin via a document navigation.
        // Allow that entry point; the human JWT is still verified before assets.
        const documentNavigation = request.method === 'GET'
          && request.headers.get('Sec-Fetch-Mode') === 'navigate'
          && request.headers.get('Sec-Fetch-Dest') === 'document'
          && !url.pathname.startsWith('/api/');
        if (request.headers.get('Sec-Fetch-Site') === 'cross-site' && !documentNavigation) {
          fail(403, 'Cross-site request not allowed');
        }
        if (url.href.length > 2048) fail(414, 'URL too long');
        const now = (options.now ?? Date.now)();
        await rate(env.REQUEST_LIMIT, `ip:${request.headers.get('CF-Connecting-IP') ?? 'local'}`);
        if (['/api/v1/heartbeat', '/api/v1/history'].includes(url.pathname)) {
          if (request.method !== 'POST') fail(405, 'Method not allowed');
          queryParams(url, []);
          return secure(await ingest(request, env, url.pathname.endsWith('/history'), now));
        }
        if (url.pathname === '/api/v1/alert-feed') {
          if (request.method !== 'GET') fail(405, 'Method not allowed');
          return secure(await alertFeed(request, url, env, now));
        }
        if (request.method !== 'GET' && request.method !== 'HEAD') fail(405, 'Method not allowed');
        let user: string;
        try { user = await humanVerifier(request, env); }
        catch { return secure(json({ error: 'Cloudflare Access authentication required' }, 401)); }
        await rate(env.REQUEST_LIMIT, `user:${user}`);
        if (url.pathname.startsWith('/api/')) return secure(await readApi(url, env, now));
        return secure(await env.ASSETS.fetch(request));
      } catch (error) {
        const status = error instanceof HttpError ? error.status : 503;
        const result = json({ error: error instanceof HttpError ? error.message : 'Monitoring service unavailable' }, status);
        if (status === 429) result.headers.set('Retry-After', '60');
        return secure(result);
      }
    },
    async scheduled(_event: ScheduledController, env: Env, ctx: ExecutionContext) {
      ctx.waitUntil(maintain(env, (options.now ?? Date.now)()));
    },
  };
}

export default createWorker();
