import { z } from 'zod';
import { sampleSchema } from '../worker/schema';

const metric = z.number().finite().nullable();
const machineSchema = z.object({
  id: z.string().regex(/^[A-Za-z0-9_-]{1,64}$/), label: z.string(), revoked: z.boolean(),
  lastSeenAt: metric, ageSeconds: z.number().nonnegative().nullable(), latest: sampleSchema.nullable(),
});
const machinesSchema = z.object({ serverTime: z.number().finite(), machines: z.array(machineSchema), nextCursor: z.string().nullable() });
const seriesSchema = z.object({
  machineId: z.string(), from: z.number(), to: z.number(), resolution: z.literal('5m'), cursor: z.number().optional(),
  points: z.array(z.object({ bucketAt: z.number(), generationFps: metric, displayFps: metric })),
});
const alertsSchema = z.object({
  serverTime: z.number(), nextCursor: z.number().nullable(),
  alerts: z.array(z.object({ id: z.number(), machineId: z.string(), code: z.string(), openedAt: z.number(), resolvedAt: metric })),
});
export type Series = z.infer<typeof seriesSchema>;
export type Alerts = z.infer<typeof alertsSchema>;
export class ApiError extends Error {
  constructor(public status: number) { super(status === 401 || status === 403 ? 'Sign in again to see the exhibition.' : 'The monitor could not refresh.'); }
}
async function get<T>(path: string, schema: z.ZodType<T>, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { credentials: 'same-origin', cache: 'no-store', signal, redirect: 'error', headers: { Accept: 'application/json' } });
  if (!response.ok) throw new ApiError(response.status);
  // An Access login page is not a successful telemetry response.
  if (!response.headers.get('content-type')?.includes('application/json')) throw new ApiError(401);
  return schema.parse(await response.json());
}
export async function getMachines(signal: AbortSignal) {
  let data = await get('/api/v1/machines', machinesSchema, signal);
  const machines = [...data.machines];
  const cursors = new Set<string>();
  while (data.nextCursor !== null) {
    if (cursors.has(data.nextCursor)) throw new Error('Repeated machine cursor');
    cursors.add(data.nextCursor);
    data = await get(`/api/v1/machines?after=${encodeURIComponent(data.nextCursor)}`, machinesSchema, signal);
    machines.push(...data.machines);
  }
  return { serverTime: data.serverTime, machines: machines.map(machine => ({ ...machine, ageSeconds: machine.lastSeenAt === null ? null : Math.max(0, (data.serverTime - machine.lastSeenAt) / 1000) })) };
}
let cachedSeries: { id: string; until: number; data: Series } | undefined;
export async function getSeries(id: string, signal: AbortSignal, force = false): Promise<Series> {
  signal.throwIfAborted();
  // Live status polls do not rescan history. After the initial load, receipt
  // cursors refresh only changed buckets, including delayed history uploads.
  if (!force && cachedSeries?.id === id && performance.now() < cachedSeries.until) return cachedSeries.data;
  const previous = cachedSeries?.id === id ? cachedSeries.data : undefined;
  const cursor = previous?.cursor;
  const incremental = cursor !== undefined && Date.now() - cursor < 6 * 86400_000;
  const data = await get(`/api/v1/series?machineId=${encodeURIComponent(id)}&resolution=5m${incremental ? `&since=${cursor}` : ''}`, seriesSchema, signal);
  signal.throwIfAborted();
  if (incremental && previous) {
    const buckets = new Map(previous.points.map(point => [point.bucketAt, point]));
    for (const point of data.points) buckets.set(point.bucketAt, point);
    data.points = [...buckets.values()].filter(point => point.bucketAt >= Math.ceil(data.from / 300_000) * 300_000 && point.bucketAt < data.to).sort((a, b) => a.bucketAt - b.bucketAt);
  }
  cachedSeries = { id, until: performance.now() + 300_000, data };
  return data;
}
export const getAlerts = (id: string, signal: AbortSignal) => get(`/api/v1/alerts?machineId=${encodeURIComponent(id)}`, alertsSchema, signal);
