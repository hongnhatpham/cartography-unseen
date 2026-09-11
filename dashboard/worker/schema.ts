import { z } from 'zod';

export const RAW_RETENTION_MS = 7 * 24 * 60 * 60 * 1000;
export const ONLINE_MS = 90_000;
export const OFFLINE_MS = 180_000;
export const HEARTBEAT_BYTES = 16 * 1024;
export const HISTORY_BYTES = 256 * 1024;

const number = (max: number) => z.number().finite().min(0).max(max).nullable();
const bytes = number(Number.MAX_SAFE_INTEGER);
const count = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER).nullable();
const percent = number(100);
const seconds = number(315_576_000);
const utc = z.iso.datetime({ precision: undefined, offset: false });
const code = z.string().regex(/^[A-Za-z0-9_.:-]{1,80}$/).nullable();

export const sampleSchema = z.strictObject({
  schemaVersion: z.literal(1),
  bootId: z.uuid(),
  sequence: z.number().int().min(0).max(Number.MAX_SAFE_INTEGER),
  sampledAt: utc,
  agentVersion: z.string().regex(/^[A-Za-z0-9_.+-]{1,40}$/),
  system: z.strictObject({
    cpuPercent: percent, ramUsedBytes: bytes, ramTotalBytes: bytes,
    diskFreeBytes: bytes, diskTotalBytes: bytes,
    diskReadBytesPerSecond: bytes, diskWriteBytesPerSecond: bytes,
    networkRxBytesPerSecond: bytes, networkTxBytesPerSecond: bytes,
    gpuPercent: percent, vramUsedBytes: bytes, vramTotalBytes: bytes,
    gpuTemperatureC: z.number().finite().min(-40).max(200).nullable(),
  }),
  app: z.strictObject({
    state: z.enum(['unknown', 'starting', 'running', 'stalled', 'stopped', 'error']),
    processRunning: z.boolean().nullable(), statusAgeSeconds: seconds,
    displayFps: number(10_000), generationFps: number(10_000),
    lastFrameAgeSeconds: seconds, errorCode: code,
  }),
  archiveSync: z.strictObject({
    enabled: z.boolean().nullable(), processRunning: z.boolean().nullable(),
    state: z.enum(['unknown', 'disabled', 'idle', 'scanning', 'packing', 'uploading', 'verifying', 'pruning', 'retrying', 'error']),
    statusAgeSeconds: seconds, pendingArchives: count, pendingBytes: bytes,
    verifiedLocalArchives: count, incompleteArchives: count, invalidArchives: count,
    localBytes: bytes, currentArchiveId: z.string().regex(/^[A-Za-z0-9_.-]{1,128}$/).nullable(),
    completedPayloadBytes: bytes, verifiedBytes: bytes,
    lastVerifiedAt: utc.nullable(), errorCode: code,
  }),
});

export const historySchema = z.strictObject({ schemaVersion: z.literal(1), samples: z.array(sampleSchema).min(1).max(120) });
export type Sample = z.infer<typeof sampleSchema>;

export function connectionStatus(lastSeenAt: number | null, now: number) {
  if (lastSeenAt === null) return 'unknown';
  const age = now - lastSeenAt;
  return age <= ONLINE_MS ? 'online' : age <= OFFLINE_MS ? 'stale' : 'offline';
}
