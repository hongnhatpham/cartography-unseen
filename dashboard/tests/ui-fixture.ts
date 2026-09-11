import type { Sample } from '../worker/schema';
import type { Machine } from '../src/model';

export function uiMachine(now = Date.now()): Machine {
  const sample: Sample = {
    schemaVersion: 1, bootId: '2e100384-24de-4b48-aa51-b91d5b7a2b52', sequence: 100,
    sampledAt: new Date(now - 3000).toISOString(), agentVersion: '1.0.0',
    system: {
      cpuPercent: 24, ramUsedBytes: 14.7 * 1024 ** 3, ramTotalBytes: 32 * 1024 ** 3,
      diskFreeBytes: 162 * 1024 ** 3, diskTotalBytes: 512 * 1024 ** 3,
      diskReadBytesPerSecond: 1280, diskWriteBytesPerSecond: 89125,
      networkRxBytesPerSecond: 128000, networkTxBytesPerSecond: 240000,
      gpuPercent: 92, vramUsedBytes: 3.1 * 1024 ** 3, vramTotalBytes: 8 * 1024 ** 3, gpuTemperatureC: 72,
    },
    app: { state: 'running', processRunning: true, statusAgeSeconds: 0.2, displayFps: 59.9,
      generationFps: 9.8, lastFrameAgeSeconds: 0.12, errorCode: null },
    archiveSync: { enabled: true, processRunning: true, state: 'verifying', statusAgeSeconds: 1,
      pendingArchives: 2, pendingBytes: 91 * 1024 ** 2, verifiedLocalArchives: 3,
      incompleteArchives: 1, invalidArchives: 0, localBytes: 1.8 * 1024 ** 3,
      currentArchiveId: 'map-20260911-162401', completedPayloadBytes: 428 * 1024 ** 2,
      verifiedBytes: 164 * 1024 ** 2, lastVerifiedAt: new Date(now - 1200_000).toISOString(), errorCode: null },
  };
  return { id: 'exhibition-01', label: 'Exhibition 01', revoked: false, lastSeenAt: now - 3000, ageSeconds: 3, latest: sample };
}
