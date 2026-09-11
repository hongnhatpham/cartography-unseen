import type { Sample } from '../worker/schema';

export interface Machine {
  id: string; label: string; revoked: boolean; lastSeenAt: number | null;
  ageSeconds: number | null; latest: Sample | null;
}
export type Tone = 'good' | 'warn' | 'bad' | 'muted';
export interface Signal { title: string; detail: string; tone: Tone }
export type Connection = 'online' | 'stale' | 'offline' | 'unknown';

export function connection(ageSeconds: number | null): Connection {
  return ageSeconds === null ? 'unknown' : ageSeconds <= 90 ? 'online' : ageSeconds <= 180 ? 'stale' : 'offline';
}
export function receiptAge(machine: Machine, elapsedSeconds: number): number | null {
  return machine.ageSeconds === null ? null : machine.ageSeconds + Math.max(0, elapsedSeconds);
}
export function duration(seconds: number | null): string {
  if (seconds === null) return 'Unavailable';
  if (seconds < 1) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m`;
  return `${Math.floor(seconds / 86400)}d ${Math.floor(seconds % 86400 / 3600)}h`;
}
export function number(value: number | null | undefined, digits = 0): string {
  return value == null ? 'Unavailable' : value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}
export function bytes(value: number | null | undefined): string {
  if (value == null) return 'Unavailable';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  const i = value > 0 ? Math.min(4, Math.floor(Math.log(value) / Math.log(1024))) : 0;
  return `${number(value / 1024 ** i, i >= 2 ? 1 : 0)} ${units[i]}`;
}
export function percent(used: number | null | undefined, total: number | null | undefined): number | null {
  return used == null || total == null || total <= 0 ? null : Math.min(100, Math.max(0, used / total * 100));
}
export function dateTime(value: number | string | null | undefined): string {
  if (value == null) return 'Not reported';
  return new Date(value).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', timeZoneName: 'short' });
}

/** Positive activity requires a fresh collector observation, not just a live process. */
export function artwork(sample: Sample | null): Signal {
  const app = sample?.app;
  if (!app) return { title: 'Artwork unknown', detail: 'No artwork report', tone: 'muted' };
  if (app.processRunning === false || app.state === 'stopped') return { title: 'Artwork stopped', detail: 'Check the artwork process on site', tone: 'bad' };
  if (app.state === 'error' || app.state === 'stalled') return { title: `Artwork ${app.state}`, detail: 'Check the artwork process and display', tone: 'bad' };
  if (app.statusAgeSeconds === null || app.statusAgeSeconds > 20 || app.state === 'unknown') return { title: 'Artwork unknown', detail: 'No recent artwork observation', tone: 'warn' };
  if (app.state === 'starting') return { title: 'Artwork starting', detail: 'Rendering has not been confirmed', tone: 'warn' };
  if (app.displayFps === null || app.lastFrameAgeSeconds === null) return { title: 'Rendering unknown', detail: 'Frame or display reading unavailable', tone: 'warn' };
  if (app.displayFps <= 0 || app.lastFrameAgeSeconds > 20) return { title: 'Rendering needs a check', detail: 'No recent frame or display activity', tone: 'warn' };
  return { title: 'Artwork rendering', detail: `${number(app.displayFps, 1)} display FPS in the report`, tone: 'good' };
}

export function archive(sample: Sample | null): Signal {
  const sync = sample?.archiveSync;
  if (!sync) return { title: 'Uploads unknown', detail: 'No uploader report', tone: 'muted' };
  if (sync.enabled === false || sync.state === 'disabled') return { title: 'Uploads disabled', detail: 'Remote backup is not running', tone: 'warn' };
  if (sync.processRunning === false && sync.state !== 'idle') return { title: 'Uploader stopped', detail: 'Check the uploader process', tone: 'bad' };
  if (sync.statusAgeSeconds === null || sync.statusAgeSeconds > 1800 || sync.state === 'unknown') return { title: 'Uploads unknown', detail: 'No recent uploader observation', tone: 'warn' };
  if (sync.state === 'error') return { title: 'Upload failed', detail: 'Check the uploader and venue connection', tone: 'bad' };
  if (sync.state === 'retrying') return { title: 'Upload delayed', detail: 'Retry pending · Backup not yet confirmed', tone: 'warn' };
  if (sync.invalidArchives !== null && sync.invalidArchives > 0) return { title: 'Maps need a check', detail: `${number(sync.invalidArchives)} invalid · Backup not confirmed`, tone: 'warn' };
  if (sync.state === 'verifying') return { title: 'Verification in progress', detail: 'Transferred does not mean backed up', tone: 'good' };
  if (sync.state === 'uploading') return { title: 'Maps transferring', detail: 'Verification is still required', tone: 'good' };
  if (sync.state === 'packing' || sync.state === 'scanning') return { title: 'Preparing map uploads', detail: 'Backup not yet confirmed', tone: 'good' };
  if ((sync.pendingArchives ?? 0) > 0) return { title: 'Maps waiting', detail: `${number(sync.pendingArchives)} pending verification`, tone: 'warn' };
  if (sync.lastVerifiedAt !== null && sync.pendingArchives === 0) return { title: 'Latest backup verified', detail: 'No pending maps in the report', tone: 'good' };
  return { title: 'No backup confirmed', detail: 'Waiting for a verified map receipt', tone: 'muted' };
}

export function machineView(machine: Machine, elapsedSeconds = 0, feedAvailable = true) {
  const age = receiptAge(machine, elapsedSeconds);
  const contact = connection(age);
  const current = feedAvailable && !machine.revoked && contact === 'online' && machine.latest !== null;
  const host: Signal = !feedAvailable
    ? { title: 'Host status unknown', detail: 'The monitor could not refresh', tone: 'warn' }
    : machine.revoked ? { title: 'Credential revoked', detail: 'This machine cannot send new reports', tone: 'warn' }
    : { title: { online: 'Machine online', stale: 'Machine report late', offline: 'Connection lost', unknown: 'No contact yet' }[contact], detail: age === null ? 'Waiting for the first heartbeat' : `Last heard ${duration(age)} ago`, tone: contact === 'online' ? 'good' : contact === 'offline' ? 'bad' : 'warn' };
  const art = current ? artwork(machine.latest) : { title: 'Artwork unknown', detail: 'No current artwork report', tone: 'muted' as const };
  const maps = current ? archive(machine.latest) : { title: 'Uploads unknown', detail: 'No current verification report', tone: 'muted' as const };
  let headline: string;
  let guidance: string;
  if (!feedAvailable) {
    headline = 'The monitor cannot confirm the exhibition.';
    guidance = 'Check your connection and refresh. The readings below are from the last successful check.';
  } else if (machine.revoked) {
    headline = 'This machine can no longer report.';
    guidance = 'Its monitoring credential was revoked. Ask the administrator to check its setup.';
  } else if (contact === 'unknown') {
    headline = 'This machine has not reported yet.';
    guidance = "Check that the collector is running and has this machine's monitoring credential.";
  } else if (!current) {
    headline = contact === 'offline' ? 'The machine is unreachable. The artwork may still be running.' : 'The machine report is late. Current artwork and uploads are unknown.';
    guidance = 'Check the venue connection, then confirm the machine is powered on. Last-known readings remain below.';
  } else if (art.tone !== 'good') {
    headline = `${art.title}. The host is still connected.`;
    guidance = 'Check the artwork process and the display at the venue. Machine contact alone does not confirm rendering.';
  } else if (maps.tone === 'bad' || maps.tone === 'warn') {
    headline = `The artwork is rendering. ${maps.title}.`;
    guidance = maps.title === 'Upload delayed' ? 'The uploader reports a retry. Check its next report; if retries continue, check the venue connection and uploader. Pending maps are not verified backups.' : 'Check the uploader report and venue connection. Pending maps are not verified backups.';
  } else {
    const backup = { 'Verification in progress': 'A map is being verified.', 'Maps transferring': 'Map files are transferring.', 'Preparing map uploads': 'Map uploads are being prepared.', 'Latest backup verified': 'The latest map backup is verified.' }[maps.title] ?? 'No map backup has been confirmed.';
    headline = `The artwork is rendering. ${backup}`;
    guidance = maps.title === 'Latest backup verified' ? 'The latest completed backup was verified. Physical projector output still needs an on-site check.' : 'A map is backed up only after verification succeeds. Physical projector output still needs an on-site check.';
  }
  return { age, contact, current, host, art, maps, headline, guidance };
}

export const alertCopy: Record<string, { title: string; detail: string }> = {
  offline: { title: 'Machine stopped reporting', detail: 'Check the venue connection and machine power.' },
  app_error: { title: 'Artwork needs attention', detail: 'The artwork reported an error, stalled or stopped. Check the process and display.' },
  archive_error: { title: 'Upload interrupted', detail: 'The uploader reported an error or retry. Check its next report and the venue connection.' },
  disk_low: { title: 'Disk space below 10 GiB', detail: 'Check local storage before the machine runs out of room.' },
  gpu_hot: { title: 'GPU reached 85°C', detail: "Check ventilation and the machine's temperature at the venue." },
};
