import { alertCopy, bytes, dateTime, duration, machineView, number, percent, type Machine, type Signal } from './model';
import type { Alerts, Series } from './api';

export function escape(value: unknown): string {
  return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]!));
}
const e = escape;
export interface DetailState { series: Series | null; alerts: Alerts | null; seriesError: boolean; alertsError: boolean; loading: boolean }
const signal = (label: string, item: Signal) => `<div class="healthcell"><span class="status-mark ${item.tone}" aria-hidden="true"></span><div><span class="eyebrow">${label}</span><strong class="${item.tone}">${e(item.title)}</strong><small>${e(item.detail)}</small></div></div>`;
const pair = (label: string, value: string) => `<div><dt>${e(label)}</dt><dd>${e(value)}</dd></div>`;
const unit = (value: number | null | undefined, suffix: string, digits = 0) => value == null ? 'Unavailable' : `${number(value, digits)}${suffix}`;
const ratio = (used: number | null | undefined, total: number | null | undefined) => used == null || total == null ? 'Unavailable' : `${bytes(used)} / ${bytes(total)}`;

function meter(label: string, amount: number | null | undefined, value: string, note = '') {
  return `<div class="metric"><div>${e(label)}${note ? `<small>${e(note)}</small>` : ''}</div><svg class="meter" viewBox="0 0 100 4" preserveAspectRatio="none" aria-hidden="true"><rect width="100" height="4" class="meter-track"/>${amount == null ? '' : `<rect width="${Math.max(0, Math.min(100, amount))}" height="4" class="meter-fill"/>`}</svg><span class="mono">${e(value)}</span></div>`;
}

/** Gaps and nulls break the trace; time position is never compressed to fill missing reports. */
export function chartPaths(series: Series): string[] {
  const values = series.points.flatMap(point => point.generationFps === null ? [] : [point.generationFps]);
  const max = Math.max(1, ...values);
  const paths: string[] = [];
  let path = '';
  let previous: number | null = null;
  for (const point of series.points) {
    if (point.generationFps === null || previous !== null && point.bucketAt - previous > 300_000) {
      if (path) paths.push(path);
      path = '';
    }
    if (point.generationFps !== null) {
      const x = Math.max(0, Math.min(500, (point.bucketAt - series.from) / (series.to - series.from) * 500));
      const y = 43 - point.generationFps / max * 36;
      path += `${path ? ' L' : 'M'}${x.toFixed(2)} ${y.toFixed(2)}`;
    }
    previous = point.bucketAt;
  }
  if (path) paths.push(path);
  return paths;
}
function chart(details: DetailState) {
  const series = details.series;
  if (!series) return `<p class="chart-empty">${details.seriesError ? 'History unavailable. Refresh to try again.' : details.loading ? 'Loading generation history…' : 'No generation history yet.'}</p>`;
  const values = series.points.flatMap(p => p.generationFps === null ? [] : [p.generationFps]);
  if (values.length < 2) return `<p class="chart-empty">${details.seriesError ? 'History could not refresh. ' : ''}Not enough generation readings to draw a trend.</p>`;
  return `<div class="trend"><div class="label-row"><span>Generation · 24 hours</span><span>0–${number(Math.max(1, ...values), 1)} FPS</span></div><svg class="trace" viewBox="0 0 500 50" preserveAspectRatio="none" role="img" aria-label="Generation FPS over the last 24 hours, five-minute averages. Range ${number(Math.min(...values), 1)} to ${number(Math.max(...values), 1)} FPS. Gaps mean no data."><line x1="0" y1="43" x2="500" y2="43"/>${chartPaths(series).map(path => `<path d="${path}"/>`).join('')}</svg><div class="label-row"><span>5-minute averages · Gaps mean no data</span><span>${details.seriesError ? 'History could not refresh' : `Through ${e(new Date(series.to).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }))}`}</span></div></div>`;
}

function alerts(details: DetailState) {
  const events = details.alerts?.alerts ?? [];
  const activeCount = events.filter(event => event.resolvedAt === null).length;
  return `<section class="events" aria-labelledby="alerts-title"><div class="section-head"><h2 id="alerts-title">Recent alerts</h2><span class="tag">${details.alerts ? events.length ? `${activeCount} open · ${events.length} recent` : 'No events' : 'Awaiting report'}</span></div><p class="section-note">Checks can lag a minute. Resolved means the alert condition cleared; it does not confirm recovery.</p>${details.alertsError ? '<p class="inline-warning">Alerts could not refresh. Any events below are from the previous check.</p>' : ''}${!details.alerts ? `<p class="empty-note">${details.loading ? 'Loading alerts…' : 'Alert history is unavailable.'}</p>` : events.length === 0 ? '<p class="empty-note">No recent alerts for this machine.</p>' : `<ol class="event-list">${events.slice(0, 8).map(event => {
    const copy = alertCopy[event.code] ?? { title: 'Machine alert', detail: 'Check the latest machine report.' };
    return `<li class="event"><div class="event-top"><strong>${e(copy.title)}</strong><span class="${event.resolvedAt === null ? 'warn' : 'muted'}">${event.resolvedAt === null ? 'Open' : 'Resolved'}</span></div><p>${e(copy.detail)}</p><time datetime="${new Date(event.openedAt).toISOString()}">Opened ${e(dateTime(event.openedAt))}</time>${event.resolvedAt === null ? '' : `<time datetime="${new Date(event.resolvedAt).toISOString()}">Resolved ${e(dateTime(event.resolvedAt))}</time>`}</li>`;
  }).join('')}</ol>${events.length > 8 || details.alerts?.nextCursor !== null ? '<p class="section-note">Showing the eight most recent events. Older events remain in the monitoring API.</p>' : ''}`}</section>`;
}

export function renderMachine(machine: Machine, elapsed: number, available: boolean, details: DetailState): { html: string; announcement: string } {
  const view = machineView(machine, elapsed, available);
  const sample = machine.latest;
  const system = sample?.system;
  const app = sample?.app;
  const sync = sample?.archiveSync;
  const uploaderCurrent = view.current && sync !== undefined && sync.statusAgeSeconds !== null && sync.statusAgeSeconds <= 1800 && sync.state !== 'unknown';
  const readoutLabel = view.current ? 'Latest report' : 'Last-known report';
  const ageNote = view.current ? `Received ${duration(view.age)} ago · Measurements describe that report` : `Received ${dateTime(machine.lastSeenAt)} · Current activity cannot be confirmed`;
  const phase = sync?.state ?? 'unknown';
  const phaseName: Record<string, string> = { unknown: 'Not reported', disabled: 'Disabled', idle: 'Idle', scanning: 'Scanning maps', packing: 'Preparing a map', uploading: 'Transferring files', verifying: 'Verifying remote files', pruning: 'Cleaning verified cache', retrying: 'Waiting to retry', error: 'Upload failed' };
  const uploadDescription = !uploaderCurrent ? 'Current progress and local file state cannot be confirmed. Values below are from the last uploader observation.' : phase === 'verifying' ? 'Remote checks are running. This map is not a confirmed backup until verification finishes.' : phase === 'retrying' ? 'The uploader is waiting to retry. Its next retry time and transfer percentage are not reported.' : phase === 'uploading' ? 'Files are transferring. Verification must succeed before the map counts as backed up.' : phase === 'error' ? 'The uploader reported a failure. Check its status and the venue connection.' : 'The uploader reports transfer and verification separately. Use the last verified receipt to confirm a completed backup.';
  const diskUsed = system?.diskTotalBytes == null || system.diskFreeBytes == null ? null : system.diskTotalBytes - system.diskFreeBytes;
  return {
    announcement: `${machine.label}. ${view.host.title}. ${view.art.title}. ${view.maps.title}.`,
    html: `<div class="healthline" aria-label="Independent health signals">${signal('Host contact', view.host)}${signal('Artwork', view.art)}${signal('Map backup', view.maps)}</div>
      <section class="incident"><span class="eyebrow">${e(machine.label)}</span><h1>${e(view.headline)}</h1><p>${e(view.guidance)}</p></section>
      <div class="work ${view.current ? '' : 'historical'}">
        <section class="readings" aria-labelledby="readings-title"><div class="section-head"><h2 id="readings-title">${readoutLabel}</h2><span class="tag ${view.current ? '' : 'warn'}">${view.current ? 'Reported readings' : 'Historical readings'}</span></div><p class="report-age">${e(ageNote)}</p>
          <div class="projection-strip">
            <section class="projection"><span class="eyebrow">Artwork projection</span><h3>${view.current ? e(view.art.title) : 'Last reported activity'}</h3><div class="readout mono">${app?.generationFps == null ? '<span class="unavailable">Unavailable</span>' : number(app.generationFps, 1)}<small>generation FPS</small></div><p><b class="mono">${e(unit(app?.displayFps, ' FPS', 1))}</b> display refresh</p><p>Frame age <b class="mono">${e(duration(app?.lastFrameAgeSeconds ?? null))}</b> at sample</p></section>
            <section class="projection map-projection"><span class="eyebrow">Map projection</span><h3>Recording not measured</h3><div class="readout mono">${sync?.incompleteArchives == null ? '<span class="unavailable">Unavailable</span>' : number(sync.incompleteArchives)}<small>open / incomplete maps</small></div><p>${uploaderCurrent ? 'Reported by the uploader' : 'Last-known uploader count'}</p><p>Capture activity and map display are not reported.</p></section>
          </div>
          ${chart(details)}
          <section class="metrics" aria-labelledby="resource-title"><div class="section-head"><h2 id="resource-title">Machine resources</h2><span class="section-note">${view.current ? 'At latest sample' : 'At last-known sample'}</span></div>
          ${meter('GPU', system?.gpuPercent, unit(system?.gpuPercent, '%'))}
          ${meter('CPU', system?.cpuPercent, unit(system?.cpuPercent, '%'))}
          ${meter('RAM', percent(system?.ramUsedBytes, system?.ramTotalBytes), ratio(system?.ramUsedBytes, system?.ramTotalBytes))}
          ${meter('VRAM', percent(system?.vramUsedBytes, system?.vramTotalBytes), ratio(system?.vramUsedBytes, system?.vramTotalBytes))}
          ${meter('Local disk', percent(diskUsed, system?.diskTotalBytes), `${bytes(system?.diskFreeBytes)}${system?.diskFreeBytes == null ? '' : ' free'}`, 'Used capacity')}
          <dl class="resource-extras">${pair('GPU temperature', unit(system?.gpuTemperatureC, '°C', 1))}${pair('Network upload', system?.networkTxBytesPerSecond == null ? 'Unavailable' : `${bytes(system.networkTxBytesPerSecond)}/s`)}${pair('Network download', system?.networkRxBytesPerSecond == null ? 'Unavailable' : `${bytes(system.networkRxBytesPerSecond)}/s`)}</dl></section>
          <details id="machine-evidence"><summary>Inspect ${view.current ? 'machine' : 'last-known machine'} evidence</summary><dl class="evidence">${pair('Sample time', dateTime(sample?.sampledAt))}${pair('Server receipt', dateTime(machine.lastSeenAt))}${pair('Artwork process', app?.processRunning == null ? 'Unknown' : app.processRunning ? 'Reported running' : 'Reported stopped')}${pair('Artwork status age at sample', duration(app?.statusAgeSeconds ?? null))}${pair('Artwork status', app?.state ?? 'Not reported')}${pair('Artwork error code', app ? app.errorCode ?? 'None reported' : 'Unavailable')}${pair('Disk read', system?.diskReadBytesPerSecond == null ? 'Unavailable' : `${bytes(system.diskReadBytesPerSecond)}/s`)}${pair('Disk write', system?.diskWriteBytesPerSecond == null ? 'Unavailable' : `${bytes(system.diskWriteBytesPerSecond)}/s`)}</dl><p>Frame and process reports cannot confirm the physical projector output. Host uptime, resolution and open windows are not reported.</p></details>
        </section>
        <aside><section class="upload" aria-labelledby="upload-title"><div class="section-head"><h2 id="upload-title">Map uploads</h2><span class="tag">${uploaderCurrent ? 'Latest observation' : 'Last known'}</span></div><h3 class="upload-state ${view.maps.tone}">${e(uploaderCurrent ? phaseName[phase] : 'Upload state unknown')}</h3><p>${e(uploadDescription)}</p>
          <ol class="phase-list" aria-label="Backup stages, highlighted stage is the latest reported phase">${[['prepare', 'Prepare'], ['uploading', 'Transfer'], ['verifying', 'Verify'], ['complete', 'Backed up']].map(([key, label]) => `<li${uploaderCurrent && (phase === key || key === 'prepare' && ['packing', 'scanning'].includes(phase)) ? ' class="current" aria-current="step"' : ''}>${label}</li>`).join('')}</ol>
          <dl class="queue">${pair('Pending maps', number(sync?.pendingArchives))}${pair('Verified locally', number(sync?.verifiedLocalArchives))}${pair('Open / incomplete', number(sync?.incompleteArchives))}${pair('Invalid maps', number(sync?.invalidArchives))}</dl>
          <div class="verification"><span class="eyebrow">Last verified backup</span><strong>${e(sync?.lastVerifiedAt ? dateTime(sync.lastVerifiedAt) : 'No verified receipt reported')}</strong><p>${sync?.lastVerifiedAt ? 'This receipt confirms a past backup. It does not verify pending maps.' : 'Transfer completion alone does not confirm a safe remote copy.'}</p></div>
          <details id="upload-evidence"><summary>Inspect upload and verification evidence</summary><dl class="evidence">${pair('Reported phase', phaseName[phase])}${pair('Uploader status age at sample', duration(sync?.statusAgeSeconds ?? null))}${pair('Uploader process', sync?.processRunning == null ? 'Unknown' : sync.processRunning ? 'Reported running' : 'Reported stopped')}${pair('Current archive ID', sync?.currentArchiveId ?? 'Not reported')}${pair('Transfer completed this session', bytes(sync?.completedPayloadBytes))}${pair('Verified bytes this session', bytes(sync?.verifiedBytes))}${pair('Pending bytes', bytes(sync?.pendingBytes))}${pair('Local cache', bytes(sync?.localBytes))}${pair('Uploader error code', sync ? sync.errorCode ?? 'None reported' : 'Unavailable')}</dl><p>Session byte counters can reset when the uploader restarts. They are not transfer percentages or verified-map counts. The uploader retains unverified completed maps until verification succeeds; this monitor does not inspect local files.</p></details>
        </section>${alerts(details)}</aside>
      </div>`,
  };
}
