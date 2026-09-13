import './style.css';
import { ApiError, getAlerts, getMachines, getSeries } from './api';
import { connection, dateTime, duration, receiptAge, type Machine } from './model';
import { escape, renderMachine, type DetailState } from './render';

const main = document.querySelector<HTMLElement>('#main')!;
const picker = document.querySelector<HTMLSelectElement>('#machine')!;
const refreshButton = document.querySelector<HTMLButtonElement>('#refresh')!;
const refreshStatus = document.querySelector<HTMLElement>('#refresh-status')!;
const notice = document.querySelector<HTMLElement>('#notice')!;
const announcer = document.querySelector<HTMLElement>('#announcer')!;
const fleet = document.querySelector<HTMLElement>('#fleet')!;
const emptyDetails = (): DetailState => ({ series: null, alerts: null, seriesError: false, alertsError: false, loading: true });
let machines: Machine[] = [];
let selected = '';
let details = emptyDetails();
let receivedAt = 0;
let serverTime: number | null = null;
let error: 'auth' | 'network' | null = null;
let loaded = false;
let refreshing = false;
let recoveredUntil = 0;
let controller: AbortController | null = null;
let pollTimer: ReturnType<typeof setTimeout>;
let lastAnnouncement = '';

function render() {
  const elapsed = receivedAt ? Math.max(0, (performance.now() - receivedAt) / 1000) : 0;
  refreshButton.setAttribute('aria-busy', String(refreshing));
  refreshButton.textContent = refreshing ? 'Refreshing…' : 'Refresh now';
  refreshStatus.textContent = refreshing ? 'Checking for new reports' : serverTime === null ? 'No successful check yet' : `Checked ${duration(elapsed)} ago`;
  refreshStatus.title = serverTime === null ? '' : `Server time ${dateTime(serverTime)}`;
  const options = machines.map(machine => `<option value="${escape(machine.id)}">${escape(machine.label)}${machine.revoked ? ' · Revoked' : ` · ${connection(receiptAge(machine, elapsed))}`}</option>`).join('');
  // Do not replace a focused select or an open disclosure during polling.
  if (document.activeElement !== picker && picker.innerHTML !== options) picker.innerHTML = options || `<option>${loaded ? 'No machines' : 'Waiting for machines…'}</option>`;
  picker.disabled = machines.length === 0;
  if (selected) picker.value = selected;
  const online = machines.filter(machine => !machine.revoked && connection(receiptAge(machine, elapsed)) === 'online').length;
  fleet.textContent = machines.length ? `${machines.length} ${machines.length === 1 ? 'machine' : 'machines'} · ${error ? 'Current contact unknown' : `${online} reporting`}` : '';
  const noticeMarkup = error === 'auth' ? '<div class="notice">Your monitoring session needs attention. Sign in again to resume live checks. <a href="/">Sign in again</a></div>' : error === 'network' ? '<div class="notice">The monitor could not refresh. Check your connection and use Refresh now. Automatic checks will keep trying.</div>' : performance.now() < recoveredUntil ? '<div class="notice recovered">Connection to the monitor restored. New reports are available.</div>' : '';
  if (notice.innerHTML !== noticeMarkup) notice.innerHTML = noticeMarkup;
  const machine = machines.find(item => item.id === selected);
  if (!machine) {
    const emptyMarkup = `<section class="empty-state"><span class="eyebrow">Exhibition monitor</span><h1>${error === 'auth' ? 'Sign in to see the exhibition.' : error ? 'The monitor is unavailable.' : loaded ? 'No machines have been added yet.' : 'Waiting for the first response.'}</h1><p>${error === 'auth' ? 'Your Access session has ended or does not allow this dashboard. Sign in again with an approved account.' : error ? 'Machine contact, artwork activity and map backups cannot be checked. Check your connection, then refresh.' : loaded ? 'Provision a monitoring credential for the exhibition machine, then start its collector. It will appear here when setup is complete.' : 'Loading machine contact, artwork activity and map verification.'}</p>${error === 'auth' ? '<a href="/">Sign in again</a>' : ''}</section>`;
    if (main.innerHTML !== emptyMarkup) main.innerHTML = emptyMarkup;
    return;
  }
  const open = Array.from(main.querySelectorAll<HTMLDetailsElement>('details[open]')).map(item => item.id);
  const focusedDetails = document.activeElement?.closest('details')?.id;
  const rendered = renderMachine(machine, elapsed, error === null, details);
  main.innerHTML = rendered.html;
  for (const id of open) main.querySelector<HTMLDetailsElement>(`#${id}`)?.setAttribute('open', '');
  if (focusedDetails) main.querySelector<HTMLElement>(`#${focusedDetails} summary`)?.focus({ preventScroll: true });
  if (lastAnnouncement !== rendered.announcement) {
    announcer.textContent = rendered.announcement;
    lastAnnouncement = rendered.announcement;
  }
}

async function refresh(forceHistory = false) {
  if (document.hidden) return;
  const startedAt = performance.now();
  clearTimeout(pollTimer);
  controller?.abort();
  const request = new AbortController();
  controller = request;
  refreshing = true;
  render();
  const timeout = setTimeout(() => request.abort(new Error('Request timed out')), 12_000);
  try {
    const result = await getMachines(request.signal);
    if (controller !== request) return;
    machines = result.machines;
    serverTime = result.serverTime;
    receivedAt = performance.now();
    loaded = true;
    if (error) recoveredUntil = performance.now() + 20_000;
    error = null;
    if (!machines.some(machine => machine.id === selected)) {
      selected = machines.find(machine => !machine.revoked)?.id ?? machines[0]?.id ?? '';
      details = emptyDetails();
    }
    render();
    if (selected) {
      const id = selected;
      const [series, alerts] = await Promise.allSettled([getSeries(id, request.signal, forceHistory), getAlerts(id, request.signal)]);
      if (controller !== request || id !== selected) return;
      details.loading = false;
      details.seriesError = series.status === 'rejected';
      details.alertsError = alerts.status === 'rejected';
      if (series.status === 'fulfilled' && series.value.machineId === id) details.series = series.value;
      if (alerts.status === 'fulfilled') details.alerts = alerts.value;
      for (const result of [series, alerts]) {
        if (result.status === 'rejected' && result.reason instanceof ApiError && [401, 403].includes(result.reason.status)) error = 'auth';
      }
    }
  } catch (failure) {
    if (controller !== request) return;
    error = failure instanceof ApiError && [401, 403].includes(failure.status) ? 'auth' : 'network';
    details.loading = false;
  } finally {
    clearTimeout(timeout);
    if (controller === request) {
      refreshing = false;
      render();
      if (!document.hidden) pollTimer = setTimeout(() => { void refresh(); }, Math.max(0, 15_000 - (performance.now() - startedAt)));
    }
  }
}
picker.addEventListener('change', () => {
  selected = picker.value;
  details = emptyDetails();
  void refresh();
});
refreshButton.addEventListener('click', () => { void refresh(true); });
document.addEventListener('visibilitychange', () => { if (!document.hidden && !refreshing) void refresh(); });
window.addEventListener('online', () => { if (!refreshing) void refresh(); });
setInterval(() => { if (!document.hidden && !refreshing) render(); }, 5_000);
void refresh();
