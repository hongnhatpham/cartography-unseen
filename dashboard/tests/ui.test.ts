import { describe, expect, it, vi, afterEach } from 'vitest';
import { archive, artwork, bytes, connection, machineView, percent, receiptAge } from '../src/model';
import { chartPaths, escape, renderMachine } from '../src/render';
import { ApiError, getMachines, type Series } from '../src/api';
import { uiMachine } from './ui-fixture';

const details = { series: null, alerts: null, seriesError: false, alertsError: false, loading: false };
afterEach(() => vi.unstubAllGlobals());

describe('independent exhibition health', () => {
  it('uses exact server-receipt boundaries, including aging between checks', () => {
    expect([null, 90, 90.01, 180, 180.01].map(connection)).toEqual(['unknown', 'online', 'stale', 'stale', 'offline']);
    const machine = uiMachine();
    machine.ageSeconds = 89;
    expect(connection(receiptAge(machine, 2))).toBe('stale');
    expect(machineView(machine, 2).art.title).toBe('Artwork unknown');
  });
  it('never carries positive artwork or backup state into lost contact, revoked credentials or a failed refresh', () => {
    const machine = uiMachine();
    for (const view of [machineView(machine, 180), machineView({ ...machine, revoked: true }), machineView(machine, 0, false)]) {
      expect(view.current).toBe(false);
      expect(view.art.title).toBe('Artwork unknown');
      expect(view.maps.title).toBe('Uploads unknown');
    }
    expect(machineView({ ...machine, lastSeenAt: null, ageSeconds: null, latest: null }).headline).toContain('has not reported');
  });
  it('requires render evidence beyond a connected host or running process', () => {
    const sample = uiMachine().latest!;
    expect(artwork(sample).title).toBe('Artwork rendering');
    for (const app of [{ displayFps: null }, { displayFps: 0 }, { lastFrameAgeSeconds: null }, { lastFrameAgeSeconds: 21 }, { statusAgeSeconds: 21 }, { processRunning: false }]) {
      expect(artwork({ ...sample, app: { ...sample.app, ...app } }).tone).not.toBe('good');
    }
  });
  it('keeps transferring, verifying, and verified receipts distinct', () => {
    const sample = uiMachine().latest!;
    expect(archive(sample).title).toBe('Verification in progress');
    const withSync = (change: Partial<typeof sample.archiveSync>) => ({ ...sample, archiveSync: { ...sample.archiveSync, ...change } });
    expect(archive(withSync({ state: 'uploading' })).title).toBe('Maps transferring');
    expect(archive(withSync({ state: 'idle', pendingArchives: 0 })).title).toBe('Latest backup verified');
    expect(archive(withSync({ state: 'idle', pendingArchives: 0, lastVerifiedAt: null })).title).toBe('No backup confirmed');
    expect(archive(withSync({ state: 'retrying' })).title).toBe('Upload delayed');
    expect(archive(withSync({ state: 'idle', invalidArchives: 1 })).tone).toBe('warn');
    expect(archive(withSync({ statusAgeSeconds: 1801 })).title).toBe('Uploads unknown');
  });
  it('does not turn unavailable measurements into zero or fabricate map activity', () => {
    expect(bytes(null)).toBe('Unavailable');
    expect(bytes(0)).toBe('0 B');
    expect(percent(0, 0)).toBeNull();
    expect(percent(null, 10)).toBeNull();
    const rendered = renderMachine(uiMachine(), 0, true, details).html;
    expect(rendered).toContain('Recording not measured');
    expect(rendered).toContain('Transfer completed this session');
    expect(rendered).not.toContain('% uploaded');
  });
  it('marks stale uploader observations historical throughout the upload card', () => {
    const machine = uiMachine();
    machine.latest!.archiveSync.statusAgeSeconds = 1801;
    machine.latest!.archiveSync.state = 'uploading';
    const rendered = renderMachine(machine, 0, true, details).html;
    expect(rendered).toContain('Upload state unknown');
    expect(rendered).toContain('last uploader observation');
    expect(rendered).not.toContain('aria-current="step"');
    expect(rendered).not.toContain('>Transferring files</h3>');
  });
});

describe('read-only rendering and API handling', () => {
  it('escapes labels and opaque archive IDs in HTML', () => {
    const machine = uiMachine();
    machine.label = '<img src=x onerror=alert(1)>';
    const rendered = renderMachine(machine, 0, true, details).html;
    expect(rendered).not.toContain(machine.label);
    expect(rendered).toContain(escape(machine.label));
    expect(rendered).not.toContain('onclick=');
  });
  it('leaves gaps in history instead of joining absent observations', () => {
    const series: Series = { machineId: 'a', from: 0, to: 1800_000, resolution: '5m', points: [
      { bucketAt: 0, generationFps: 10, displayFps: 60 },
      { bucketAt: 300_000, generationFps: 11, displayFps: 60 },
      { bucketAt: 600_000, generationFps: null, displayFps: null },
      { bucketAt: 900_000, generationFps: 10, displayFps: 60 },
      { bucketAt: 1500_000, generationFps: 9, displayFps: 60 },
    ] };
    expect(chartPaths(series)).toHaveLength(3);
  });
  it('recognizes a successful HTML login response as expired authentication', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>Sign in</html>', { headers: { 'Content-Type': 'text/html' } })));
    await expect(getMachines(new AbortController().signal)).rejects.toMatchObject({ status: 401 });
  });
  it('rejects invalid telemetry and keeps server-clock age across machine pages', async () => {
    const machine = uiMachine(10_000);
    const fetcher = vi.fn().mockResolvedValueOnce(Response.json({ serverTime: 10_000, machines: [machine], nextCursor: 'exhibition-01' }))
      .mockResolvedValueOnce(Response.json({ serverTime: 12_000, machines: [], nextCursor: null }));
    vi.stubGlobal('fetch', fetcher);
    const result = await getMachines(new AbortController().signal);
    expect(result.machines[0].ageSeconds).toBe(5);
    expect(fetcher.mock.calls[1][0]).toContain('after=exhibition-01');
    fetcher.mockResolvedValueOnce(Response.json({ serverTime: 1, machines: [{ ...machine, latest: { fake: true } }], nextCursor: null }));
    await expect(getMachines(new AbortController().signal)).rejects.toThrow();
    fetcher.mockResolvedValueOnce(new Response('', { status: 503 }));
    await expect(getMachines(new AbortController().signal)).rejects.toBeInstanceOf(ApiError);
  });
});
