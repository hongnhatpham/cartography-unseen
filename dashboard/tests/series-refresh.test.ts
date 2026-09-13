import { afterEach, expect, it, vi } from 'vitest';
import { getSeries } from '../src/api';

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it('keeps live polling from repeatedly scanning history, with expiry and explicit refresh', async () => {
  const clock = vi.spyOn(performance, 'now').mockReturnValue(0);
  const request = vi.fn(async () => new Response(JSON.stringify({
    machineId: 'history-refresh-test', from: 0, to: 1, resolution: '5m', points: [],
  }), { headers: { 'content-type': 'application/json' } }));
  vi.stubGlobal('fetch', request);
  const signal = new AbortController().signal;
  await getSeries('history-refresh-test', signal);
  for (let second = 15; second < 300; second += 15) {
    clock.mockReturnValue(second * 1000);
    await getSeries('history-refresh-test', signal);
  }
  expect(request).toHaveBeenCalledTimes(1);
  clock.mockReturnValue(300_000);
  await getSeries('history-refresh-test', signal);
  expect(request).toHaveBeenCalledTimes(2);
  await getSeries('history-refresh-test', signal, true);
  expect(request).toHaveBeenCalledTimes(3);
  await expect(getSeries('history-refresh-test', AbortSignal.abort())).rejects.toBeDefined();
  expect(request).toHaveBeenCalledTimes(3);
});

it('merges incremental and delayed buckets, prunes expired data, and preserves its cursor after failure', async () => {
  vi.spyOn(performance, 'now').mockReturnValue(0);
  vi.spyOn(Date, 'now').mockReturnValue(90_000_000);
  const response = (cursor: number, points: { bucketAt: number; generationFps: number; displayFps: number }[], from = 0) => new Response(JSON.stringify({ machineId: 'deltas', from, to: 90_000_000, resolution: '5m', cursor, points }), { headers: { 'content-type': 'application/json' } });
  const point = (bucketAt: number, generationFps = 20) => ({ bucketAt, generationFps, displayFps: 60 });
  const request = vi.fn().mockResolvedValueOnce(response(89_000_000, [point(0), point(300_000), point(600_000)]))
    .mockResolvedValueOnce(response(89_300_000, [point(300_000, 25), point(900_000)], 300_000))
    .mockResolvedValueOnce(new Response('', { status: 503 }))
    .mockResolvedValueOnce(response(89_600_000, [], 600_000));
  vi.stubGlobal('fetch', request);
  const signal = new AbortController().signal;
  await getSeries('deltas', signal);
  const merged = await getSeries('deltas', signal, true);
  expect(request.mock.calls[1][0]).toContain('&since=89000000');
  expect(merged.points).toEqual([point(300_000, 25), point(600_000), point(900_000)]);
  await expect(getSeries('deltas', signal, true)).rejects.toThrow();
  const pruned = await getSeries('deltas', signal, true);
  expect(request.mock.calls[3][0]).toContain('&since=89300000');
  expect(pruned.points).toEqual([point(600_000), point(900_000)]);
});
