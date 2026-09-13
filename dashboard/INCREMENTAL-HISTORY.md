# Incremental monitoring history

The initial chart loads the last 24 hours. Live machine status refreshes every
15 seconds; chart updates run at most once every five minutes. Hidden tabs stop
polling. Explicit refresh bypasses the timer but still uses the receipt cursor.

Subsequent chart requests send `since`, the server's previous receipt cursor.
`samples_machine_received` restricts the read to newly received records. The
Worker then recalculates only affected five-minute buckets using `samples_rollup`.
Late records replayed after an outage therefore repair old chart buckets as well
as adding recent ones. Bucket values replace cached values rather than being
added twice. Expired points are discarded. The cache contains one machine's
24-hour chart; a new page, another machine, or a cursor older than six days starts
a fresh initial load.

The receipt cursor overlaps 30 seconds to accommodate allowed device clock skew
and equal-millisecond writes. Failed/aborted requests never advance the cache.
Initial reads explicitly use the machine/sample-time index so SQLite cannot
choose a full machine-history scan just to avoid sorting grouped results.

Deploy migration `0003_incremental_series.sql` before deploying this Worker.
Regression tests execute the real SQL with SQLite and inspect its query plans.
They cover delayed history, cursor boundaries, device clock lead, merge/pruning,
failed requests, cache timing, and browser visibility/focus behavior.

This reduces chart reads; it does not impose an account-wide spending cap.
Cloudflare billing includes other applications and products on the account.
