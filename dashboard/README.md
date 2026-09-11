# Hosted monitor

One Cloudflare Worker receives machine health samples, serves the dashboard assets,
and exposes read-only human APIs and an ARIA alert feed. D1 stores credentials as SHA-256 hashes and keeps
seven days of raw samples. Nothing in this service runs commands on machines.
The dashboard uses the selected Projection desk design with plain-language incident
messages. It separates machine contact, artwork rendering and verified map backups.
See [`DESIGN.md`](DESIGN.md) for the approved visual and state conventions.

## Run locally

Use Node 24 or newer and pnpm 10. Tests use Node's SQLite implementation to execute
the actual D1 migration, triggers, and queries. No Cloudflare account is needed for
the tests or build.

```powershell
cd dashboard
pnpm install
pnpm test
pnpm typecheck
pnpm build
pnpm db:local
Copy-Item .dev.vars.example .dev.vars
pnpm dev:worker
```

The Worker listens on `http://127.0.0.1:8787`. `.dev.vars` sets that exact origin.
Human requests still require a valid Access assertion, including local requests.
Only tests inject a verifier. There is no production authentication bypass or
development password. `pnpm dev` serves the frontend on loopback for development;
it does not supply the API. Use the built assets through the Worker for an
authenticated integration. Opening Vite alone displays an unavailable/session
state because it cannot return authenticated telemetry.

## Dashboard behavior

The browser polls the read-only machines, series and alerts APIs every 15 seconds.
Machine contact ages from the server's receipt time using elapsed browser time,
so changing the browser's wall clock does not make a machine appear online.
Select a machine to inspect its latest sample, 24-hour generation trend and alerts.
The dashboard validates API payloads and keeps all requests on the same origin.
Machine credentials never enter the browser.

A live host does not prove that the artwork or uploader is healthy. Rendering
requires a recent artwork status observation, display activity and frame evidence.
Artwork status older than 20 seconds and uploader status older than 30 minutes at
the sample are treated as unknown, matching the collector's observation windows.
Late host reports, revoked credentials and failed monitor requests make current
artwork and upload health unknown. Last-known readings remain labelled as history.
Refresh now retries a failed request; a new successful check shows recovery.

The API does not report map capture activity, projector output, resolution,
uptime, machine hardware names or per-transfer percentage. The paired map readout
therefore says recording is not measured and shows the uploader's open/incomplete
map count. Transfer and verified byte counters are uploader-session totals, not
progress percentages. A past verification receipt does not verify pending maps.
No dashboard action runs a command on a machine.

`tests/ui.test.ts` covers freshness boundaries, independent signal states, null
measurements, stale uploader evidence, safe text rendering, history gaps and API
failure handling. Browser TypeScript uses `tsconfig.app.json` to keep DOM types
separate from Cloudflare's Worker globals; `pnpm typecheck` checks both targets.
Run `pnpm exec playwright install chromium` once, then `pnpm test:ui` for the real
browser focus regression. It checks authentication links and the refresh action
across periodic renders and automatic retries. Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE`
to use an existing Chrome executable instead of downloading the test browser.

## Provision a machine credential

Each machine gets a separate random 256-bit bearer token. The server derives the
machine ID from its hash lookup and rejects identity fields in request bodies.
Generate credentials on a trusted administrator computer.

```powershell
pnpm credential gallery-a "Gallery A"
pnpm exec wrangler d1 execute monitor --local --file credentials/gallery-a.sql
```

The generator writes the token to `credentials/gallery-a.json` and bootstrap SQL
containing only its SHA-256 hash to `credentials/gallery-a.sql`. It refuses to
overwrite existing files and never prints the token. Give the token file to the
collector through your private setup channel, then apply the collector's documented
credential configuration. Never place machine tokens in dashboard JavaScript,
URLs, shell arguments, screenshots, logs, or tracked config.

The generated directory is ignored by Git. On Windows, review its ACL and restrict
the token file to the collector or administrator account; POSIX file modes alone
do not enforce Windows ACLs. Keep the token outside any directory the archive
uploader scans. Remove the administrator's extra token copy after installation.

For a provisioned remote database, the bootstrap command uses `--remote` instead
of `--local`. This mutates production credentials, so run it only during an
authorized setup. Revocation takes effect on the next request:

```sql
UPDATE machines SET revoked_at = unixepoch() * 1000 WHERE id = 'gallery-a';
```

Rotation uses a newly generated token and updates `token_hash` on that same row,
with `revoked_at = NULL`. Do not replace the machine row, which owns its history.
Generate the replacement in a separate private working directory to preserve the
generator's no-overwrite rule. The token hash is safe to put in the rotation SQL;
the token itself stays in the private credential file.

## API contract

All payload object levels reject unknown fields. See
[`worker/schema.ts`](worker/schema.ts) for the complete v1 contract and limits.
All listed fields are required; unavailable measurements and process observations
use `null`. Booleans distinguish positively observed absence from unavailable data.
Samples contain counters, bounded status codes, and opaque archive IDs, with no
prompts, screenshots, file paths, hostnames, or exception text.

| Route | Authentication | Behavior |
| --- | --- | --- |
| `POST /api/v1/heartbeat` | Machine bearer token | One sample, at most 16 KiB. |
| `POST /api/v1/history` | Machine bearer token | `{schemaVersion:1,samples:[...]}`, 1 to 120 samples, at most 256 KiB. Never refreshes live state. |
| `GET /api/v1/machines` | Human Access JWT | Latest snapshot and receipt age, 200 machines per page, optional `after` cursor. |
| `GET /api/v1/series` | Human Access JWT | Required `machineId`; optional epoch-ms `from`, `to`, and `resolution=raw` or `5m`. |
| `GET /api/v1/alerts` | Human Access JWT | Latest 200 alert events; optional `machineId` and numeric `before` cursor. |
| `GET /api/v1/alert-feed` | Alert-reader bearer token | Ordered incident transitions, at most 200 per page; optional exclusive `after` event ID. |

Ingestion returns HTTP 202 with `{accepted,receivedAt,history}`. `accepted` counts
valid records in an idempotent request, including duplicates. Clients may retry
the same envelope after network failure. The primary key is machine ID, boot UUID,
and sequence. An insert trigger owns the latest-state update, so duplicate inserts
cannot refresh liveness. Each boot has a heartbeat sequence watermark. An older
sequence or older sample timestamp cannot overwrite newer state. History inserted
before a heartbeat of the same identity remains history and cannot become live.

The intended interval is 30 seconds. Connection age uses the server receipt time
of the last eligible heartbeat. A heartbeat older than 90 seconds can enter the
history window but cannot refresh liveness. A machine is online through 90 seconds,
stale above 90 through 180 seconds, and offline above 180 seconds. No heartbeat yet
means unknown. Machines include exact `ageSeconds` and `lastSeenAt`, and responses
include `serverTime`. Client timestamps more than 30 seconds in the future or more
than seven days old fail validation. Keep machine clocks synchronized.

Series default to the last 24 hours with five-minute means. SQL averages ignore
unknown measurements. Raw queries return at most 2,000 points and flag `truncated`;
use smaller time windows for the remaining data. The maximum query window is seven
days, at most 2,017 partial or complete five-minute buckets. The `bucket_at` index
and `rollups_5m` table leave space for a later retention policy; this version does
not claim to retain aggregates beyond seven days.

The every-minute scheduled task opens or resolves offline, app error/stalled/stopped,
archive error/retrying, disk under 10 GiB, and GPU at least 85 C alerts. Hardware and
app alerts open only from heartbeats received within 90 seconds. Recovery requires
fresh healthy evidence. Silence, null readings, artwork startup, and disabled
uploads do not resolve those incidents. Artwork recovery requires a running process,
status and frame observations at most 20 seconds old, and positive display FPS.
Upload recovery requires enabled uploads, a healthy upload phase and a status
observation at most 30 minutes old. An idle uploader may have exited normally.
Revoking a machine closes its incidents with reason `revoked`, not `recovered`.
Resolved dashboard incidents last 30 days. Unknown machine health does not create
a fabricated offline event.
Alert evaluation can lag receipt by a minute. Connection status is calculated on
each machines request, so it does not depend on cron timing.

## ARIA/Hermes Discord alerts

ARIA polls the Worker and uses its existing Discord integration to send notices.
It needs no public intake endpoint. The Worker does not hold a Discord credential
or send messages. An alert-reader credential can read only `/api/v1/alert-feed`;
it cannot ingest telemetry, read the dashboard, or call the human APIs. Reader
credentials grant access to incidents for all monitored machines.

Apply migration `0002_alert_feed.sql` before deploying this Worker version. It adds
the reader table, an incident journal and transactional triggers. It backfills
existing incident transitions, preserving incident IDs. Historical evidence is
`null`, and historical resolutions have reason `unknown` because older cron logic
could resolve an incident when reports stopped. Neither migration edits telemetry
nor replaces machine credentials.

Create ARIA's credential on the administrator computer:

```powershell
pnpm credential --alert-reader aria "ARIA Discord alerts"
pnpm exec wrangler d1 execute monitor --local --file credentials/alert-reader-aria.sql
```

For an authorized production setup, apply migrations and bootstrap SQL with
`--remote`. Privately install `credentials/alert-reader-aria.json` for ARIA/Hermes
and restrict its filesystem permissions. The JSON has `readerId`, `role` and
`token`; SQL contains only the hash. The generator never prints the token or
overwrites a file. Use `Authorization: Bearer <token>` from ARIA's secret storage,
never a query string. Revoke with:

```sql
UPDATE alert_readers SET revoked_at = unixepoch() * 1000 WHERE id = 'aria';
```

For rotation, generate a fresh reader credential in a private working directory
and update that reader's `token_hash` and `revoked_at = NULL`. Preserve ARIA's
cursor and dedupe state across rotation.

`GET /api/v1/alert-feed?after=0` returns:

```json
{
  "schemaVersion": 1,
  "serverTime": 1789106400000,
  "events": [{
    "id": 1,
    "alertId": 1,
    "machineId": "gallery-a",
    "machineLabel": "Gallery A",
    "code": "gpu_hot",
    "transition": "opened",
    "occurredAt": 1789106400000,
    "reason": null,
    "evidence": {
      "lastSeenAt": 1789106400000,
      "appState": "running",
      "appErrorCode": null,
      "archiveState": "idle",
      "archiveErrorCode": null,
      "diskFreeBytes": 21474836480,
      "gpuTemperatureC": 90
    }
  }],
  "nextCursor": 1,
  "hasMore": false
}
```

Times are epoch milliseconds. Event `id` is immutable and strictly increasing,
but IDs need not be contiguous. `alertId` links an opening and its eventual
resolution. A recurrence gets a new `alertId`. The journal snapshots the machine
label and listed evidence when cron records each transition. Evidence can be stale
for offline and revoked incidents; it is not a current health report.

The exclusive `after` cursor defaults to `0`. Follow `nextCursor` while `hasMore`
is true, then poll again after 60 seconds using the same cursor. An empty page
keeps the supplied cursor. Persist only a returned cursor after processing its
events. Retries return the same event IDs. Opening and recovery between polls
remain separate events, so ARIA cannot miss a recovery by querying only open rows.
The compact journal has no automatic expiry, even after dashboard incidents are
pruned. Keep it when backing up or migrating D1. An explicit future archival
policy must also define how consumers reset; do not manually prune this table.

Configure ARIA's recurring task with these rules:

1. Keep a durable cursor, event dedupe records and incident state scoped to this
   monitor URL. Use a single poller or a lock so runs do not overlap. On first use,
   replay all pages from `0` to reconstruct incident state, then announce only
   incidents still open. This avoids sending a backlog of historical recoveries.
2. On subsequent polls, process events in ID order. Send one Discord notice per
   `opened` event. Use the monitor URL and event ID as the dedupe key, and group
   follow-up notices by `alertId`. Include the machine label, incident code,
   occurrence time and relevant evidence, plus the human dashboard URL.
3. Send a recovery notice only for `resolved` with reason `recovered`. Say that
   an upload error cleared when its healthy phase returns; do not claim pending
   maps are backed up. Reason `revoked` means monitoring was administratively
   disabled, and `unknown` must never produce a healthy-recovery claim.
4. Save delivery results and advance the cursor after successful processing.
   Retry failures without advancing past an undelivered event. Respect HTTP 429
   `Retry-After`; retry network and 5xx errors with backoff. A 401 requires fixing
   the reader credential. A feed failure is a monitoring failure, never evidence
   that the exhibition recovered.
5. Check the target Discord channel's recent messages for the same event marker
   when delivery success is uncertain. Discord delivery and local checkpointing
   are not one transaction, so a crash after sending can otherwise duplicate a
   notice. The feed supports durable replay, not exactly-once Discord delivery.

Incident codes are `offline`, `app_error`, `archive_error`, `disk_low`, and
`gpu_hot`. Dedupe holds while a condition remains active. Threshold crossings after
a confirmed recovery are new incidents. Allow roughly one cron interval plus one
poll interval after a condition becomes eligible. An offline incident becomes
eligible after more than 180 seconds without a heartbeat.

## Cloudflare deployment checklist

No deployment, account resource creation, or DNS change is part of the local build.
Before an authorized deployment:

1. Create a D1 database and replace the zero `database_id` in `wrangler.jsonc`.
   Apply the migration with `pnpm exec wrangler d1 migrations apply monitor --remote`.
2. Choose the custom HTTPS hostname and set `PUBLIC_ORIGIN` to that exact origin
   without a trailing slash. Configure the custom domain in Cloudflare.
3. Configure a Cloudflare Access self-hosted application for the dashboard hostname
   with an explicit human allow policy. Set its team hostname in
   `ACCESS_TEAM_DOMAIN` and audience in `ACCESS_AUD`. These are public identifiers,
   not secrets. The Worker validates signature, issuer, audience, expiry and human
   identity for static assets and human read APIs. It never trusts an email header.
4. Configure Access path bypass applications for exactly `/api/v1/heartbeat` and
   `/api/v1/history`, and `/api/v1/alert-feed` when enabling ARIA. Collectors and
   ARIA reach these routes using their distinct bearer credentials. Keep the rest
   of the hostname behind human Access. Each bypassed path still requires its
   matching credential inside the Worker. Do not bypass `/api/v1/*` as a group.
5. Choose unused rate-limit namespace IDs in the account. The defaults allow 180
   requests per minute per client IP before authentication, 180 human requests per
   minute per human or alert-reader identity, 10 ingest requests per minute per machine, and 2 history
   requests per minute per machine. Cloudflare counters are per location, not a
   global billing cap. Add account-level WAF limits if your exposure needs them.
6. Run `pnpm test`, `pnpm typecheck`, `pnpm build`, and `pnpm build:worker`.
   Deploy with `pnpm exec wrangler deploy` only after deployment approval.
7. Provision each machine, verify invalid credentials fail, verify a human Access
   session can read, and stop a collector to confirm the 90/180-second states.

`workers_dev` and preview URLs are disabled. `run_worker_first: true` protects
static assets with the same authentication as the API. Requests require the exact
configured host, HTTPS outside loopback, and same-origin browser traffic. No
cross-origin CORS permissions are issued. JSON limits are enforced while streaming,
not just from `Content-Length`; compressed bodies are rejected. Responses disable
caching and framing. Logs are disabled by default and never include credentials
or sample bodies. The Worker needs only D1, static assets, three rate limit bindings,
and the three config strings above. There is no shared device secret binding.

Cloudflare references: [JWT validation](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/validating-json/),
[rate limit bindings](https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/),
and [static asset routing](https://developers.cloudflare.com/workers/static-assets/binding/).
