-- A separate credential grants only access to the incident journal.
CREATE TABLE alert_readers (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE CHECK(length(token_hash) = 64),
  created_at INTEGER NOT NULL,
  revoked_at INTEGER
);

ALTER TABLE alerts ADD COLUMN resolution_reason TEXT
  CHECK(resolution_reason IN ('recovered', 'revoked', 'unknown'));
UPDATE alerts SET resolution_reason = 'unknown' WHERE resolved_at IS NOT NULL;

-- No foreign key: removing an old alert must not erase a consumer's unread event.
-- This compact journal has no automatic expiry. IDs are never reused.
CREATE TABLE alert_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  alert_id INTEGER NOT NULL,
  machine_id TEXT NOT NULL,
  machine_label TEXT NOT NULL,
  code TEXT NOT NULL,
  transition TEXT NOT NULL CHECK(transition IN ('opened', 'resolved')),
  occurred_at INTEGER NOT NULL,
  reason TEXT CHECK(reason IN ('recovered', 'revoked', 'unknown')),
  evidence_json TEXT,
  UNIQUE(alert_id, transition)
);

-- Older resolutions were not necessarily confirmed recoveries. Consumers must
-- not announce recovery for reason=unknown. No historical evidence is invented.
INSERT INTO alert_events(alert_id, machine_id, machine_label, code, transition, occurred_at, reason)
SELECT alert_id, machine_id, label, code, transition, occurred_at, reason FROM (
  SELECT a.id AS alert_id, a.machine_id, m.label, a.code, 'opened' AS transition,
    a.opened_at AS occurred_at, NULL AS reason
  FROM alerts a JOIN machines m ON m.id = a.machine_id
  UNION ALL
  SELECT a.id, a.machine_id, m.label, a.code, 'resolved', a.resolved_at, 'unknown'
  FROM alerts a JOIN machines m ON m.id = a.machine_id WHERE a.resolved_at IS NOT NULL
) ORDER BY occurred_at, alert_id, transition;

CREATE VIEW alert_evidence AS
SELECT id AS machine_id, label,
  json_object(
    'lastSeenAt', last_seen_at,
    'appState', json_extract(latest_json, '$.app.state'),
    'appErrorCode', json_extract(latest_json, '$.app.errorCode'),
    'archiveState', json_extract(latest_json, '$.archiveSync.state'),
    'archiveErrorCode', json_extract(latest_json, '$.archiveSync.errorCode'),
    'diskFreeBytes', json_extract(latest_json, '$.system.diskFreeBytes'),
    'gpuTemperatureC', json_extract(latest_json, '$.system.gpuTemperatureC')
  ) AS evidence_json
FROM machines;

CREATE TRIGGER alert_opened AFTER INSERT ON alerts
BEGIN
  INSERT INTO alert_events(alert_id, machine_id, machine_label, code, transition, occurred_at, evidence_json)
  SELECT NEW.id, NEW.machine_id, label, NEW.code, 'opened', NEW.opened_at, evidence_json
  FROM alert_evidence WHERE machine_id = NEW.machine_id;
END;

CREATE TRIGGER alert_resolved AFTER UPDATE OF resolved_at ON alerts
WHEN OLD.resolved_at IS NULL AND NEW.resolved_at IS NOT NULL
BEGIN
  INSERT INTO alert_events(alert_id, machine_id, machine_label, code, transition, occurred_at, reason, evidence_json)
  SELECT NEW.id, NEW.machine_id, label, NEW.code, 'resolved', NEW.resolved_at,
    COALESCE(NEW.resolution_reason, 'unknown'), evidence_json
  FROM alert_evidence WHERE machine_id = NEW.machine_id;
END;
