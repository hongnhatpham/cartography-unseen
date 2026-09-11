PRAGMA foreign_keys = ON;

CREATE TABLE machines (
  id TEXT PRIMARY KEY,
  label TEXT NOT NULL,
  token_hash TEXT NOT NULL UNIQUE CHECK(length(token_hash) = 64),
  revoked_at INTEGER,
  created_at INTEGER NOT NULL,
  last_seen_at INTEGER,
  last_sampled_at INTEGER,
  last_boot_id TEXT,
  last_sequence INTEGER,
  latest_json TEXT
);

CREATE TABLE machine_boots (
  machine_id TEXT NOT NULL REFERENCES machines(id) ON DELETE CASCADE,
  boot_id TEXT NOT NULL,
  live_sequence INTEGER NOT NULL,
  last_received_at INTEGER NOT NULL,
  PRIMARY KEY(machine_id, boot_id)
);

CREATE TABLE samples (
  machine_id TEXT NOT NULL REFERENCES machines(id) ON DELETE CASCADE,
  boot_id TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  sampled_at INTEGER NOT NULL,
  received_at INTEGER NOT NULL,
  bucket_at INTEGER NOT NULL,
  source TEXT NOT NULL CHECK(source IN ('heartbeat', 'history')),
  live_eligible INTEGER NOT NULL CHECK(live_eligible IN (0, 1)),
  payload_json TEXT NOT NULL,
  PRIMARY KEY(machine_id, boot_id, sequence)
);
CREATE INDEX samples_machine_time ON samples(machine_id, sampled_at);
CREATE INDEX samples_retention ON samples(sampled_at);
CREATE INDEX samples_rollup ON samples(machine_id, bucket_at);

-- One insert transaction owns both deduplication and latest state. Duplicate
-- INSERT OR IGNORE requests never execute this trigger or refresh liveness.
CREATE TRIGGER sample_latest AFTER INSERT ON samples
WHEN NEW.source = 'heartbeat' AND NEW.live_eligible = 1
BEGIN
  UPDATE machines SET last_seen_at = NEW.received_at,
    last_sampled_at = NEW.sampled_at, last_boot_id = NEW.boot_id,
    last_sequence = NEW.sequence, latest_json = NEW.payload_json
  WHERE id = NEW.machine_id AND revoked_at IS NULL
    AND (last_seen_at IS NULL OR NEW.received_at >= last_seen_at)
    AND (last_sampled_at IS NULL OR NEW.sampled_at >= last_sampled_at)
    AND NOT EXISTS (
      SELECT 1 FROM machine_boots WHERE machine_id = NEW.machine_id
      AND boot_id = NEW.boot_id AND live_sequence >= NEW.sequence
    );
  INSERT INTO machine_boots(machine_id, boot_id, live_sequence, last_received_at)
  VALUES(NEW.machine_id, NEW.boot_id, NEW.sequence, NEW.received_at)
  ON CONFLICT(machine_id, boot_id) DO UPDATE SET
    live_sequence = MAX(live_sequence, NEW.sequence),
    last_received_at = MAX(last_received_at, NEW.received_at);
END;

CREATE TABLE alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  machine_id TEXT NOT NULL REFERENCES machines(id) ON DELETE CASCADE,
  code TEXT NOT NULL,
  opened_at INTEGER NOT NULL,
  resolved_at INTEGER
);
CREATE UNIQUE INDEX alerts_open ON alerts(machine_id, code) WHERE resolved_at IS NULL;
CREATE INDEX alerts_time ON alerts(opened_at);
CREATE INDEX alerts_machine_time ON alerts(machine_id, opened_at);

-- Prepared for retained aggregates. The initial API computes five-minute
-- rollups over the seven-day raw window; no long-term retention is implied.
CREATE TABLE rollups_5m (
  machine_id TEXT NOT NULL REFERENCES machines(id) ON DELETE CASCADE,
  bucket_at INTEGER NOT NULL,
  sample_count INTEGER NOT NULL,
  metrics_json TEXT NOT NULL,
  PRIMARY KEY(machine_id, bucket_at)
);
