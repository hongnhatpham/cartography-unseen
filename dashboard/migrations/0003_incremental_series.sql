-- The dashboard follows receipt time, so delayed/offline history also updates
-- existing chart buckets without scanning the machine's entire history.
CREATE INDEX samples_machine_received ON samples(machine_id, received_at, bucket_at);
