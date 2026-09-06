"""Incremental journey snapshots shared by the map and archive workers."""
from __future__ import annotations


def _snapshot_delta(previous: dict | None, snapshot: dict) -> dict:
    """Only the last sampled point can change: snapshots add an unsampled live tail."""
    reset = previous is None or previous.get("id") != snapshot.get("id")
    prior = {} if reset else previous
    old_segments = {segment["id"]: segment for segment in prior.get("segments", [])}
    segments = []
    for segment in snapshot.get("segments", []):
        old = old_segments.get(segment["id"])
        if old is not None and old.get("ended") is not None:
            continue
        offset = max(0, len(old["points"])-1) if old is not None else 0
        segments.append({"segment": {key: value for key, value in segment.items() if key != "points"},
                         "offset": offset, "points": segment["points"][offset:]})
    return {"reset": reset,
            "metadata": {key: value for key, value in snapshot.items()
                         if key not in ("segments", "images", "prompts")},
            "segments": segments, "images": snapshot.get("images", [])[len(prior.get("images", [])):],
            "prompts": snapshot.get("prompts", [])}


def _merge_snapshot(snapshot: dict | None, delta: dict) -> dict:
    """Apply every queued delta, including while the map is black, in FIFO order."""
    if snapshot is None or delta["reset"]:
        snapshot = {"segments": [], "images": []}
    snapshot.update(delta["metadata"])
    snapshot["prompts"] = delta["prompts"]
    segments = {segment["id"]: segment for segment in snapshot["segments"]}
    for patch in delta["segments"]:
        segment_id = patch["segment"]["id"]
        segment = segments.get(segment_id)
        if segment is None:
            segment = {"points": []}
            snapshot["segments"].append(segment)
            segments[segment_id] = segment
        segment.update(patch["segment"])
        segment["points"][patch["offset"]:] = patch["points"]
    snapshot["images"].extend(delta["images"])
    return snapshot
