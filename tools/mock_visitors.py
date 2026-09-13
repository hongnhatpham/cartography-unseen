"""Repeatable visitor workload, with no operator controls or gaps between players."""
import bisect
import random


def build_plan(seed, start=60, duration=7200):
    rng = random.Random(seed)
    sessions, actions, saves = [], [], []
    cursor, end = start, start + duration
    # Only visitor controls: WASD and mouse. Every action keeps moving.
    choices = [
        ('explore', ('w',), 0), ('turn_left', ('w',), -65),
        ('turn_right', ('w',), 65), ('strafe_left', ('w', 'a'), -15),
        ('strafe_right', ('w', 'd'), 15), ('backtrack', ('s',), 45),
    ]
    while cursor < end:
        remaining = end - cursor
        length = remaining if remaining <= 900 else rng.randint(120, min(900, remaining - 120))
        session_end = cursor + length
        visitor = len(sessions) + 1
        sessions.append(dict(id=visitor, start=cursor, end=session_end, duration=length))
        moment = cursor
        while moment < session_end:
            name, keys, yaw = rng.choice(choices)
            action_end = min(session_end, moment + rng.randint(4, 16))
            actions.append(dict(visitor=visitor, start=moment, end=action_end,
                                action=name, keys=keys, yaw=yaw))
            moment = action_end
        # SPACE changes the prompt and finalizes a real journey. Long visits
        # also try another prompt every 2-4 minutes; avoid tiny final archives.
        moment = cursor
        while session_end - moment > 300:
            moment += rng.randint(120, 240)
            saves.append(dict(at=moment, visitor=visitor, reason='visitor_prompt_change'))
        saves.append(dict(at=session_end, visitor=visitor, reason='visitor_handoff'))
        cursor = session_end
    return dict(seed=seed, start=start, end=end, sessions=sessions, actions=actions,
                action_starts=[a['start'] for a in actions], saves=saves)


def action_at(plan, seconds):
    if not plan['start'] <= seconds < plan['end']:
        return None
    index = bisect.bisect_right(plan['action_starts'], seconds) - 1
    return plan['actions'][index]
