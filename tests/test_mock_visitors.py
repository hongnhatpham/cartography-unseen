from tools.mock_visitors import build_plan, action_at


def test_random_visitors_cover_two_hours_without_gaps_or_operator_keys():
    for seed in range(100):
        plan = build_plan(seed)
        cursor = 60
        for session in plan['sessions']:
            assert session['start'] == cursor
            assert 120 <= session['duration'] <= 900
            cursor = session['end']
        assert cursor == 7260
        cursor = 60
        for action in plan['actions']:
            assert action['start'] == cursor
            assert set(action['keys']) <= {'w', 'a', 's', 'd'}
            assert action['keys']
            assert action_at(plan, action['start']) == action
            cursor = action['end']
        assert cursor == 7260
        assert action_at(plan, 59) is None
        assert action_at(plan, 7260) is None
        moments = [save['at'] for save in plan['saves']]
        assert len(moments) >= 20
        assert moments == sorted(set(moments))
        assert min(b-a for a, b in zip([60] + moments, moments)) >= 60
        assert moments[-1] == 7260
        assert {a['action'] for a in plan['actions']} == {
            'explore', 'turn_left', 'turn_right', 'strafe_left', 'strafe_right', 'backtrack'}


def test_plan_can_be_replayed_from_saved_seed():
    assert build_plan(123) == build_plan(123)
    assert build_plan(123)['sessions'] != build_plan(456)['sessions']
