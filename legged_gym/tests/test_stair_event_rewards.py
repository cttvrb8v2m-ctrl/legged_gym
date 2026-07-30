"""Unit tests for the planned event-based stair reward accounting."""

import unittest


class StairEventTracker:
    def __init__(self, dt, rear_stable_steps, stagnation_steps):
        self.dt = dt
        self.rear_stable_steps = rear_stable_steps
        self.stagnation_steps = stagnation_steps
        self.reset()

    def reset(self):
        self.rear_rewarded_level = [0, 0]
        self.rear_level = [0, 0]
        self.rear_stable_count = [0, 0]
        self.summit_rewarded = False
        self.stagnation_count = 0
        self.stagnation_rewarded = False

    def rear_update(self, foot, level, stable):
        if not stable or level != self.rear_level[foot]:
            self.rear_stable_count[foot] = 0
            self.rear_level[foot] = level
        if stable:
            self.rear_stable_count[foot] += 1
        event = (
            stable
            and level > self.rear_rewarded_level[foot]
            and self.rear_stable_count[foot] == self.rear_stable_steps
        )
        if event:
            self.rear_rewarded_level[foot] = level
        return float(event) / self.dt

    def summit_update(self, passed_steps, step_count):
        event = passed_steps >= step_count and not self.summit_rewarded
        if event:
            self.summit_rewarded = True
        return float(event) / self.dt

    def stagnation_update(self, made_progress):
        if made_progress:
            self.stagnation_count = 0
            self.stagnation_rewarded = False
            return 0.0
        self.stagnation_count += 1
        event = (
            self.stagnation_count == self.stagnation_steps
            and not self.stagnation_rewarded
        )
        if event:
            self.stagnation_rewarded = True
        return float(event) / self.dt


class RearFollowTracker:
    def __init__(self, dt, timeout_steps):
        self.dt = dt
        self.timeout_steps = timeout_steps
        self.target_level = 0
        self.active = False
        self.elapsed_steps = 0
        self.clearance_rewarded = False

    def update(
        self,
        front_level,
        rear_level,
        in_swing,
        crossed_riser,
        above_target,
        stable_landing,
    ):
        if (
            front_level > rear_level
            and front_level > self.target_level
        ):
            self.target_level = front_level
            self.active = True
            self.elapsed_steps = 0
            self.clearance_rewarded = False

        clearance = (
            self.active
            and not self.clearance_rewarded
            and in_swing
            and crossed_riser
            and above_target
        )
        if clearance:
            self.clearance_rewarded = True

        if self.active:
            self.elapsed_steps += 1
        completed = (
            self.active
            and stable_landing
            and rear_level >= self.target_level
        )
        timeout = (
            self.active
            and not completed
            and self.elapsed_steps >= self.timeout_steps
        )
        self.active = self.active and not (completed or timeout)
        return float(clearance) / self.dt, float(timeout) / self.dt


def dense_rear_follow_progress(
    forward_delta,
    up_delta,
    front_level,
    rear_level,
    swing,
    tread_depth=0.30,
    step_height=0.155,
    min_forward_delta=0.001,
    clip_per_step=0.08,
):
    enabled = (
        front_level > rear_level
        and swing
        and forward_delta > min_forward_delta
    )
    if not enabled:
        return 0.0, 0.0
    forward = min(
        max(forward_delta / tread_depth, 0.0), clip_per_step
    )
    upward = min(
        max(up_delta / step_height, 0.0), clip_per_step
    )
    return forward, upward


class RearTargetPotential:
    """Minimal scalar model of the best-so-far target reward."""

    def __init__(self, initial_distance, clip_per_step=0.08):
        self.best_distance = initial_distance
        self.clip_per_step = clip_per_step

    def update(self, distance, active=True):
        if not active:
            return 0.0
        progress = min(
            max(self.best_distance - distance, 0.0),
            self.clip_per_step,
        )
        self.best_distance = min(self.best_distance, distance)
        return progress


class TwoStageRearTarget:
    def __init__(self, clearance_distance, landing_distance):
        self.phase = "clearance"
        self.clearance = RearTargetPotential(clearance_distance)
        self.landing_distance = landing_distance
        self.landing = None

    def update(self, distance, cleared=False, in_swing=True):
        if cleared and self.phase == "clearance":
            self.phase = "landing"
            self.landing = RearTargetPotential(self.landing_distance)
            return 0.0
        tracker = (
            self.clearance if self.phase == "clearance"
            else self.landing
        )
        return tracker.update(distance, active=in_swing)


class StairEventRewardTests(unittest.TestCase):
    def setUp(self):
        self.dt = 0.02
        self.tracker = StairEventTracker(
            dt=self.dt,
            rear_stable_steps=3,
            stagnation_steps=100,
        )

    def weighted(self, raw, coefficient):
        prepared_scale = coefficient * self.dt
        return raw * prepared_scale

    def test_event_dt_accounting_equals_coefficient(self):
        raw = 1.0 / self.dt
        self.assertAlmostEqual(self.weighted(raw, 0.50), 0.50)
        self.assertAlmostEqual(self.weighted(raw, -0.05), -0.05)

    def test_same_rear_foot_level_cannot_repeat(self):
        outputs = [
            self.tracker.rear_update(0, 1, True)
            for _ in range(3)
        ]
        self.assertEqual(sum(value > 0 for value in outputs), 1)
        self.tracker.rear_update(0, 1, False)
        repeated = [
            self.tracker.rear_update(0, 1, True)
            for _ in range(3)
        ]
        self.assertEqual(sum(value > 0 for value in repeated), 0)
        upgraded = [
            self.tracker.rear_update(0, 2, True)
            for _ in range(3)
        ]
        self.assertEqual(sum(value > 0 for value in upgraded), 1)

    def test_summit_only_once_per_episode(self):
        values = [
            self.tracker.summit_update(level, 9)
            for level in (8, 9, 9, 9)
        ]
        self.assertEqual(sum(value > 0 for value in values), 1)
        self.tracker.reset()
        self.assertGreater(self.tracker.summit_update(9, 9), 0)

    def test_stagnation_is_one_shot_until_progress(self):
        values = [
            self.tracker.stagnation_update(False)
            for _ in range(150)
        ]
        self.assertEqual(sum(value > 0 for value in values), 1)
        self.tracker.stagnation_update(True)
        values = [
            self.tracker.stagnation_update(False)
            for _ in range(100)
        ]
        self.assertEqual(sum(value > 0 for value in values), 1)

    def test_rear_follow_clearance_is_once_per_front_target(self):
        tracker = RearFollowTracker(self.dt, timeout_steps=40)
        outputs = [
            tracker.update(1, 0, True, True, True, False)[0]
            for _ in range(5)
        ]
        self.assertEqual(sum(value > 0 for value in outputs), 1)
        tracker.update(1, 1, False, True, True, True)
        next_target = tracker.update(2, 1, True, True, True, False)
        self.assertGreater(next_target[0], 0)

    def test_rear_follow_timeout_is_one_shot_per_front_target(self):
        tracker = RearFollowTracker(self.dt, timeout_steps=4)
        outputs = [
            tracker.update(1, 0, False, False, False, False)[1]
            for _ in range(10)
        ]
        self.assertEqual(sum(value > 0 for value in outputs), 1)
        repeated = tracker.update(
            1, 0, False, False, False, False
        )[1]
        self.assertEqual(repeated, 0.0)
        new_target = tracker.update(
            2, 0, False, False, False, False
        )
        self.assertEqual(new_target[1], 0.0)

    def test_dense_follow_requires_front_lead_swing_and_forward_motion(self):
        self.assertEqual(
            dense_rear_follow_progress(0.01, 0.02, 1, 1, True),
            (0.0, 0.0),
        )
        self.assertEqual(
            dense_rear_follow_progress(0.0, 0.02, 2, 1, True),
            (0.0, 0.0),
        )
        self.assertEqual(
            dense_rear_follow_progress(0.01, 0.02, 2, 1, False),
            (0.0, 0.0),
        )

    def test_dense_follow_progress_is_clipped_per_step(self):
        forward, upward = dense_rear_follow_progress(
            1.0, 1.0, 2, 1, True
        )
        self.assertAlmostEqual(forward, 0.08)
        self.assertAlmostEqual(upward, 0.08)

    def test_target_potential_cannot_be_farmed_by_backtracking(self):
        tracker = RearTargetPotential(1.0)
        self.assertAlmostEqual(tracker.update(0.95), 0.05)
        self.assertEqual(tracker.update(1.20), 0.0)
        self.assertEqual(tracker.update(0.95), 0.0)
        self.assertAlmostEqual(tracker.update(0.90), 0.05)

    def test_target_potential_is_clipped_and_swing_gated(self):
        tracker = RearTargetPotential(1.0)
        self.assertEqual(tracker.update(0.0, active=False), 0.0)
        self.assertAlmostEqual(tracker.update(0.0), 0.08)

    def test_two_stage_target_resets_at_clearance_transition(self):
        tracker = TwoStageRearTarget(1.0, landing_distance=0.7)
        self.assertAlmostEqual(tracker.update(0.95), 0.05)
        self.assertEqual(tracker.update(0.7, cleared=True), 0.0)
        self.assertEqual(tracker.phase, "landing")
        self.assertAlmostEqual(tracker.update(0.65), 0.05)

    def test_two_stage_target_stops_reward_after_contact(self):
        tracker = TwoStageRearTarget(1.0, landing_distance=0.7)
        tracker.update(0.7, cleared=True)
        self.assertEqual(tracker.update(0.5, in_swing=False), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
