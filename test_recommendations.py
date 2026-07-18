"""Tests for the recommendation system (cooldown + varied picking).

Run: .venv/Scripts/python.exe -m unittest test_recommendations

Nothing here touches the database, the search service or the LLM.
"""

import os
import random
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from app.data.models import Track
from app.recommendations import sampling, settings
from app.recommendations.cooldown import PlayStat, is_resting, rest_hours, resting_track_ids

_COOLDOWN_ENV = {
    "PLAY_COOLDOWN_BASE_HOURS": "6",
    "PLAY_COOLDOWN_GROWTH": "2.0",
    "PLAY_COOLDOWN_MAX_HOURS": "336",
}


def _hours_ago(hours: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


class RestLengthTests(unittest.TestCase):
    """The more often a track has played, the longer it rests."""

    def test_rest_doubles_with_each_play(self):
        with mock.patch.dict(os.environ, _COOLDOWN_ENV):
            self.assertEqual(rest_hours(1), 6)
            self.assertEqual(rest_hours(2), 12)
            self.assertEqual(rest_hours(3), 24)

    def test_a_track_never_played_does_not_rest(self):
        with mock.patch.dict(os.environ, _COOLDOWN_ENV):
            self.assertEqual(rest_hours(0), 0)

    def test_rest_stops_growing_at_the_cap(self):
        # Without a cap a much-loved track would be banned for years.
        with mock.patch.dict(os.environ, _COOLDOWN_ENV):
            self.assertEqual(rest_hours(50), 336)

    def test_cap_is_configurable(self):
        with mock.patch.dict(os.environ, {**_COOLDOWN_ENV, "PLAY_COOLDOWN_MAX_HOURS": "10"}):
            self.assertEqual(rest_hours(5), 10)


class RestingTests(unittest.TestCase):
    """Whether a specific track is resting right now."""

    def test_just_played_is_resting(self):
        with mock.patch.dict(os.environ, _COOLDOWN_ENV):
            self.assertTrue(is_resting(PlayStat(1, _hours_ago(1))))

    def test_played_long_enough_ago_is_free(self):
        with mock.patch.dict(os.environ, _COOLDOWN_ENV):
            self.assertFalse(is_resting(PlayStat(1, _hours_ago(7))))

    def test_often_played_track_rests_longer(self):
        # 7 hours is enough after one play, but not after three.
        with mock.patch.dict(os.environ, _COOLDOWN_ENV):
            self.assertFalse(is_resting(PlayStat(1, _hours_ago(7))))
            self.assertTrue(is_resting(PlayStat(3, _hours_ago(7))))

    def test_naive_timestamp_is_treated_as_utc(self):
        # Postgres can hand back a timestamp without a timezone; it must not crash.
        naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        with mock.patch.dict(os.environ, _COOLDOWN_ENV):
            self.assertTrue(is_resting(PlayStat(1, naive)))

    def test_picks_out_only_the_resting_ones(self):
        stats = {
            "fresh": PlayStat(1, _hours_ago(1)),   # played an hour ago -> resting
            "old": PlayStat(1, _hours_ago(48)),    # played two days ago -> free
        }
        with mock.patch.dict(os.environ, _COOLDOWN_ENV):
            self.assertEqual(resting_track_ids(stats), {"fresh"})


class PoolSizeTests(unittest.TestCase):
    """We ask the search service for more than we need, to have a choice."""

    def test_asks_for_more_than_wanted(self):
        with mock.patch.dict(os.environ, {"CANDIDATE_POOL_FACTOR": "4"}):
            self.assertEqual(sampling.pool_size(5), 20)

    def test_never_asks_for_less_than_wanted(self):
        with mock.patch.dict(os.environ, {"CANDIDATE_POOL_FACTOR": "0.1"}):
            self.assertEqual(sampling.pool_size(5), 5)


class PickVariedTests(unittest.TestCase):
    """Choosing which of the found tracks to offer."""

    @staticmethod
    def _tracks(count: int) -> list[Track]:
        return [Track(id=str(i), title=f"Song {i}") for i in range(count)]

    def test_returns_everything_when_there_is_nothing_to_choose_from(self):
        tracks = self._tracks(3)
        self.assertEqual(sampling.pick_varied(tracks, 5), tracks)

    def test_returns_the_number_asked_for(self):
        picked = sampling.pick_varied(self._tracks(20), 5, rng=random.Random(1))
        self.assertEqual(len(picked), 5)

    def test_never_picks_the_same_track_twice(self):
        picked = sampling.pick_varied(self._tracks(20), 8, rng=random.Random(2))
        self.assertEqual(len({t.id for t in picked}), 8)

    def test_keeps_the_original_order(self):
        picked = sampling.pick_varied(self._tracks(20), 6, rng=random.Random(3))
        positions = [int(t.id) for t in picked]
        self.assertEqual(positions, sorted(positions))

    def test_two_runs_differ(self):
        # This is the whole point: the same query must stop giving the same songs.
        with mock.patch.dict(os.environ, {"CANDIDATE_RANK_BIAS": "1.5"}):
            first = sampling.pick_varied(self._tracks(30), 5, rng=random.Random(1))
            second = sampling.pick_varied(self._tracks(30), 5, rng=random.Random(2))
        self.assertNotEqual([t.id for t in first], [t.id for t in second])

    def test_high_bias_stays_near_the_top(self):
        # With a strong bias the top results should still dominate, so an explicit
        # request ("play Psychosocial") usually gets its exact match.
        with mock.patch.dict(os.environ, {"CANDIDATE_RANK_BIAS": "6"}):
            rng = random.Random(4)
            hits = sum(
                1 for _ in range(50) if "0" in {t.id for t in sampling.pick_varied(self._tracks(30), 3, rng=rng)}
            )
        self.assertGreater(hits, 40)

    def test_zero_bias_spreads_out(self):
        # With no bias, later results appear about as often as early ones.
        with mock.patch.dict(os.environ, {"CANDIDATE_RANK_BIAS": "0"}):
            rng = random.Random(5)
            seen = set()
            for _ in range(40):
                seen.update(t.id for t in sampling.pick_varied(self._tracks(30), 3, rng=rng))
        self.assertGreater(len(seen), 20)

    def test_empty_input(self):
        self.assertEqual(sampling.pick_varied([], 5), [])
        self.assertEqual(sampling.pick_varied(self._tracks(5), 0), [])


class SettingsTests(unittest.TestCase):
    """Knobs fall back to sane defaults rather than crashing."""

    def test_garbage_value_falls_back(self):
        with mock.patch.dict(os.environ, {"PLAY_COOLDOWN_BASE_HOURS": "six"}):
            self.assertEqual(settings.cooldown_base_hours(), 6)

    def test_missing_value_falls_back(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.cooldown_growth(), 2.0)
            self.assertEqual(settings.rank_bias(), 1.5)


if __name__ == "__main__":
    unittest.main()
