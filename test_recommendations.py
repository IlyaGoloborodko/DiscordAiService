"""Tests for the recommendation system (cooldown + varied picking).

Run: .venv/Scripts/python.exe -m unittest test_recommendations

Nothing here touches the database, the search service or the LLM.
"""

import asyncio
import os
import random
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

from app.data.models import Track
from app.recommendations import genres, sampling, settings
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

    def test_queued_but_never_heard_gets_only_the_short_rest(self):
        # The listener was handed 5 tracks and heard 2. The other 3 must not be
        # pushed away as if they had been listened to — but they still rest
        # briefly, because they may be sitting in the queue right now.
        never_heard = PlayStat(play_count=0, last_played_at=_hours_ago(1))
        with mock.patch.dict(os.environ, _COOLDOWN_ENV):
            self.assertTrue(is_resting(never_heard))                     # 1h < 6h
            self.assertFalse(is_resting(PlayStat(0, _hours_ago(7))))     # 7h > 6h
            # ...while a track heard three times is still resting at 7 hours.
            self.assertTrue(is_resting(PlayStat(3, _hours_ago(7))))

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


class GenreCacheTests(unittest.IsolatedAsyncioTestCase):
    """Genres are looked up once per artist and remembered, because each lookup
    takes about a second and the answer never changes."""

    def setUp(self):
        self.stored: dict[str, list[dict]] = {}
        self.lookups: list[str] = []

        async def fake_cached(artists):
            return {a: self.stored[a] for a in artists if a in self.stored}

        async def fake_store(artist, tags):
            self.stored[artist] = tags

        for target, replacement in (
            ("cached_tags", fake_cached),
            ("_store", fake_store),
        ):
            patch = mock.patch(f"app.recommendations.genres.{target}", replacement)
            patch.start()
            self.addCleanup(patch.stop)

    def _search(self, answers: dict[str, list[dict]]):
        search = mock.Mock()

        async def tags(artist, limit=10):
            self.lookups.append(artist)
            return answers.get(artist, [])

        search.tags = tags
        return search

    async def test_unknown_artist_is_looked_up_and_cached(self):
        search = self._search({"Slipknot": [{"name": "nu metal", "weight": 100}]})
        await genres.fetch_missing(["Slipknot"], search)
        self.assertEqual(self.lookups, ["Slipknot"])
        self.assertEqual(self.stored["Slipknot"][0]["name"], "nu metal")

    async def test_already_cached_artist_is_not_looked_up_again(self):
        self.stored["Slipknot"] = [{"name": "nu metal", "weight": 100}]
        await genres.fetch_missing(["Slipknot"], self._search({}))
        self.assertEqual(self.lookups, [])

    async def test_artist_nobody_knows_is_cached_as_empty(self):
        # Otherwise we would ask about the same obscure uploader on every play.
        await genres.fetch_missing(["Mid9PHONK"], self._search({}))
        self.assertEqual(self.stored["Mid9PHONK"], [])

    async def test_each_artist_is_looked_up_once_even_if_repeated(self):
        search = self._search({"Korn": [{"name": "nu metal", "weight": 90}]})
        await genres.fetch_missing(["Korn", "Korn", "Korn"], search)
        self.assertEqual(self.lookups, ["Korn"])

    async def test_a_failing_lookup_does_not_lose_the_others(self):
        search = mock.Mock()

        async def tags(artist, limit=10):
            if artist == "Broken":
                raise RuntimeError("boom")
            return [{"name": "metal", "weight": 100}]

        search.tags = tags
        await genres.fetch_missing(["Broken", "Pantera"], search)
        self.assertNotIn("Broken", self.stored)
        self.assertIn("Pantera", self.stored)

    def test_warming_the_cache_does_nothing_without_a_loop(self):
        # Called from scripts and tests where no event loop is running. It must
        # not raise, and must not leave an un-awaited coroutine behind either.
        genres.warm_cache_in_background(["Slipknot"])  # no loop here — must be a no-op

    async def test_warming_the_cache_schedules_the_lookup(self):
        done = asyncio.Event()

        async def fake_fetch(artists):
            self.lookups.extend(artists)
            done.set()

        with mock.patch("app.recommendations.genres.fetch_missing", fake_fetch):
            genres.warm_cache_in_background(["Slipknot"])
            await asyncio.wait_for(done.wait(), timeout=2)
        self.assertEqual(self.lookups, ["Slipknot"])


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
