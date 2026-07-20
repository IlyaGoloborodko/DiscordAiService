"""Tests for the recommendation system (freshness + varied picking).

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
from app.recommendations.freshness import Play, soft_multiplier, staleness, within_floor

_FRESHNESS_ENV = {
    "FRESHNESS_HALFLIFE_HOURS": "24",
    "FRESHNESS_HALFLIFE_UNHEARD_HOURS": "2",
    "FRESHNESS_WEIGHT": "1.0",
    "FRESHNESS_ABSOLUTE_MIN_MINUTES": "45",
}


def _hours_ago(hours: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _heard(hours_ago: float) -> Play:
    return Play(played_at=_hours_ago(hours_ago), heard=True)


def _queued(hours_ago: float) -> Play:
    return Play(played_at=_hours_ago(hours_ago), heard=False)


class StalenessTests(unittest.TestCase):
    """Staleness = the sum of every play's fading score. Higher = held back more."""

    def test_no_history_is_not_stale(self):
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            self.assertEqual(staleness([]), 0.0)

    def test_a_fresh_play_scores_about_one(self):
        # A play right now weighs ~1 (0.5^0); a play one half-life old weighs ~0.5.
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            self.assertAlmostEqual(staleness([_heard(0)]), 1.0, places=2)
            self.assertAlmostEqual(staleness([_heard(24)]), 0.5, places=2)

    def test_it_fades_to_nothing(self):
        # A play from long ago weighs almost nothing: the track drifts back to base.
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            self.assertLess(staleness([_heard(24 * 8)]), 0.01)

    def test_more_plays_pile_up(self):
        # Several recent plays add up — "played a lot lately" -> pushed down harder.
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            one = staleness([_heard(1)])
            three = staleness([_heard(1), _heard(2), _heard(3)])
            self.assertGreater(three, one * 2)

    def test_unheard_fades_faster_than_heard(self):
        # At the same age, a merely-queued play weighs less than a listened-to one.
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            self.assertLess(staleness([_queued(3)]), staleness([_heard(3)]))

    def test_naive_timestamp_is_treated_as_utc(self):
        # Postgres can hand back a timestamp without a timezone; it must not crash.
        naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            self.assertGreater(staleness([Play(played_at=naive, heard=True)]), 0.9)


class SoftMultiplierTests(unittest.TestCase):
    """The factor a track's pick weight is multiplied by: 1 = untouched, less = held back."""

    def test_untouched_track_keeps_full_weight(self):
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            self.assertEqual(soft_multiplier([]), 1.0)

    def test_recent_play_lowers_the_weight_but_never_to_zero(self):
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            factor = soft_multiplier([_heard(1)])
            self.assertLess(factor, 1.0)
            self.assertGreater(factor, 0.0)

    def test_weight_zero_means_ignore_history(self):
        with mock.patch.dict(os.environ, {**_FRESHNESS_ENV, "FRESHNESS_WEIGHT": "0"}):
            self.assertEqual(soft_multiplier([_heard(0)]), 1.0)


class FloorTests(unittest.TestCase):
    """The hard floor: a just-played track is held out completely for a few minutes."""

    def test_played_moments_ago_is_floored(self):
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            self.assertTrue(within_floor([Play(_hours_ago(0.5), heard=True)]))  # 30min < 45min

    def test_past_the_floor_is_free(self):
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            self.assertFalse(within_floor([_heard(1)]))  # 60min > 45min

    def test_a_queued_track_is_floored_too(self):
        # It may be sitting in the queue right now; do not offer it a second time.
        with mock.patch.dict(os.environ, _FRESHNESS_ENV):
            self.assertTrue(within_floor([Play(_hours_ago(0.1), heard=False)]))


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


class FreshnessWeightingTests(unittest.TestCase):
    """`soft` holds recently-played tracks back; `floored` holds them out entirely."""

    @staticmethod
    def _tracks(count: int) -> list[Track]:
        return [Track(id=str(i), title=f"Song {i}") for i in range(count)]

    def test_a_held_back_track_is_picked_less(self):
        # Track "0" would normally dominate (top rank); a low soft factor should
        # make it show up far less often than without it.
        rng = random.Random(1)
        tracks = self._tracks(10)
        held = sum(
            1 for _ in range(200)
            if "0" in {t.id for t in sampling.pick_varied(tracks, 3, rng=rng, soft={"0": 0.01})}
        )
        rng = random.Random(1)
        free = sum(
            1 for _ in range(200)
            if "0" in {t.id for t in sampling.pick_varied(tracks, 3, rng=rng)}
        )
        self.assertLess(held, free)

    def test_a_floored_track_is_never_picked(self):
        # ...as long as there are other candidates to fill the request.
        tracks = self._tracks(10)
        for seed in range(30):
            picked = sampling.pick_varied(tracks, 3, rng=random.Random(seed), floored={"0", "1"})
            self.assertNotIn("0", {t.id for t in picked})
            self.assertNotIn("1", {t.id for t in picked})

    def test_floor_relaxes_when_everything_is_floored(self):
        # Better a repeat than handing back nothing: if every candidate is floored,
        # the floor is ignored and we still return a full set.
        tracks = self._tracks(4)
        picked = sampling.pick_varied(
            tracks, 3, rng=random.Random(0), floored={"0", "1", "2", "3"}
        )
        self.assertEqual(len(picked), 3)


class ArtistSeparationTests(unittest.TestCase):
    """One artist should not fill the whole selection when others are available."""

    @staticmethod
    def _by(artists: list[str]) -> list[Track]:
        return [Track(id=str(i), title=f"Song {i}", uploader=a) for i, a in enumerate(artists)]

    def test_spreads_across_artists(self):
        # 3 tracks by "Olivia" up top, then others. They rank highest, so without
        # separation a 3-track pick is often all-Olivia; turning it on must make
        # that markedly rarer.
        tracks = self._by(["Olivia", "Olivia", "Olivia", "Malcolm", "Ariana", "Bag Raiders"])

        def all_olivia_rate(gamma: str) -> int:
            with mock.patch.dict(
                os.environ, {"ARTIST_SEPARATION_WEIGHT": gamma, "CANDIDATE_RANK_BIAS": "2"}
            ):
                rng = random.Random(1)
                return sum(
                    1 for _ in range(200)
                    if {t.uploader for t in sampling.pick_varied(tracks, 3, rng=rng)} == {"Olivia"}
                )

        self.assertLess(all_olivia_rate("8"), all_olivia_rate("0"))

    def test_single_artist_pool_is_a_no_op(self):
        # "queue by [artist]": the whole pool is one artist, so separation must not
        # break it — the request still gets its tracks.
        tracks = self._by(["Slipknot"] * 8)
        with mock.patch.dict(os.environ, {"ARTIST_SEPARATION_WEIGHT": "5"}):
            picked = sampling.pick_varied(tracks, 5, rng=random.Random(3))
        self.assertEqual(len(picked), 5)


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
        with mock.patch.dict(os.environ, {"FRESHNESS_HALFLIFE_HOURS": "a day"}):
            self.assertEqual(settings.freshness_halflife_hours(), 24)

    def test_missing_value_falls_back(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(settings.freshness_weight(), 1.0)
            self.assertEqual(settings.freshness_floor_minutes(), 45)
            self.assertEqual(settings.artist_separation_weight(), 1.0)
            self.assertEqual(settings.rank_bias(), 1.5)


if __name__ == "__main__":
    unittest.main()
