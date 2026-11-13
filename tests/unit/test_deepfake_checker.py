"""Tests for the AI-image checker."""

import asyncio

from scambot.core.config import AppConfig
from scambot.core.media_store import ORIGINAL, MediaStore
from scambot.mcp_server.checkers.deepfake import DeepfakeChecker
from scambot.mcp_server.clients.sightengine import SightengineError, SynthScores
from scambot.mcp_server.media import MediaFiles
from tests.fakes import FakeImageAnalyzer
from tests.media import exif_jpeg, png_bytes, store_bytes

KEY = "job1"


async def put(store: MediaStore, name: str, data: bytes) -> None:
    await store_bytes(store, KEY, name, data)


def checker(
    app_config: AppConfig, store: MediaStore, analyzer: FakeImageAnalyzer | None
) -> DeepfakeChecker:
    return DeepfakeChecker(analyzer, MediaFiles(store), app_config.thresholds.deepfake)


async def test_high_score_is_likely_synthetic(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    await put(media_store, ORIGINAL, exif_jpeg())
    analyzer.by_name[ORIGINAL] = SynthScores(0.93, None)

    result = await checker(app_config, media_store, analyzer).check(KEY, [ORIGINAL])

    assert (result.status, result.signal, result.confidence) == ("ok", "likely_synthetic", 0.93)
    assert result.evidence[0].summary.startswith("An AI-image detector rated this image 93%")
    assert "frame" not in result.evidence[0].summary
    assert result.evidence[0].url is None


async def test_low_score_is_not_a_claim_of_authenticity(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    await put(media_store, ORIGINAL, exif_jpeg())

    result = await checker(app_config, media_store, analyzer).check(KEY, [ORIGINAL])

    assert (result.signal, round(result.confidence, 2)) == ("none", 0.98)
    summary = result.evidence[0].summary
    assert summary.startswith("No manipulation detected")
    assert "real" not in summary.lower()
    assert "genuine" not in summary.lower()


async def test_unusable_images_are_inconclusive(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    await put(media_store, "tiny", png_bytes((64, 48)))
    await put(media_store, "flat", png_bytes((1000, 1000)))
    sut = checker(app_config, media_store, analyzer)

    tiny = await sut.check(KEY, ["tiny"])
    flat = await sut.check(KEY, ["flat"])

    assert (tiny.status, tiny.details["reason"]) == ("inconclusive", "low_resolution")
    assert (flat.status, flat.details["reason"]) == ("inconclusive", "over_compressed")
    assert analyzer.calls == []


async def test_highest_frame_is_reported(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    for name in ("kf_00.jpg", "kf_01.jpg", "kf_02.jpg"):
        await put(media_store, name, exif_jpeg())
    analyzer.by_name["kf_01.jpg"] = SynthScores(0.4, 0.88)
    analyzer.by_name["kf_02.jpg"] = SightengineError("status_500")

    result = await checker(app_config, media_store, analyzer).check(
        KEY, ["kf_00.jpg", "kf_01.jpg", "kf_02.jpg"]
    )

    assert (result.signal, result.confidence) == ("likely_synthetic", 0.88)
    assert result.details["top_frame"] == "kf_01.jpg"
    assert "frame 2" in result.evidence[0].summary
    statuses = {f["name"]: f["status"] for f in result.details["frames"]}
    assert statuses == {"kf_00.jpg": "scored", "kf_01.jpg": "scored", "kf_02.jpg": "error"}


async def test_every_frame_failing_is_an_error(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    await put(media_store, ORIGINAL, exif_jpeg())
    analyzer.by_name[ORIGINAL] = SightengineError("status_429")

    result = await checker(app_config, media_store, analyzer).check(KEY, [ORIGINAL])

    assert (result.status, result.error) == ("error", "status_429")


async def test_timeout_is_an_error(app_config: AppConfig, media_store: MediaStore) -> None:
    await put(media_store, ORIGINAL, exif_jpeg())

    class Slow:
        async def scores(self, image: bytes, filename: str) -> SynthScores:
            await asyncio.sleep(1)
            return SynthScores(0.1, 0.1)

    limits = app_config.thresholds.deepfake.model_copy(update={"timeout_s": 0.01})
    sut = DeepfakeChecker(Slow(), MediaFiles(media_store), limits)

    result = await sut.check(KEY, [ORIGINAL])

    assert (result.status, result.error) == ("error", "timeout")


async def test_no_analyzer_no_frames_and_missing_file(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    unconfigured = await checker(app_config, media_store, None).check(KEY, [ORIGINAL])
    sut = checker(app_config, media_store, analyzer)
    empty = await sut.check(KEY, [])
    missing = await sut.check(KEY, [ORIGINAL])

    assert (unconfigured.status, unconfigured.error) == ("error", "not_configured")
    assert (empty.status, empty.details["reason"]) == ("inconclusive", "no_frames")
    assert missing.status == "inconclusive"
    assert missing.details["reason"] == "media_missing"


async def test_undecodable_file(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    await put(media_store, ORIGINAL, b"not an image")

    result = await checker(app_config, media_store, analyzer).check(KEY, [ORIGINAL])

    assert (result.status, result.details["reason"]) == ("inconclusive", "unreadable_image")


async def test_oversize_image_is_re_encoded(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    data = exif_jpeg(size=600)
    await put(media_store, ORIGINAL, data)
    limits = app_config.thresholds.deepfake.model_copy(update={"max_upload_bytes": len(data) - 1})

    result = await DeepfakeChecker(analyzer, MediaFiles(media_store), limits).check(KEY, [ORIGINAL])

    assert result.status == "ok"
    assert analyzer.calls[0][1] != len(data)


async def test_frames_beyond_the_cap_are_ignored(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    names = [f"kf_{i:02d}.jpg" for i in range(8)]
    for name in names:
        await put(media_store, name, exif_jpeg())

    await checker(app_config, media_store, analyzer).check(KEY, names)

    assert len(analyzer.calls) == app_config.thresholds.deepfake.max_frames


async def test_a_response_without_scores_is_not_a_clean_result(
    app_config: AppConfig, media_store: MediaStore, analyzer: FakeImageAnalyzer
) -> None:
    await put(media_store, ORIGINAL, exif_jpeg())
    analyzer.by_name[ORIGINAL] = SynthScores(None, None)

    result = await checker(app_config, media_store, analyzer).check(KEY, [ORIGINAL])

    assert (result.status, result.signal, result.error) == ("error", "none", "no_score")
