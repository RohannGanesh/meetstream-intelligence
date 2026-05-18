"""Pure-algorithmic speaker timeline analytics (no ML)."""

from collections import defaultdict

import numpy as np

from src.speaker.schemas import AnalyzeRequest, SpeakerStats

# Gap threshold (seconds) under which a turn-switch is counted as an interruption
_INTERRUPTION_GAP = 1.5


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged = [list(sorted(intervals)[0])]
    for start, end in sorted(intervals)[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def _silence_seconds(segments: list, duration: float) -> float:
    intervals = [(max(0.0, s.start), min(duration, s.end)) for s in segments]
    covered = sum(e - s for s, e in _merge_intervals(intervals))
    return max(0.0, duration - covered)


def _count_interruptions(segments: list) -> int:
    sorted_segs = sorted(segments, key=lambda s: s.start)
    count = 0
    for i in range(1, len(sorted_segs)):
        prev, curr = sorted_segs[i - 1], sorted_segs[i]
        if curr.speaker != prev.speaker and (curr.start - prev.end) < _INTERRUPTION_GAP:
            count += 1
    return count


def compute(request: AnalyzeRequest) -> dict:
    """Compute all speaker analytics from a meeting timeline."""
    duration = request.duration_seconds
    segments = request.segments

    by_speaker: dict[str, list] = defaultdict(list)
    for seg in segments:
        by_speaker[seg.speaker].append(seg)

    stats: list[SpeakerStats] = []
    for speaker, segs in by_speaker.items():
        durations = [min(s.end, duration) - max(s.start, 0.0) for s in segs]
        talk_time = sum(durations)
        stats.append(
            SpeakerStats(
                speaker=speaker,
                talk_time_seconds=round(talk_time, 3),
                talk_ratio=round(talk_time / duration, 4),
                turn_count=len(segs),
                longest_monologue_seconds=round(max(durations), 3),
                avg_turn_duration_seconds=round(talk_time / len(segs), 3),
            )
        )

    stats.sort(key=lambda s: s.talk_time_seconds, reverse=True)

    ratios = np.array([s.talk_ratio for s in stats])
    dominance_index = round(float(np.sum(ratios ** 2)), 4)
    max_ratio = float(ratios.max())
    mean_ratio = float(ratios.mean())
    cv = float(ratios.std() / mean_ratio) if mean_ratio > 0 else 0.0

    silence_secs = _silence_seconds(segments, duration)
    silence_ratio = round(silence_secs / duration, 4)
    interruptions = _count_interruptions(segments)

    return {
        "speaker_stats": stats,
        "silence_ratio": silence_ratio,
        "interruption_count": interruptions,
        "dominance_index": dominance_index,
        "speaker_count": len(stats),
        # feature vector passed to the engagement classifier
        "_features": {
            "dominance_index": dominance_index,
            "max_talk_ratio": max_ratio,
            "speaker_count": float(len(stats)),
            "silence_ratio": silence_ratio,
            "cv_talk_ratio": cv,
        },
    }
