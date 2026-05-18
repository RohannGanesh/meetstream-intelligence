"""Pydantic schemas for the speaker intelligence API."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

EngagementLabel = Literal["balanced", "presenter_led", "dominated"]


class SpeakerSegment(BaseModel):
    speaker: str = Field(..., min_length=1, max_length=100)
    start: float = Field(..., ge=0.0)
    end: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def end_after_start(self) -> "SpeakerSegment":
        if self.end <= self.start:
            raise ValueError("segment end must be after start")
        return self


class AnalyzeRequest(BaseModel):
    meeting_id: str = Field(..., min_length=1, max_length=100)
    duration_seconds: float = Field(..., gt=0.0, le=86400.0)
    segments: list[SpeakerSegment] = Field(..., min_length=1, max_length=10000)


class SpeakerStats(BaseModel):
    speaker: str
    talk_time_seconds: float = Field(..., ge=0.0)
    talk_ratio: float = Field(..., ge=0.0, le=1.0)
    turn_count: int = Field(..., ge=1)
    longest_monologue_seconds: float = Field(..., ge=0.0)
    avg_turn_duration_seconds: float = Field(..., ge=0.0)


class AnalyzeResponse(BaseModel):
    meeting_id: str
    duration_seconds: float
    speaker_count: int
    speaker_stats: list[SpeakerStats]
    silence_ratio: float = Field(..., ge=0.0, le=1.0)
    interruption_count: int = Field(..., ge=0)
    dominance_index: float = Field(..., ge=0.0, le=1.0)
    engagement: EngagementLabel
    engagement_confidence: float = Field(..., ge=0.0, le=1.0)


class BatchAnalyzeRequest(BaseModel):
    meetings: list[AnalyzeRequest] = Field(..., min_length=1, max_length=20)


class BatchAnalyzeResponse(BaseModel):
    results: list[AnalyzeResponse]
