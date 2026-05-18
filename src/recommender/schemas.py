"""Pydantic schemas for the agent config recommender API."""

from typing import Literal

from pydantic import BaseModel, Field

MeetingType = Literal[
    "standup", "planning", "one_on_one", "client",
    "all_hands", "interview", "workshop", "social",
]

RecordingMode = Literal["audio_only", "audio_video"]
SummaryDepth = Literal["brief", "standard", "detailed"]


class RecommendRequest(BaseModel):
    meeting_type: MeetingType
    attendee_count: int = Field(..., ge=1, le=500)
    duration_minutes: int = Field(..., ge=1, le=480)


class AgentConfig(BaseModel):
    summary_enabled: bool
    action_items_enabled: bool
    speaker_diarization: bool
    recording_mode: RecordingMode
    summary_depth: SummaryDepth
    bot_join_offset_seconds: int


class RecommendResponse(BaseModel):
    meeting_type: str
    attendee_count: int
    duration_minutes: int
    config: AgentConfig
    confidence: float = Field(..., ge=0.0, le=1.0)


class BatchRecommendRequest(BaseModel):
    meetings: list[RecommendRequest] = Field(..., min_length=1, max_length=100)


class BatchRecommendResponse(BaseModel):
    results: list[RecommendResponse]
