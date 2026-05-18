"""Pydantic schemas for the meeting title classifier API."""

from pydantic import BaseModel, Field


LABELS = [
    "standup",
    "planning",
    "one_on_one",
    "client",
    "all_hands",
    "interview",
    "workshop",
    "social",
]


class ClassifyRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, examples=["Eng Standup Q2"])


class ClassifyResponse(BaseModel):
    title: str
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class BatchClassifyRequest(BaseModel):
    titles: list[str] = Field(..., min_length=1, max_length=100)


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]
