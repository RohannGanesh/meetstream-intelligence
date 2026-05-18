"""Pydantic schemas for the bot demand forecaster API."""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ForecastRequest(BaseModel):
    start: datetime
    end: datetime
    granularity_minutes: int = Field(default=60, ge=15, le=1440)

    @model_validator(mode="after")
    def end_after_start(self) -> "ForecastRequest":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        total_minutes = (self.end - self.start).total_seconds() / 60
        if total_minutes > 7 * 24 * 60:
            raise ValueError("forecast window cannot exceed 7 days")
        return self


class ForecastSlot(BaseModel):
    slot_start: datetime
    slot_end: datetime
    predicted_bots: float = Field(..., ge=0.0)


class ForecastResponse(BaseModel):
    slots: list[ForecastSlot]
    peak_concurrent: float = Field(..., ge=0.0)


class CalendarEvent(BaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def end_after_start(self) -> "CalendarEvent":
        if self.end <= self.start:
            raise ValueError("event end must be after start")
        return self


class EventsForecastRequest(BaseModel):
    events: list[CalendarEvent] = Field(..., min_length=1, max_length=1000)
    granularity_minutes: int = Field(default=60, ge=15, le=1440)
