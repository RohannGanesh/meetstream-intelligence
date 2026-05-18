"""MeetStream Intelligence API — Builds 1–4: Classifier, Forecaster, Recommender, Speaker Intelligence."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.classifiers.predict import classify, classify_batch
from src.classifiers.predict import get_model as get_classifier
from src.classifiers.schemas import (
    BatchClassifyRequest,
    BatchClassifyResponse,
    ClassifyRequest,
    ClassifyResponse,
)
from src.forecaster.predict import forecast, forecast_from_events
from src.forecaster.predict import get_model as get_forecaster
from src.forecaster.schemas import (
    EventsForecastRequest,
    ForecastRequest,
    ForecastResponse,
)
from src.recommender.predict import recommend, recommend_batch
from src.recommender.predict import get_models as get_recommender
from src.recommender.schemas import (
    BatchRecommendRequest,
    BatchRecommendResponse,
    RecommendRequest,
    RecommendResponse,
)
from src.speaker.predict import analyze, analyze_batch
from src.speaker.predict import get_model as get_speaker
from src.speaker.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_classifier()   # warm up classifier on startup
    get_forecaster()   # warm up forecaster on startup
    get_recommender()  # warm up recommender on startup
    get_speaker()      # warm up engagement classifier on startup
    yield


app = FastAPI(
    title="MeetStream Intelligence",
    version="0.1.0",
    description="ML-powered meeting analytics for MeetStream AI",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify_title(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a single meeting title into one of 8 categories."""
    try:
        return classify(request.title)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first")


@app.post("/classify/batch", response_model=BatchClassifyResponse)
def classify_titles_batch(request: BatchClassifyRequest) -> BatchClassifyResponse:
    """Classify up to 100 meeting titles in a single request."""
    try:
        return BatchClassifyResponse(results=classify_batch(request.titles))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first")


# ── Build 2: Bot Demand Forecaster ──────────────────────────────────────────


@app.post("/forecast", response_model=ForecastResponse)
def forecast_demand(request: ForecastRequest) -> ForecastResponse:
    """Predict concurrent bot demand for a future time window using learned patterns."""
    try:
        return forecast(request)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Forecaster not loaded — run forecaster/train.py first")


@app.post("/forecast/from-events", response_model=ForecastResponse)
def forecast_demand_from_events(request: EventsForecastRequest) -> ForecastResponse:
    """Compute exact concurrent bot count per slot from a list of known calendar events."""
    return forecast_from_events(request)


# ── Build 3: Agent Config Recommender ───────────────────────────────────────


@app.post("/recommend", response_model=RecommendResponse)
def recommend_config(request: RecommendRequest) -> RecommendResponse:
    """Recommend optimal MIA agent config for a single meeting."""
    try:
        return recommend(request)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Recommender not loaded — run recommender/train.py first")


@app.post("/recommend/batch", response_model=BatchRecommendResponse)
def recommend_config_batch(request: BatchRecommendRequest) -> BatchRecommendResponse:
    """Recommend agent configs for up to 100 meetings in one request."""
    try:
        return recommend_batch(request)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Recommender not loaded — run recommender/train.py first")


# ── Build 4: Speaker Intelligence ────────────────────────────────────────────


@app.post("/speaker/analyze", response_model=AnalyzeResponse)
def analyze_speaker_timeline(request: AnalyzeRequest) -> AnalyzeResponse:
    """Compute per-speaker stats and classify meeting engagement from a speaker timeline."""
    try:
        return analyze(request)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Engagement model not loaded — run speaker/train.py first")


@app.post("/speaker/analyze/batch", response_model=BatchAnalyzeResponse)
def analyze_speaker_timelines_batch(request: BatchAnalyzeRequest) -> BatchAnalyzeResponse:
    """Analyze speaker timelines for up to 20 meetings in one request."""
    try:
        return analyze_batch(request)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Engagement model not loaded — run speaker/train.py first")
