# MeetStream Intelligence Layer

## Context
This project builds an intelligence layer on top of MeetStream AI's
meeting bot platform (docs.meetstream.ai). It provides ML-powered
features for calendar automation, bot management, and meeting analytics.

## What we're building (4 builds, in order)
1. Meeting Title Classifier - NLP classifier that categorizes meeting
   titles (standup, planning, 1:1, client, all-hands, interview, etc.)
2. Bot Demand Forecaster - time-series model predicting concurrent
   bot demand from calendar data
3. Agent Config Recommender - recommends optimal MIA agent configuration
   based on meeting type, attendees, and duration
4. Speaker Intelligence - post-meeting analytics from speaker timelines

## Tech Stack
Python 3.11+, FastAPI, scikit-learn, pandas, numpy, joblib

## Conventions
- Type hints on all functions
- Docstrings on all public functions
- Pydantic models for all API schemas
- Tests for every endpoint
