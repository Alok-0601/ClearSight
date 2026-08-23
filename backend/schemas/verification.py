from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class VerifyTextRequest(BaseModel):
    text: str = Field(min_length=40, max_length=100_000, description="Article body or excerpt")
    title: str | None = Field(default=None, max_length=500)


class VerifyUrlRequest(BaseModel):
    url: HttpUrl


class Article(BaseModel):
    title: str | None = None
    body: str
    publisher: str | None = None
    published_at: datetime | None = None
    url: str | None = None
    extraction_method: str = "provided_text"


class EvidenceItem(BaseModel):
    source: str
    type: Literal["fact_check", "news"]
    title: str
    url: str
    publisher: str | None = None
    published_at: datetime | None = None
    summary: str | None = None
    claim: str | None = None
    rating: str | None = None
    relevance: Literal["high", "medium", "low"] = "low"
    relation: Literal["supports_claim", "contradicts_claim", "related", "not_determined"] = "not_determined"
    relation_reason: str


class SourceStatus(BaseModel):
    source: str
    enabled: bool
    status: Literal["ok", "not_configured", "error"]
    detail: str | None = None


class VerificationResponse(BaseModel):
    id: str
    prediction: Literal["REAL", "FAKE"]
    confidence: float = Field(ge=0, le=1)
    confidence_note: str
    article: Article
    evidence: list[EvidenceItem]
    sources: list[SourceStatus]
    created_at: datetime


class HistoryItem(BaseModel):
    id: str
    prediction: Literal["REAL", "FAKE"]
    confidence: float
    title: str | None = None
    url: str | None = None
    created_at: datetime


class ErrorResponse(BaseModel):
    error: str
    detail: str | list[dict] | None = None
