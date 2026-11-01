import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, status

from backend.schemas import Article, HistoryItem, VerificationResponse, VerifyTextRequest, VerifyUrlRequest
from backend.services.article_extractor import ArticleExtractionError, extract_article
from backend.services.evidence_aggregator import retrieve_evidence
from backend.services.ml_service import MLService, ModelUnavailableError
from backend.services.verification_store import VerificationStore


logger = logging.getLogger(__name__)
router = APIRouter(tags=["verification"])
store = VerificationStore()
ml_service = MLService()


async def verify_article(article: Article, *, record: bool = True) -> VerificationResponse:
    """Classify an article and gather evidence.

    ``record=False`` keeps the result out of the public /history store, for
    callers such as the Telegram bot whose input comes from private chats.
    """
    try:
        prediction, confidence = await asyncio.to_thread(ml_service.predict, article.title, article.body)
    except ModelUnavailableError as exc:
        logger.exception("Model unavailable")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    evidence, sources = await retrieve_evidence(article.title, article.body)
    verification = VerificationResponse(
        id=str(uuid4()),
        prediction=prediction,
        confidence=confidence,
        confidence_note="Confidence is derived from the Linear SVM decision margin and is not a calibrated probability.",
        article=article,
        evidence=evidence,
        sources=sources,
        created_at=datetime.now(timezone.utc),
    )
    if record:
        store.add(verification)
    logger.info("verification_created id=%s prediction=%s evidence=%s", verification.id, prediction, len(evidence))
    return verification


async def verify_article_url(url: str, *, record: bool = True) -> VerificationResponse:
    try:
        article = await extract_article(url)
    except ArticleExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return await verify_article(article, record=record)


@router.post("/verify", response_model=VerificationResponse, status_code=status.HTTP_201_CREATED)
async def verify_text(request: VerifyTextRequest) -> VerificationResponse:
    return await verify_article(Article(title=request.title, body=request.text.strip()))


@router.post("/verify/url", response_model=VerificationResponse, status_code=status.HTTP_201_CREATED)
async def verify_url(request: VerifyUrlRequest) -> VerificationResponse:
    return await verify_article_url(str(request.url))


@router.get("/verification/{verification_id}", response_model=VerificationResponse)
async def get_verification(verification_id: str) -> VerificationResponse:
    item = store.get(verification_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Verification not found.")
    return item


@router.get("/history", response_model=list[HistoryItem])
async def get_history(limit: int = Query(default=20, ge=1, le=100)) -> list[HistoryItem]:
    return store.history(limit)
