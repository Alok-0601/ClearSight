import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routes import router


load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="ClearSight Verification API",
    version="1.0.0",
    description="ML fake-news classification with optional evidence retrieval.",
)

# These are the public frontends shipped with this API.  Keep them in the
# application configuration so a stale or missing Render environment variable
# cannot take the live verification desk offline.
DEFAULT_CORS_ORIGINS = {
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "https://alok-0601.github.io",
    "https://clearsightt.streamlit.app",
}
configured_origins = {
    item.strip().rstrip("/")
    for item in os.getenv("CORS_ORIGINS", "").split(",")
    if item.strip()
}
origins = sorted(DEFAULT_CORS_ORIGINS | configured_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": jsonable_encoder(exc.errors())})


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": "request_error", "detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logging.getLogger(__name__).exception("Unhandled API error")
    return JSONResponse(status_code=500, content={"error": "internal_server_error", "detail": "An unexpected error occurred."})


@app.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "clearsight-api"}


app.include_router(router)
