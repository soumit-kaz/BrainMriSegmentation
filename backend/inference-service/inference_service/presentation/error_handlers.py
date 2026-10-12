from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from inference_service.domain.segmentation_job import (
    InvalidScanError,
    JobConflictError,
    JobNotReadyError,
    QueueFullError,
    QueueUnavailableError,
    RateLimitExceededError,
    RetryLaterError,
    SegmentationNotFoundError,
)

RETRY_AFTER_HEADER = "Retry-After"
STATUS_CODES_BY_ERROR: dict[type[Exception], int] = {
    SegmentationNotFoundError: status.HTTP_404_NOT_FOUND,
    InvalidScanError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    JobNotReadyError: status.HTTP_409_CONFLICT,
    JobConflictError: status.HTTP_409_CONFLICT,
    QueueUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    RateLimitExceededError: status.HTTP_429_TOO_MANY_REQUESTS,
    QueueFullError: status.HTTP_429_TOO_MANY_REQUESTS,
    IndexError: status.HTTP_422_UNPROCESSABLE_ENTITY,
}


def get_error_response(request: Request, error: Exception) -> JSONResponse:
    status_code = STATUS_CODES_BY_ERROR.get(type(error), status.HTTP_500_INTERNAL_SERVER_ERROR)
    detail = str(error) if not isinstance(error, SegmentationNotFoundError) else f"Segmentation job '{error}' was not found."
    headers = {RETRY_AFTER_HEADER: str(error.retry_after_seconds)} if isinstance(error, RetryLaterError) else None
    return JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)


def add_error_handlers(application: FastAPI) -> None:
    for error_type in STATUS_CODES_BY_ERROR:
        application.add_exception_handler(error_type, get_error_response)
