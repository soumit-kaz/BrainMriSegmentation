from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, File, Query, Response, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse

from inference_service.domain.segmentation_job import MaskSource, ScanSubmission, UploadedScan
from inference_service.presentation.dependencies import ClientKeyDependency, SegmentationJobServiceDependency
from inference_service.presentation.schemas import SegmentationJobResponse
from inference_service.utils import get_server_sent_event

segmentation_router = APIRouter(prefix="/api/segmentations", tags=["segmentations"])

PNG_CACHE_HEADERS = {"Cache-Control": "private, max-age=86400, immutable"}
SERVER_SENT_EVENT_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
MODALITY_PATTERN = "^(t1n|t1c|t2w|t2f)$"


def get_uploaded_scan(upload_file: UploadFile) -> UploadedScan:
    return UploadedScan(file_name=upload_file.filename or "", content_stream=upload_file.file)


@segmentation_router.post("", status_code=status.HTTP_202_ACCEPTED)
def add_segmentation_job(
    response: Response,
    segmentation_job_service: SegmentationJobServiceDependency,
    client_key: ClientKeyDependency,
    t1n: Annotated[UploadFile, File(description="T1 native")],
    t1c: Annotated[UploadFile, File(description="T1 post-contrast")],
    t2w: Annotated[UploadFile, File(description="T2 weighted")],
    t2f: Annotated[UploadFile, File(description="T2-FLAIR")],
    seg: Annotated[UploadFile | None, File(description="Optional ground truth")] = None,
) -> SegmentationJobResponse:
    submission = ScanSubmission(
        modality_scans={"t1n": get_uploaded_scan(t1n), "t1c": get_uploaded_scan(t1c),
                        "t2w": get_uploaded_scan(t2w), "t2f": get_uploaded_scan(t2f)},
        ground_truth_scan=get_uploaded_scan(seg) if seg is not None and seg.filename else None,
    )
    submission_result = segmentation_job_service.submit_job(submission, client_key)
    response.headers["Location"] = f"{segmentation_router.prefix}/{submission_result.job.job_id}"
    return SegmentationJobResponse.from_domain(submission_result.job, submission_result.queue_position, submission_result.is_reused)


@segmentation_router.get("")
def get_segmentation_jobs(segmentation_job_service: SegmentationJobServiceDependency) -> list[SegmentationJobResponse]:
    return [SegmentationJobResponse.from_domain(job) for job in segmentation_job_service.get_jobs()]


@segmentation_router.get("/{job_id}")
def get_segmentation_job(job_id: str, segmentation_job_service: SegmentationJobServiceDependency) -> SegmentationJobResponse:
    job = segmentation_job_service.get_job(job_id)
    return SegmentationJobResponse.from_domain(job, segmentation_job_service.get_queue_position(job))


@segmentation_router.get("/{job_id}/events")
async def get_segmentation_job_events(job_id: str, segmentation_job_service: SegmentationJobServiceDependency) -> StreamingResponse:
    job_updates = segmentation_job_service.get_job_updates(job_id)
    first_job, first_queue_position = await anext(job_updates)

    async def get_event_stream() -> AsyncGenerator[str, None]:
        try:
            yield get_server_sent_event(SegmentationJobResponse.from_domain(first_job, first_queue_position).model_dump(mode="json"))
            async for job, queue_position in job_updates:
                yield get_server_sent_event(SegmentationJobResponse.from_domain(job, queue_position).model_dump(mode="json"))
        finally:
            await job_updates.aclose()

    return StreamingResponse(get_event_stream(), media_type="text/event-stream", headers=SERVER_SENT_EVENT_HEADERS)


@segmentation_router.get("/{job_id}/slices/{slice_index}/image")
def get_slice_image(job_id: str, slice_index: int, segmentation_job_service: SegmentationJobServiceDependency,
                    modality: Annotated[str, Query(pattern=MODALITY_PATTERN)] = "t2f") -> Response:
    png_content = segmentation_job_service.get_slice_image_png(job_id, modality, slice_index)
    return Response(png_content, media_type="image/png", headers=PNG_CACHE_HEADERS)


@segmentation_router.get("/{job_id}/slices/{slice_index}/mask")
def get_slice_mask(job_id: str, slice_index: int, segmentation_job_service: SegmentationJobServiceDependency,
                   source: MaskSource = MaskSource.PREDICTION) -> Response:
    png_content = segmentation_job_service.get_slice_mask_png(job_id, source, slice_index)
    return Response(png_content, media_type="image/png", headers=PNG_CACHE_HEADERS)


@segmentation_router.get("/{job_id}/mask")
def get_segmentation_mask(job_id: str, segmentation_job_service: SegmentationJobServiceDependency) -> FileResponse:
    job, mask_path = segmentation_job_service.get_mask_path(job_id)
    return FileResponse(mask_path, media_type="application/gzip", filename=f"{job.case_name}-predicted-seg.nii.gz")


@segmentation_router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_segmentation_job(job_id: str, segmentation_job_service: SegmentationJobServiceDependency) -> Response:
    segmentation_job_service.delete_job(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
