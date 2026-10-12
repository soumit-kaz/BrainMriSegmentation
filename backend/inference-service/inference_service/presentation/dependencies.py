from typing import Annotated

from fastapi import Depends, Request

from inference_service.application.segmentation_job_service import SegmentationJobService

FORWARDED_FOR_HEADER = "x-forwarded-for"
UNKNOWN_CLIENT_KEY = "unknown"


def get_segmentation_job_service(request: Request) -> SegmentationJobService:
    return request.app.state.segmentation_job_service


def get_client_key(request: Request) -> str:
    forwarded_addresses = [address.strip() for address in request.headers.get(FORWARDED_FOR_HEADER, "").split(",") if address.strip()]
    trusted_proxy_hop_count: int = request.app.state.trusted_proxy_hop_count
    if forwarded_addresses and trusted_proxy_hop_count > 0:
        return forwarded_addresses[max(len(forwarded_addresses) - trusted_proxy_hop_count, 0)]
    return request.client.host if request.client else UNKNOWN_CLIENT_KEY


SegmentationJobServiceDependency = Annotated[SegmentationJobService, Depends(get_segmentation_job_service)]
ClientKeyDependency = Annotated[str, Depends(get_client_key)]
