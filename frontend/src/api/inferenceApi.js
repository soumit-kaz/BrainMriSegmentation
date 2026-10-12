import { deleteResource, getJson, postForm } from "./httpClient.js";

const SEGMENTATIONS_API_URL = "/api/segmentations";

export const getModelInformation = () => getJson("/api/model");

export const getQueueStatus = () => getJson("/api/queue");

export const getSegmentationJobs = () => getJson(SEGMENTATIONS_API_URL);

export const addSegmentationJob = (formData) => postForm(SEGMENTATIONS_API_URL, formData);

export const deleteSegmentationJob = (jobId) => deleteResource(`${SEGMENTATIONS_API_URL}/${jobId}`);

export const getJobEventsUrl = (jobId) => `${SEGMENTATIONS_API_URL}/${jobId}/events`;

export const getSliceImageUrl = (jobId, sliceIndex, modality) =>
  `${SEGMENTATIONS_API_URL}/${jobId}/slices/${sliceIndex}/image?modality=${modality}`;

export const getSliceMaskUrl = (jobId, sliceIndex, source) =>
  `${SEGMENTATIONS_API_URL}/${jobId}/slices/${sliceIndex}/mask?source=${source}`;

export const getMaskDownloadUrl = (jobId) => `${SEGMENTATIONS_API_URL}/${jobId}/mask`;
