import { getJson, postJson } from "./httpClient.js";

const TRAINING_API_URL = "/api/training";

export const trainingEventsUrl = `${TRAINING_API_URL}/events`;

export const getDatasetInformation = () => getJson(`${TRAINING_API_URL}/dataset`);

export const startTraining = (trainingOptions) => postJson(`${TRAINING_API_URL}/start`, { command: "train", ...trainingOptions });

export const resumeTraining = (epochCount) => postJson(`${TRAINING_API_URL}/start`, { command: "train", is_resume: true, epoch_count: epochCount });

export const evaluateTrainedModel = () => postJson(`${TRAINING_API_URL}/start`, { command: "evaluate" });

export const stopTraining = () => postJson(`${TRAINING_API_URL}/stop`);
