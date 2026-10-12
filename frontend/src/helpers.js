export const MAXIMUM_CHART_POINT_COUNT = 800;
export const MOVING_AVERAGE_WINDOW = 25;
export const ACTIVE_TRAINING_STATES = ["starting", "running", "stopping"];
export const TERMINAL_JOB_STATUSES = ["completed", "failed"];

export const MODALITIES = [
  { key: "t1n", label: "T1 native" },
  { key: "t1c", label: "T1 post-contrast" },
  { key: "t2w", label: "T2" },
  { key: "t2f", label: "T2-FLAIR" },
];

export const UPLOAD_FIELDS = [...MODALITIES, { key: "seg", label: "Ground truth (optional)" }];

export const CHART_COLORS = {
  training: "#4f8cff",
  validation: "#ff8a3d",
  iou: "#3ecf8e",
  average: "#f5c542",
  neutral: "#a78bfa",
};

export const PUBLISHED_MODEL_TEST_RESULTS = {
  dice: { mean: 0.9002, median: 0.9264, standard_deviation: 0.0813, percentile_25: 0.8878, percentile_75: 0.9512 },
  iou: { mean: 0.8273, median: 0.8628, standard_deviation: 0.1174, percentile_25: 0.7982, percentile_75: 0.907 },
};

export function getDownsampledPoints(points, maximumCount = MAXIMUM_CHART_POINT_COUNT) {
  if (points.length <= maximumCount) return points;
  const bucketSize = points.length / maximumCount;
  const downsampledPoints = [];
  for (let bucketIndex = 0; bucketIndex < maximumCount; bucketIndex += 1) {
    const bucket = points.slice(Math.floor(bucketIndex * bucketSize), Math.floor((bucketIndex + 1) * bucketSize));
    if (bucket.length === 0) continue;
    downsampledPoints.push({
      x: bucket.reduce((total, point) => total + point.x, 0) / bucket.length,
      y: bucket.reduce((total, point) => total + point.y, 0) / bucket.length,
    });
  }
  return downsampledPoints;
}

export function getMovingAveragePoints(points, windowSize = MOVING_AVERAGE_WINDOW) {
  let runningTotal = 0;
  return points.map((point, index) => {
    runningTotal += point.y;
    if (index >= windowSize) runningTotal -= points[index - windowSize].y;
    return { x: point.x, y: runningTotal / Math.min(index + 1, windowSize) };
  });
}

export function getTickValues(minimumValue, maximumValue, tickCount = 5) {
  if (minimumValue === maximumValue) return [minimumValue];
  const rawStep = (maximumValue - minimumValue) / tickCount;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const niceStep = [1, 2, 2.5, 5, 10].map((factor) => factor * magnitude).find((step) => step >= rawStep) ?? rawStep;
  const tickValues = [];
  for (let tickValue = Math.ceil(minimumValue / niceStep) * niceStep; tickValue <= maximumValue + niceStep * 1e-6; tickValue += niceStep) {
    tickValues.push(Number(tickValue.toPrecision(12)));
  }
  return tickValues;
}

export function formatTickValue(value) {
  if (value === 0) return "0";
  const absoluteValue = Math.abs(value);
  if (absoluteValue < 0.001 || absoluteValue >= 100000) return value.toExponential(1);
  if (Number.isInteger(value)) return String(value);
  return String(Number(value.toPrecision(3)));
}

export function getBestEpoch(epochRecords) {
  return epochRecords.reduce((best, record) => (best === null || record.validation_dice > best.validation_dice ? record : best), null);
}

export function getEstimatedRemainingSeconds(trainingState) {
  const { epochs, progress, run } = trainingState;
  if (!run || !progress || epochs.length === 0 || progress.phase === "test" || progress.phase === "caching") return null;
  const recentEpochs = epochs.slice(-3);
  const averageEpochSeconds = recentEpochs.reduce((total, record) => total + record.duration_seconds, 0) / recentEpochs.length;
  const stepFraction = progress.step / progress.step_count;
  const epochFraction = progress.phase === "training" ? stepFraction * 0.8 : 0.8 + stepFraction * 0.2;
  return (run.epoch_count - progress.epoch + 1 - epochFraction) * averageEpochSeconds;
}

export function getGlobalBatchStep(batchPoint) {
  return (batchPoint.epoch - 1) * batchPoint.step_count + batchPoint.step;
}

export function getUploadFieldKey(fileName) {
  const match = fileName.toLowerCase().match(/[-_](t1n|t1c|t2w|t2f|seg)\.nii(\.gz)?$/);
  return match ? match[1] : null;
}

export function isTerminalJob(job) {
  return TERMINAL_JOB_STATUSES.includes(job?.status);
}
