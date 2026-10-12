import { useMemo } from "react";
import { LineChart } from "../../components/charts/LineChart.jsx";
import { Panel } from "../../components/common.jsx";
import {
  CHART_COLORS,
  MOVING_AVERAGE_WINDOW,
  getBestEpoch,
  getDownsampledPoints,
  getGlobalBatchStep,
  getMovingAveragePoints,
} from "../../helpers.js";

function getBatchChartData(batchPoints) {
  const getPoints = (fieldName) => batchPoints.map((point) => ({ x: getGlobalBatchStep(point), y: point[fieldName] }));
  const lossPoints = getPoints("loss");
  const dicePoints = getPoints("dice");
  return {
    lossPoints: getDownsampledPoints(lossPoints),
    lossAveragePoints: getDownsampledPoints(getMovingAveragePoints(lossPoints)),
    dicePoints: getDownsampledPoints(dicePoints),
    diceAveragePoints: getDownsampledPoints(getMovingAveragePoints(dicePoints)),
    learningRatePoints: getDownsampledPoints(getPoints("learning_rate").filter((point) => point.y !== undefined)),
  };
}

export function TrainingCharts({ trainingState }) {
  const { epochs, batch_points: batchPoints } = trainingState;
  const getEpochPoints = (fieldName) => epochs.map((record) => ({ x: record.epoch, y: record[fieldName] }));
  const bestEpochNumber = getBestEpoch(epochs)?.epoch;
  const batchChartData = useMemo(() => getBatchChartData(batchPoints), [batchPoints]);
  const movingAverageName = `Moving average (${MOVING_AVERAGE_WINDOW})`;

  return (
    <Panel title="Learning curves" aside={<span className="muted">{epochs.length} epochs · {batchPoints.length} batches recorded</span>}>
      <div className="chart-grid">
        <LineChart title="Loss per epoch (Tumor-Aware loss)" xLabel="Epoch" yLabel="Loss" markerX={bestEpochNumber}
          series={[
            { name: "Training", color: CHART_COLORS.training, points: getEpochPoints("training_loss") },
            { name: "Validation", color: CHART_COLORS.validation, points: getEpochPoints("validation_loss") },
          ]} />
        <LineChart title="Dice per epoch" xLabel="Epoch" yLabel="Dice" markerX={bestEpochNumber}
          series={[
            { name: "Training", color: CHART_COLORS.training, points: getEpochPoints("training_dice") },
            { name: "Validation", color: CHART_COLORS.validation, points: getEpochPoints("validation_dice") },
          ]} />
        <LineChart title="Validation IoU per epoch" xLabel="Epoch" yLabel="IoU" markerX={bestEpochNumber}
          series={[{ name: "Validation IoU", color: CHART_COLORS.iou, points: getEpochPoints("validation_iou") }]} />
        <LineChart title="Learning rate (warm-up + cosine, per step)" xLabel="Batch" yLabel="Learning rate"
          series={[{ name: "Learning rate", color: CHART_COLORS.neutral, points: batchChartData.learningRatePoints }]} />
        <LineChart title="Training loss per batch" xLabel="Batch" yLabel="Loss"
          series={[
            { name: "Batch", color: CHART_COLORS.training, points: batchChartData.lossPoints, strokeWidth: 1, opacity: 0.35 },
            { name: movingAverageName, color: CHART_COLORS.average, points: batchChartData.lossAveragePoints },
          ]} />
        <LineChart title="Training Dice per batch" xLabel="Batch" yLabel="Dice" yDomain={[0, 1]}
          series={[
            { name: "Batch", color: CHART_COLORS.training, points: batchChartData.dicePoints, strokeWidth: 1, opacity: 0.35 },
            { name: movingAverageName, color: CHART_COLORS.average, points: batchChartData.diceAveragePoints },
          ]} />
        <LineChart title="GPU memory per epoch" xLabel="Epoch" yLabel="MB"
          series={[{ name: "Peak memory", color: CHART_COLORS.iou, points: getEpochPoints("gpu_memory_megabytes").filter((point) => point.y !== null) }]} />
        <LineChart title="Epoch duration" xLabel="Epoch" yLabel="Seconds"
          series={[{ name: "Duration", color: CHART_COLORS.neutral, points: getEpochPoints("duration_seconds") }]} />
      </div>
    </Panel>
  );
}
