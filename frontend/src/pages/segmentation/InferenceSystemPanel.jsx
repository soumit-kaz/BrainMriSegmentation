import { MetricTile, Panel, StateBadge } from "../../components/common.jsx";
import { formatDuration, formatNumber } from "../../utils.js";

function getWorkerState(modelInformation) {
  if (!modelInformation?.is_worker_online) return "offline";
  return modelInformation.is_loaded ? "ready" : "unavailable";
}

export function InferenceSystemPanel({ modelInformation, queueStatus }) {
  return (
    <Panel title="Inference system" aside={<StateBadge state={getWorkerState(modelInformation)} />}>
      <div className="metric-grid">
        <MetricTile label="Queued jobs" value={queueStatus ? `${queueStatus.queued_job_count} / ${queueStatus.maximum_queue_length}` : "–"} />
        <MetricTile label="Estimated wait" value={queueStatus ? formatDuration(queueStatus.estimated_wait_seconds) : "–"}
          hint={queueStatus ? `${formatNumber(queueStatus.average_processing_seconds, 1)} s per job` : null} />
        <MetricTile label="Device" value={modelInformation?.device_name ?? "–"} hint={modelInformation?.precision} />
        <MetricTile label="Checkpoint epoch" value={modelInformation?.epoch ?? "–"} hint={modelInformation?.model_name} />
        <MetricTile label="Validation Dice" value={formatNumber(modelInformation?.metrics?.validation_dice)} />
        <MetricTile label="Test-time augmentation" value={modelInformation?.is_test_time_augmentation_enabled ? "on" : "off"} />
      </div>
      {modelInformation?.message && <p className="warning-text small-text">{modelInformation.message}</p>}
    </Panel>
  );
}
