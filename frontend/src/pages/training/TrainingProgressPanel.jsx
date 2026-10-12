import { ConnectionIndicator, MetricTile, Panel, StateBadge } from "../../components/common.jsx";
import { useCurrentTime } from "../../hooks/usePeriodicValue.js";
import { ACTIVE_TRAINING_STATES, getBestEpoch, getEstimatedRemainingSeconds } from "../../helpers.js";
import { formatDuration, formatNumber, formatTime, getSecondsBetween } from "../../utils.js";

function ProgressBar({ progress, run }) {
  if (!progress) return null;
  const progressPercent = Math.round((progress.step / progress.step_count) * 100);
  const epochLabel = progress.phase === "caching" ? "Preparing cache" : `Epoch ${progress.epoch}${run ? ` / ${run.epoch_count}` : ""}`;
  return (
    <div className="progress-block">
      <div className="progress-caption">
        <span>{epochLabel} · {progress.phase}</span>
        <span>{progress.step} / {progress.step_count}</span>
      </div>
      <div className="progress-track"><div className="progress-fill" style={{ width: `${progressPercent}%` }} /></div>
    </div>
  );
}

export function TrainingProgressPanel({ trainingState, isConnected }) {
  const isActive = ACTIVE_TRAINING_STATES.includes(trainingState.state);
  useCurrentTime(isActive);
  const { progress, run } = trainingState;
  const bestEpoch = getBestEpoch(trainingState.epochs);
  const remainingSeconds = isActive ? getEstimatedRemainingSeconds(trainingState) : null;
  const batchDescription = run ? `${run.micro_batch_size} × ${run.accumulation_step_count} accumulation` : null;

  return (
    <Panel title="Live status" aside={<ConnectionIndicator isConnected={isConnected} connectedLabel="Live (SSE)" disconnectedLabel="Reconnecting…" />}>
      <div className="status-line">
        <StateBadge state={trainingState.state} />
        {trainingState.command && <span className="muted">{trainingState.command}</span>}
        {trainingState.message && <span className="muted">{trainingState.message}</span>}
      </div>
      <ProgressBar progress={progress} run={run} />
      <div className="metric-grid">
        <MetricTile label="Device" value={run?.device_name ?? "–"} hint={run?.precision} />
        <MetricTile label="GPU memory (peak)" value={progress?.gpu_memory_megabytes ? `${Math.round(progress.gpu_memory_megabytes)} MB` : "–"} />
        <MetricTile label="Batch" value={run?.batch_size ?? "–"} hint={batchDescription} />
        <MetricTile label="Current loss" value={formatNumber(progress?.loss)} hint={progress?.phase} />
        <MetricTile label="Current Dice" value={formatNumber(progress?.dice)} hint={progress?.phase} />
        <MetricTile label="Best validation Dice" value={formatNumber(bestEpoch?.validation_dice)} hint={bestEpoch ? `epoch ${bestEpoch.epoch}` : null} />
        <MetricTile label="Elapsed" value={formatDuration(getSecondsBetween(trainingState.started_at, trainingState.finished_at))} hint={`started ${formatTime(trainingState.started_at)}`} />
        <MetricTile label="Remaining (estimate)" value={remainingSeconds === null ? "–" : `≈ ${formatDuration(remainingSeconds)}`} hint="early stopping may end sooner" />
        <MetricTile label="Parameters" value={run ? run.parameter_count.toLocaleString() : "–"} hint={run?.model_name} />
        <MetricTile label="Cases" value={run ? `${run.training_case_count} / ${run.validation_case_count} / ${run.test_case_count}` : "–"} hint="train / validation / test" />
      </div>
    </Panel>
  );
}
