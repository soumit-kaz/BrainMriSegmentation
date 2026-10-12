import { useEffect, useState } from "react";
import { evaluateTrainedModel, getDatasetInformation, resumeTraining, startTraining, stopTraining } from "../../api/trainingApi.js";
import { ErrorMessage, Panel } from "../../components/common.jsx";
import { ACTIVE_TRAINING_STATES } from "../../helpers.js";

const DEFAULT_TRAINING_OPTIONS = { epoch_count: 60, batch_size: 2, learning_rate: 0.001, maximum_case_count: 0 };

function DatasetStatus({ datasetInformation }) {
  if (!datasetInformation) return null;
  if (datasetInformation.is_available) return <span className="muted">{datasetInformation.case_count} BraTS cases found</span>;
  return <span className="warning-text">{datasetInformation.message ?? "No BraTS cases found in the dataset directory"}</span>;
}

export function TrainingControlPanel({ trainingState }) {
  const [trainingOptions, setTrainingOptions] = useState(DEFAULT_TRAINING_OPTIONS);
  const [datasetInformation, setDatasetInformation] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isActive = ACTIVE_TRAINING_STATES.includes(trainingState.state);
  const hasHistory = trainingState.epochs.length > 0;

  useEffect(() => {
    getDatasetInformation().then(setDatasetInformation).catch((error) => setErrorMessage(error.message));
  }, []);

  const updateOption = (optionName) => (changeEvent) =>
    setTrainingOptions((previousOptions) => ({ ...previousOptions, [optionName]: Number(changeEvent.target.value) }));

  const runTrainingCommand = async (sendCommand) => {
    setErrorMessage(null);
    setIsSubmitting(true);
    try {
      await sendCommand();
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Panel title="Training control" aside={<DatasetStatus datasetInformation={datasetInformation} />}>
      <div className="form-grid">
        <label>Epochs<input type="number" min="1" max="1000" value={trainingOptions.epoch_count} onChange={updateOption("epoch_count")} disabled={isActive} /></label>
        <label>Batch size<input type="number" min="1" max="32" value={trainingOptions.batch_size} onChange={updateOption("batch_size")} disabled={isActive} /></label>
        <label>Learning rate<input type="number" min="0.000001" max="1" step="0.0001" value={trainingOptions.learning_rate} onChange={updateOption("learning_rate")} disabled={isActive} /></label>
        <label>Case limit (0 = all)<input type="number" min="0" value={trainingOptions.maximum_case_count} onChange={updateOption("maximum_case_count")} disabled={isActive} /></label>
      </div>
      <div className="button-row">
        <button className="primary-button" disabled={isActive || isSubmitting} onClick={() => runTrainingCommand(() => startTraining(trainingOptions))}>
          Start new training
        </button>
        <button disabled={isActive || isSubmitting || !hasHistory} onClick={() => runTrainingCommand(() => resumeTraining(trainingOptions.epoch_count))}>
          Resume
        </button>
        <button disabled={isActive || isSubmitting || !hasHistory} onClick={() => runTrainingCommand(evaluateTrainedModel)}>
          Evaluate on test set
        </button>
        <button className="danger-button" disabled={!isActive || trainingState.state === "stopping" || isSubmitting} onClick={() => runTrainingCommand(stopTraining)}>
          Stop
        </button>
      </div>
      <p className="muted small-text">
        Modality-Aware Attention U-Net with the Tumor-Aware loss. Mixed precision, gradient checkpointing and automatic micro-batch
        fitting adapt the run to the available GPU memory. Starting a new training replaces the previous run's history.
      </p>
      <ErrorMessage message={errorMessage} />
    </Panel>
  );
}
