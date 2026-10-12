import { useState } from "react";
import { addSegmentationJob } from "../../api/inferenceApi.js";
import { ErrorMessage, Panel } from "../../components/common.jsx";
import { MODALITIES, UPLOAD_FIELDS, getUploadFieldKey } from "../../helpers.js";
import { formatDuration } from "../../utils.js";

const TOO_MANY_REQUESTS_STATUS = 429;

function getSubmissionErrorMessage(error) {
  if (error.statusCode === TOO_MANY_REQUESTS_STATUS && error.retryAfterSeconds) {
    return `${error.message} Retry in about ${formatDuration(error.retryAfterSeconds)}.`;
  }
  return error.message;
}

export function UploadPanel({ onJobSubmitted }) {
  const [selectedFiles, setSelectedFiles] = useState({});
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const hasRequiredFiles = MODALITIES.every((modality) => selectedFiles[modality.key]);

  const addFiles = (fileList) => {
    const recognisedFiles = {};
    Array.from(fileList).forEach((file) => {
      const fieldKey = getUploadFieldKey(file.name);
      if (fieldKey) recognisedFiles[fieldKey] = file;
    });
    if (Object.keys(recognisedFiles).length === 0) {
      setErrorMessage("No files matched *-t1n / -t1c / -t2w / -t2f / -seg .nii(.gz). Use the individual pickers instead.");
      return;
    }
    setErrorMessage(null);
    setSelectedFiles((previousFiles) => ({ ...previousFiles, ...recognisedFiles }));
  };

  const submitJob = async (submitEvent) => {
    submitEvent.preventDefault();
    setErrorMessage(null);
    setIsSubmitting(true);
    try {
      const formData = new FormData();
      UPLOAD_FIELDS.forEach((field) => selectedFiles[field.key] && formData.append(field.key, selectedFiles[field.key]));
      onJobSubmitted(await addSegmentationJob(formData));
      setSelectedFiles({});
    } catch (error) {
      setErrorMessage(getSubmissionErrorMessage(error));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Panel title="New segmentation">
      <form onSubmit={submitJob}>
        <div
          className={`drop-zone ${isDragging ? "is-dragging" : ""}`}
          onDragOver={(dragEvent) => { dragEvent.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(dropEvent) => { dropEvent.preventDefault(); setIsDragging(false); addFiles(dropEvent.dataTransfer.files); }}
        >
          Drop a BraTS case's files here (…-t1n, -t1c, -t2w, -t2f and optional -seg .nii.gz)
          <label className="link-button">
            or choose files
            <input type="file" multiple accept=".nii,.gz" hidden onChange={(changeEvent) => addFiles(changeEvent.target.files)} />
          </label>
        </div>
        <div className="upload-grid">
          {UPLOAD_FIELDS.map((field) => (
            <label key={field.key} className={`upload-slot ${selectedFiles[field.key] ? "is-filled" : ""}`}>
              <span className="upload-label">{field.label}</span>
              <span className="upload-file-name">{selectedFiles[field.key]?.name ?? "Not selected"}</span>
              <input type="file" accept=".nii,.gz" hidden
                onChange={(changeEvent) => setSelectedFiles((previousFiles) => ({ ...previousFiles, [field.key]: changeEvent.target.files[0] }))} />
            </label>
          ))}
        </div>
        <div className="button-row">
          <button type="submit" className="primary-button" disabled={!hasRequiredFiles || isSubmitting}>
            {isSubmitting ? "Uploading…" : "Queue segmentation"}
          </button>
        </div>
      </form>
      <ErrorMessage message={errorMessage} />
    </Panel>
  );
}
