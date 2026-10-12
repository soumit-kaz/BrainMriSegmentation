import { useCallback, useState } from "react";
import { deleteSegmentationJob, getModelInformation, getQueueStatus, getSegmentationJobs } from "../../api/inferenceApi.js";
import { ErrorMessage, MetricTile, Panel, StateBadge } from "../../components/common.jsx";
import { useJobUpdates } from "../../hooks/useJobUpdates.js";
import { usePeriodicValue } from "../../hooks/usePeriodicValue.js";
import { formatDuration } from "../../utils.js";
import { InferenceSystemPanel } from "./InferenceSystemPanel.jsx";
import { JobListPanel } from "./JobListPanel.jsx";
import { SegmentationViewer } from "./SegmentationViewer.jsx";
import { UploadPanel } from "./UploadPanel.jsx";

const SYSTEM_REFRESH_MILLISECONDS = 5000;

function JobProgressPanel({ job, averageProcessingSeconds }) {
  const estimatedWaitSeconds = job.status === "queued" ? job.queue_position * averageProcessingSeconds : null;
  return (
    <Panel title={job.case_name} aside={<StateBadge state={job.status} />}>
      <div className="metric-grid">
        <MetricTile label="Status" value={job.status} hint={job.is_reused ? "identical scan already submitted" : null} />
        <MetricTile label="Queue position" value={job.status === "queued" ? job.queue_position : "–"} />
        <MetricTile label="Estimated wait" value={estimatedWaitSeconds === null ? "–" : `≈ ${formatDuration(estimatedWaitSeconds)}`} />
        <MetricTile label="Attempts" value={job.attempt_count} />
      </div>
      {job.status === "failed" && <ErrorMessage message={job.error_message} />}
    </Panel>
  );
}

export function SegmentationPage() {
  const [selectedJob, setSelectedJob] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const { value: jobs, refreshValue: refreshJobs } = usePeriodicValue(getSegmentationJobs, SYSTEM_REFRESH_MILLISECONDS);
  const { value: modelInformation } = usePeriodicValue(getModelInformation, SYSTEM_REFRESH_MILLISECONDS);
  const { value: queueStatus } = usePeriodicValue(getQueueStatus, SYSTEM_REFRESH_MILLISECONDS);
  const trackedJob = useJobUpdates(selectedJob, refreshJobs);

  const selectSubmittedJob = useCallback((submittedJob) => {
    setSelectedJob(submittedJob);
    refreshJobs();
  }, [refreshJobs]);

  const deleteJob = async (job) => {
    try {
      await deleteSegmentationJob(job.job_id);
      if (selectedJob?.job_id === job.job_id) setSelectedJob(null);
      refreshJobs();
    } catch (error) {
      setErrorMessage(error.message);
    }
  };

  return (
    <>
      <ErrorMessage message={errorMessage} />
      <div className="two-column">
        <UploadPanel onJobSubmitted={selectSubmittedJob} />
        <InferenceSystemPanel modelInformation={modelInformation} queueStatus={queueStatus} />
      </div>
      {trackedJob && trackedJob.status === "completed" && trackedJob.summary
        ? <SegmentationViewer job={trackedJob} />
        : trackedJob && <JobProgressPanel job={trackedJob} averageProcessingSeconds={queueStatus?.average_processing_seconds ?? 0} />}
      <JobListPanel jobs={jobs ?? []} selectedJobId={trackedJob?.job_id} onSelect={setSelectedJob} onDelete={deleteJob} />
    </>
  );
}
