import { Panel, StateBadge } from "../../components/common.jsx";
import { isTerminalJob } from "../../helpers.js";
import { formatDateTime } from "../../utils.js";

function getJobDescription(job) {
  if (job.status === "queued") return `position ${job.queue_position}`;
  if (job.status === "failed") return job.error_message ?? "failed";
  if (job.summary) return `${job.summary.tumor_volume_cubic_centimeters} cm³`;
  return job.status;
}

export function JobListPanel({ jobs, selectedJobId, onSelect, onDelete }) {
  return (
    <Panel title="Segmentation jobs">
      {jobs.length === 0 ? (
        <p className="muted">No segmentation jobs yet.</p>
      ) : (
        <ul className="history-list">
          {jobs.map((job) => (
            <li key={job.job_id} className={job.job_id === selectedJobId ? "is-selected" : ""}>
              <button className="history-item" onClick={() => onSelect(job)}>
                <span className="history-title"><strong>{job.case_name}</strong><StateBadge state={job.status} /></span>
                <span className="muted">{formatDateTime(job.created_at)} · {getJobDescription(job)}</span>
              </button>
              <button className="icon-button" aria-label={`Delete ${job.case_name}`} disabled={!isTerminalJob(job)} onClick={() => onDelete(job)}>✕</button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
