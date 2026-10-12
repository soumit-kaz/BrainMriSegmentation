import { useEffect, useState } from "react";
import { getJobEventsUrl } from "../api/inferenceApi.js";
import { isTerminalJob } from "../helpers.js";

export function useJobUpdates(initialJob, onJobFinished) {
  const [job, setJob] = useState(initialJob);

  useEffect(() => {
    setJob(initialJob);
    if (!initialJob || isTerminalJob(initialJob)) return undefined;
    const eventSource = new EventSource(getJobEventsUrl(initialJob.job_id));
    eventSource.onmessage = (message) => {
      const updatedJob = JSON.parse(message.data);
      setJob(updatedJob);
      if (isTerminalJob(updatedJob)) {
        eventSource.close();
        onJobFinished?.(updatedJob);
      }
    };
    return () => eventSource.close();
  }, [initialJob, onJobFinished]);

  return job;
}
