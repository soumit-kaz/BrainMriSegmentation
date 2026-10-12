import { useEffect, useRef, useState } from "react";
import { Panel } from "../../components/common.jsx";
import { formatTime } from "../../utils.js";
import { TestReportPanel } from "./TestReportPanel.jsx";
import { TrainingCharts } from "./TrainingCharts.jsx";
import { TrainingControlPanel } from "./TrainingControlPanel.jsx";
import { TrainingProgressPanel } from "./TrainingProgressPanel.jsx";

function LogConsole({ logs }) {
  const consoleReference = useRef(null);
  const [isFollowing, setIsFollowing] = useState(true);

  useEffect(() => {
    if (isFollowing && consoleReference.current) consoleReference.current.scrollTop = consoleReference.current.scrollHeight;
  }, [logs, isFollowing]);

  const followToggle = (
    <label className="inline-label">
      <input type="checkbox" checked={isFollowing} onChange={(changeEvent) => setIsFollowing(changeEvent.target.checked)} />
      Follow
    </label>
  );

  return (
    <Panel title="Training log" aside={followToggle}>
      <pre className="log-console" ref={consoleReference}>
        {logs.length === 0 ? <span className="muted">No log output for this run yet.</span> : logs.map((logEntry, index) => (
          <div key={index} className={`log-line log-${logEntry.level?.toLowerCase()}`}>
            <span className="log-time">{formatTime(logEntry.timestamp)}</span> {logEntry.message}
          </div>
        ))}
      </pre>
    </Panel>
  );
}

export function TrainingPage({ trainingState, isConnected }) {
  if (!trainingState) {
    return <Panel title="Training"><p className="muted">Connecting to the training service…</p></Panel>;
  }
  return (
    <>
      <div className="two-column">
        <TrainingControlPanel trainingState={trainingState} />
        <TrainingProgressPanel trainingState={trainingState} isConnected={isConnected} />
      </div>
      <TrainingCharts trainingState={trainingState} />
      <TestReportPanel testReport={trainingState.test_report} />
      <LogConsole logs={trainingState.logs} />
    </>
  );
}
