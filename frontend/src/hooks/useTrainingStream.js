import { useEffect, useReducer, useState } from "react";
import { trainingEventsUrl } from "../api/trainingApi.js";

const MAXIMUM_LOG_COUNT = 400;

function getTrainingStateWithEvent(trainingState, event) {
  if (event.type === "snapshot") return event.snapshot;
  if (trainingState === null) return trainingState;
  switch (event.type) {
    case "state":
      return { ...trainingState, state: event.state };
    case "run_started":
      return { ...trainingState, state: "running", run: event.run };
    case "progress": {
      const nextState = { ...trainingState, progress: event, state: trainingState.state === "starting" ? "running" : trainingState.state };
      if (event.batch_point) nextState.batch_points = [...trainingState.batch_points, event.batch_point];
      return nextState;
    }
    case "epoch":
      return { ...trainingState, epochs: [...trainingState.epochs, event.record] };
    case "test_report":
      return { ...trainingState, test_report: event.report };
    case "log":
      return { ...trainingState, logs: [...trainingState.logs, event].slice(-MAXIMUM_LOG_COUNT) };
    case "run_finished":
      return { ...trainingState, state: event.state, message: event.message, finished_at: event.timestamp };
    default:
      return trainingState;
  }
}

export function useTrainingStream() {
  const [trainingState, addTrainingEvent] = useReducer(getTrainingStateWithEvent, null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const eventSource = new EventSource(trainingEventsUrl);
    eventSource.onopen = () => setIsConnected(true);
    eventSource.onerror = () => setIsConnected(false);
    eventSource.onmessage = (message) => addTrainingEvent(JSON.parse(message.data));
    return () => eventSource.close();
  }, []);

  return { trainingState, isConnected };
}
