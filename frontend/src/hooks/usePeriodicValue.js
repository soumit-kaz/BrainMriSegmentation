import { useCallback, useEffect, useState } from "react";

export function usePeriodicValue(loadValue, intervalMilliseconds) {
  const [value, setValue] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const refreshValue = useCallback(async () => {
    try {
      setValue(await loadValue());
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(error.message);
    }
  }, [loadValue]);

  useEffect(() => {
    refreshValue();
    if (!intervalMilliseconds) return undefined;
    const timerId = setInterval(refreshValue, intervalMilliseconds);
    return () => clearInterval(timerId);
  }, [refreshValue, intervalMilliseconds]);

  return { value, errorMessage, refreshValue };
}

export function useCurrentTime(isTicking) {
  const [currentTime, setCurrentTime] = useState(Date.now());
  useEffect(() => {
    if (!isTicking) return undefined;
    const timerId = setInterval(() => setCurrentTime(Date.now()), 1000);
    return () => clearInterval(timerId);
  }, [isTicking]);
  return currentTime;
}
