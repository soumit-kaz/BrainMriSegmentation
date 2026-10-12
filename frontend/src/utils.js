export function formatNumber(value, digitCount = 4) {
  return value === null || value === undefined || Number.isNaN(value) ? "–" : Number(value).toFixed(digitCount);
}

export function formatDuration(totalSeconds) {
  if (totalSeconds === null || totalSeconds === undefined || !Number.isFinite(totalSeconds)) return "–";
  const roundedSeconds = Math.max(0, Math.round(totalSeconds));
  const hours = Math.floor(roundedSeconds / 3600);
  const minutes = Math.floor((roundedSeconds % 3600) / 60);
  const seconds = roundedSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

export function formatTime(timestamp) {
  return timestamp ? new Date(timestamp).toLocaleTimeString() : "–";
}

export function formatDateTime(timestamp) {
  return timestamp ? new Date(timestamp).toLocaleString() : "–";
}

export function getSecondsBetween(startTimestamp, endTimestamp) {
  if (!startTimestamp) return null;
  const endTime = endTimestamp ? new Date(endTimestamp).getTime() : Date.now();
  return (endTime - new Date(startTimestamp).getTime()) / 1000;
}

export function getClampedValue(value, minimumValue, maximumValue) {
  return Math.min(maximumValue, Math.max(minimumValue, value));
}
