export function StateBadge({ state }) {
  return <span className={`state-badge state-${state ?? "unknown"}`}>{state ?? "unknown"}</span>;
}

export function MetricTile({ label, value, hint }) {
  return (
    <div className="metric-tile">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {hint && <span className="metric-hint">{hint}</span>}
    </div>
  );
}

export function ErrorMessage({ message }) {
  return message ? <div className="error-message" role="alert">{message}</div> : null;
}

export function Panel({ title, aside, children }) {
  return (
    <section className="panel">
      <header className="panel-header">
        <h2>{title}</h2>
        {aside}
      </header>
      {children}
    </section>
  );
}

export function ConnectionIndicator({ isConnected, connectedLabel, disconnectedLabel }) {
  return (
    <span className="connection-status">
      <i className={isConnected ? "dot dot-online" : "dot dot-offline"} />
      {isConnected ? connectedLabel : disconnectedLabel}
    </span>
  );
}
