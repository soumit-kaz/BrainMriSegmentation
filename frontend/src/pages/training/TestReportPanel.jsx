import { Histogram } from "../../components/charts/Histogram.jsx";
import { LineChart } from "../../components/charts/LineChart.jsx";
import { Panel } from "../../components/common.jsx";
import { CHART_COLORS, PUBLISHED_MODEL_TEST_RESULTS } from "../../helpers.js";
import { formatDateTime, formatNumber } from "../../utils.js";

const STATISTIC_COLUMNS = [
  ["mean", "Mean"],
  ["median", "Median"],
  ["standard_deviation", "Std dev"],
  ["percentile_25", "25%"],
  ["percentile_75", "75%"],
];
const METRIC_LABELS = { dice: "Dice", iou: "IoU" };
const HARDEST_CASE_COUNT = 8;

function StatisticsRow({ modelLabel, metricName, statistics, isHighlighted }) {
  return (
    <tr className={isHighlighted ? "highlight-row" : undefined}>
      <td>{modelLabel}</td>
      <td>{METRIC_LABELS[metricName]}</td>
      {STATISTIC_COLUMNS.map(([statisticName]) => <td key={statisticName}>{formatNumber(statistics?.[statisticName])}</td>)}
    </tr>
  );
}

function HardestCasesTable({ caseRecords }) {
  const hardestCases = [...caseRecords].sort((first, second) => first.dice - second.dice).slice(0, HARDEST_CASE_COUNT);
  if (hardestCases.length === 0) return null;
  return (
    <>
      <h3>Hardest test cases</h3>
      <div className="table-scroll">
        <table className="data-table compact-table">
          <thead><tr><th>Case</th><th>Dice</th><th>IoU</th><th>Loss</th></tr></thead>
          <tbody>
            {hardestCases.map((record) => (
              <tr key={record.case_name}>
                <td>{record.case_name}</td><td>{formatNumber(record.dice)}</td><td>{formatNumber(record.iou)}</td><td>{formatNumber(record.loss)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function TestReportPanel({ testReport }) {
  if (!testReport) {
    return (
      <Panel title="Test set results">
        <p className="muted">Test results appear here when training finishes, or after "Evaluate on test set".</p>
      </Panel>
    );
  }
  const caseRecords = testReport.cases ?? [];
  const getSortedPoints = (metricName) =>
    caseRecords.map((record) => record[metricName]).sort((first, second) => first - second).map((value, index) => ({ x: index + 1, y: value }));

  return (
    <Panel title="Test set results"
      aside={<span className="muted">{testReport.case_count} cases · model from epoch {testReport.model_epoch} · {formatDateTime(testReport.evaluated_at)}</span>}>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr><th>Model</th><th>Metric</th>{STATISTIC_COLUMNS.map(([, label]) => <th key={label}>{label}</th>)}</tr>
          </thead>
          <tbody>
            {Object.keys(METRIC_LABELS).map((metricName) => (
              <StatisticsRow key={`current-${metricName}`} modelLabel="Modality-Aware Attention U-Net" metricName={metricName}
                statistics={testReport[metricName]} isHighlighted />
            ))}
            {Object.keys(METRIC_LABELS).map((metricName) => (
              <StatisticsRow key={`published-${metricName}`} modelLabel="3D Residual Attention U-Net (ICCA 2024)" metricName={metricName}
                statistics={PUBLISHED_MODEL_TEST_RESULTS[metricName]} isHighlighted={false} />
            ))}
          </tbody>
        </table>
      </div>
      <div className="chart-grid">
        <Histogram title="Distribution of per-case scores" xLabel="Score"
          seriesList={[
            { name: "Dice", color: CHART_COLORS.validation, values: caseRecords.map((record) => record.dice) },
            { name: "IoU", color: CHART_COLORS.iou, values: caseRecords.map((record) => record.iou) },
          ]} />
        <LineChart title="Per-case scores, sorted" xLabel="Case rank" yLabel="Score" yDomain={[0, 1]}
          series={[
            { name: "Dice", color: CHART_COLORS.validation, points: getSortedPoints("dice") },
            { name: "IoU", color: CHART_COLORS.iou, points: getSortedPoints("iou") },
          ]} />
      </div>
      <HardestCasesTable caseRecords={caseRecords} />
    </Panel>
  );
}
