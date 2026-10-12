import { formatTickValue, getTickValues } from "../../helpers.js";
import { formatNumber } from "../../utils.js";
import { CHART_MARGIN, CHART_WIDTH, ChartLegend, EmptyChart } from "./LineChart.jsx";

function getBinCounts(values, binCount, domainMinimum, binWidth) {
  const binCounts = new Array(binCount).fill(0);
  values.forEach((value) => {
    const binIndex = Math.min(binCount - 1, Math.max(0, Math.floor((value - domainMinimum) / binWidth)));
    binCounts[binIndex] += 1;
  });
  return binCounts;
}

export function Histogram({ title, seriesList, binCount = 20, domain = [0, 1], xLabel, height = 240 }) {
  const visibleSeriesList = seriesList.filter((item) => item.values.length > 0);
  if (visibleSeriesList.length === 0) return <EmptyChart title={title} height={height} />;

  const [domainMinimum, domainMaximum] = domain;
  const binWidth = (domainMaximum - domainMinimum) / binCount;
  const binnedSeries = visibleSeriesList.map((item) => ({ ...item, binCounts: getBinCounts(item.values, binCount, domainMinimum, binWidth) }));
  const maximumCount = Math.max(1, ...binnedSeries.flatMap((item) => item.binCounts));
  const plotWidth = CHART_WIDTH - CHART_MARGIN.left - CHART_MARGIN.right;
  const plotHeight = height - CHART_MARGIN.top - CHART_MARGIN.bottom;
  const binPixelWidth = plotWidth / binCount;
  const barPixelWidth = (binPixelWidth - 2) / binnedSeries.length;
  const getScaledY = (count) => CHART_MARGIN.top + plotHeight - (count / maximumCount) * plotHeight;

  return (
    <figure className="chart">
      <figcaption><span>{title}</span><ChartLegend items={binnedSeries} /></figcaption>
      <svg viewBox={`0 0 ${CHART_WIDTH} ${height}`} role="img" aria-label={title}>
        {getTickValues(0, maximumCount, 4).map((tickValue) => (
          <g key={tickValue}>
            <line className="grid-line" x1={CHART_MARGIN.left} x2={CHART_WIDTH - CHART_MARGIN.right} y1={getScaledY(tickValue)} y2={getScaledY(tickValue)} />
            <text className="tick-label" x={CHART_MARGIN.left - 6} y={getScaledY(tickValue) + 4} textAnchor="end">{formatTickValue(tickValue)}</text>
          </g>
        ))}
        {binnedSeries.map((item, seriesIndex) =>
          item.binCounts.map((count, binIndex) => (
            <rect key={`${item.name}${binIndex}`} fill={item.color} opacity="0.85"
              x={CHART_MARGIN.left + binIndex * binPixelWidth + 1 + seriesIndex * barPixelWidth}
              y={getScaledY(count)} width={Math.max(barPixelWidth, 1)} height={CHART_MARGIN.top + plotHeight - getScaledY(count)}>
              <title>{`${item.name} ${formatNumber(domainMinimum + binIndex * binWidth, 2)}–${formatNumber(domainMinimum + (binIndex + 1) * binWidth, 2)}: ${count} cases`}</title>
            </rect>
          )),
        )}
        {getTickValues(domainMinimum, domainMaximum, 5).map((tickValue) => (
          <text key={`x${tickValue}`} className="tick-label" textAnchor="middle" y={height - CHART_MARGIN.bottom + 16}
            x={CHART_MARGIN.left + ((tickValue - domainMinimum) / (domainMaximum - domainMinimum)) * plotWidth}>
            {formatTickValue(tickValue)}
          </text>
        ))}
        <text className="axis-label" x={CHART_MARGIN.left + plotWidth / 2} y={height - 4} textAnchor="middle">{xLabel}</text>
        <text className="axis-label" transform={`translate(14 ${CHART_MARGIN.top + plotHeight / 2}) rotate(-90)`} textAnchor="middle">Cases</text>
      </svg>
    </figure>
  );
}
