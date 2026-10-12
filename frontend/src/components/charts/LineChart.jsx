import { useRef, useState } from "react";
import { formatTickValue, getTickValues } from "../../helpers.js";
import { formatNumber } from "../../utils.js";

export const CHART_WIDTH = 640;
export const CHART_MARGIN = { top: 12, right: 16, bottom: 36, left: 56 };

export function EmptyChart({ title, height }) {
  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <div className="chart-empty" style={{ height }}>No data yet</div>
    </figure>
  );
}

export function ChartLegend({ items }) {
  return (
    <span className="chart-legend">
      {items.map((item) => (
        <span key={item.name}><i style={{ background: item.color }} />{item.name}</span>
      ))}
    </span>
  );
}

function getValueRange(points, yDomain) {
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  if (yDomain) return { xMinimum: Math.min(...xValues), xMaximum: Math.max(...xValues), yMinimum: yDomain[0], yMaximum: yDomain[1] };
  const yMinimum = Math.min(...yValues);
  const yMaximum = Math.max(...yValues);
  const yPadding = (yMaximum - yMinimum) * 0.08 || Math.abs(yMaximum) * 0.1 || 1;
  return { xMinimum: Math.min(...xValues), xMaximum: Math.max(...xValues), yMinimum: yMinimum - yPadding, yMaximum: yMaximum + yPadding };
}

function getNearestPoint(points, xValue) {
  return points.reduce((nearest, point) => (Math.abs(point.x - xValue) < Math.abs(nearest.x - xValue) ? point : nearest), points[0]);
}

export function LineChart({ title, series, xLabel, yLabel, yDomain, height = 240, onPointSelect, markerX }) {
  const svgReference = useRef(null);
  const [hoverX, setHoverX] = useState(null);
  const visibleSeries = series.filter((item) => item.points.length > 0);
  const allPoints = visibleSeries.flatMap((item) => item.points);
  if (allPoints.length === 0) return <EmptyChart title={title} height={height} />;

  const plotWidth = CHART_WIDTH - CHART_MARGIN.left - CHART_MARGIN.right;
  const plotHeight = height - CHART_MARGIN.top - CHART_MARGIN.bottom;
  const { xMinimum, xMaximum, yMinimum, yMaximum } = getValueRange(allPoints, yDomain);
  const getScaledX = (value) => CHART_MARGIN.left + (xMaximum === xMinimum ? plotWidth / 2 : ((value - xMinimum) / (xMaximum - xMinimum)) * plotWidth);
  const getScaledY = (value) => CHART_MARGIN.top + plotHeight - ((value - yMinimum) / (yMaximum - yMinimum || 1)) * plotHeight;
  const getPathData = (points) =>
    points.map((point, index) => `${index === 0 ? "M" : "L"}${getScaledX(point.x).toFixed(1)},${getScaledY(point.y).toFixed(1)}`).join("");
  const getDataXFromMouse = (mouseEvent) => {
    const bounds = svgReference.current.getBoundingClientRect();
    const svgX = ((mouseEvent.clientX - bounds.left) / bounds.width) * CHART_WIDTH;
    const ratio = Math.min(1, Math.max(0, (svgX - CHART_MARGIN.left) / plotWidth));
    return xMinimum + ratio * (xMaximum - xMinimum);
  };
  const hoveredPoints = hoverX === null ? [] : visibleSeries.map((item) => ({ ...item, point: getNearestPoint(item.points, hoverX) }));
  const hoverLineX = hoveredPoints.length > 0 ? getScaledX(hoveredPoints[0].point.x) : null;

  return (
    <figure className="chart">
      <figcaption><span>{title}</span><ChartLegend items={visibleSeries} /></figcaption>
      <div className="chart-body">
        <svg
          ref={svgReference}
          viewBox={`0 0 ${CHART_WIDTH} ${height}`}
          role="img"
          aria-label={title}
          className={onPointSelect ? "is-clickable" : undefined}
          onMouseMove={(mouseEvent) => setHoverX(getDataXFromMouse(mouseEvent))}
          onMouseLeave={() => setHoverX(null)}
          onClick={(mouseEvent) => onPointSelect?.(Math.round(getDataXFromMouse(mouseEvent)))}
        >
          {getTickValues(yMinimum, yMaximum).map((tickValue) => (
            <g key={`y${tickValue}`}>
              <line className="grid-line" x1={CHART_MARGIN.left} x2={CHART_WIDTH - CHART_MARGIN.right} y1={getScaledY(tickValue)} y2={getScaledY(tickValue)} />
              <text className="tick-label" x={CHART_MARGIN.left - 6} y={getScaledY(tickValue) + 4} textAnchor="end">{formatTickValue(tickValue)}</text>
            </g>
          ))}
          {getTickValues(xMinimum, xMaximum, 6).map((tickValue) => (
            <text key={`x${tickValue}`} className="tick-label" x={getScaledX(tickValue)} y={height - CHART_MARGIN.bottom + 16} textAnchor="middle">
              {formatTickValue(tickValue)}
            </text>
          ))}
          <text className="axis-label" x={CHART_MARGIN.left + plotWidth / 2} y={height - 4} textAnchor="middle">{xLabel}</text>
          <text className="axis-label" transform={`translate(14 ${CHART_MARGIN.top + plotHeight / 2}) rotate(-90)`} textAnchor="middle">{yLabel}</text>
          {markerX !== undefined && markerX !== null && (
            <line className="marker-line" x1={getScaledX(markerX)} x2={getScaledX(markerX)} y1={CHART_MARGIN.top} y2={CHART_MARGIN.top + plotHeight} />
          )}
          {visibleSeries.map((item) =>
            item.points.length === 1 ? (
              <circle key={item.name} cx={getScaledX(item.points[0].x)} cy={getScaledY(item.points[0].y)} r="3.5" fill={item.color} />
            ) : (
              <path key={item.name} d={getPathData(item.points)} fill="none" stroke={item.color} strokeWidth={item.strokeWidth ?? 2}
                strokeOpacity={item.opacity ?? 1} strokeLinejoin="round" />
            ),
          )}
          {hoverLineX !== null && (
            <g>
              <line className="hover-line" x1={hoverLineX} x2={hoverLineX} y1={CHART_MARGIN.top} y2={CHART_MARGIN.top + plotHeight} />
              {hoveredPoints.map((item) => (
                <circle key={item.name} cx={getScaledX(item.point.x)} cy={getScaledY(item.point.y)} r="4" fill={item.color} stroke="var(--surface)" strokeWidth="1.5" />
              ))}
            </g>
          )}
        </svg>
        {hoveredPoints.length > 0 && (
          <div className="chart-tooltip" style={{ left: `${(hoverLineX / CHART_WIDTH) * 100}%` }}>
            <strong>{xLabel} {formatTickValue(Number(hoveredPoints[0].point.x.toFixed(2)))}</strong>
            {hoveredPoints.map((item) => (
              <span key={item.name}><i style={{ background: item.color }} />{item.name}: {formatNumber(item.point.y, 4)}</span>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}
