import { useEffect, useState } from "react";
import { getMaskDownloadUrl, getSliceImageUrl, getSliceMaskUrl } from "../../api/inferenceApi.js";
import { LineChart } from "../../components/charts/LineChart.jsx";
import { MetricTile, Panel } from "../../components/common.jsx";
import { CHART_COLORS, MODALITIES } from "../../helpers.js";
import { formatNumber, getClampedValue } from "../../utils.js";

const DEFAULT_MODALITY = "t2f";
const DEFAULT_OVERLAY_OPACITY = 0.45;

export function SegmentationViewer({ job }) {
  const summary = job.summary;
  const [selectedModality, setSelectedModality] = useState(DEFAULT_MODALITY);
  const [sliceIndex, setSliceIndex] = useState(summary.largest_tumor_slice_index);
  const [isPredictionVisible, setIsPredictionVisible] = useState(true);
  const [isGroundTruthVisible, setIsGroundTruthVisible] = useState(false);
  const [overlayOpacity, setOverlayOpacity] = useState(DEFAULT_OVERLAY_OPACITY);
  const getSliceIndex = (index) => getClampedValue(index, 0, summary.slice_count - 1);
  const tumorAreaPoints = summary.tumor_area_per_slice.map((area, index) => ({ x: index, y: area }));

  useEffect(() => {
    setSliceIndex(summary.largest_tumor_slice_index);
    setIsGroundTruthVisible(false);
  }, [job.job_id, summary.largest_tumor_slice_index]);

  return (
    <Panel title={job.case_name} aside={<a className="link-button" href={getMaskDownloadUrl(job.job_id)} download>Download mask (.nii.gz)</a>}>
      <div className="viewer-layout">
        <div>
          <div className="slice-stage" style={{ "--overlay-opacity": overlayOpacity }}
            onWheel={(wheelEvent) => setSliceIndex((index) => getSliceIndex(index + (wheelEvent.deltaY > 0 ? -1 : 1)))}>
            <img src={getSliceImageUrl(job.job_id, sliceIndex, selectedModality)} alt={`${selectedModality} slice ${sliceIndex}`} />
            {isGroundTruthVisible && job.has_ground_truth && <img className="overlay" src={getSliceMaskUrl(job.job_id, sliceIndex, "ground_truth")} alt="" />}
            {isPredictionVisible && <img className="overlay" src={getSliceMaskUrl(job.job_id, sliceIndex, "prediction")} alt="" />}
          </div>
          <label className="slider-label">
            Axial slice {sliceIndex} / {summary.slice_count - 1}
            <input type="range" min="0" max={summary.slice_count - 1} value={sliceIndex}
              onChange={(changeEvent) => setSliceIndex(Number(changeEvent.target.value))} />
          </label>
          <div className="thumbnail-row">
            {MODALITIES.map((modality) => (
              <button key={modality.key} className={`thumbnail ${selectedModality === modality.key ? "is-selected" : ""}`}
                onClick={() => setSelectedModality(modality.key)}>
                <img src={getSliceImageUrl(job.job_id, sliceIndex, modality.key)} alt={modality.label} />
                <span>{modality.label}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="viewer-sidebar">
          <div className="toggle-group">
            <label className="inline-label">
              <input type="checkbox" checked={isPredictionVisible} onChange={(changeEvent) => setIsPredictionVisible(changeEvent.target.checked)} />
              <i className="swatch swatch-prediction" /> Predicted tumor
            </label>
            <label className="inline-label">
              <input type="checkbox" checked={isGroundTruthVisible} disabled={!job.has_ground_truth}
                onChange={(changeEvent) => setIsGroundTruthVisible(changeEvent.target.checked)} />
              <i className="swatch swatch-ground-truth" /> Ground truth {job.has_ground_truth ? "" : "(not uploaded)"}
            </label>
            <label className="slider-label">
              Overlay opacity {Math.round(overlayOpacity * 100)}%
              <input type="range" min="0.1" max="1" step="0.05" value={overlayOpacity}
                onChange={(changeEvent) => setOverlayOpacity(Number(changeEvent.target.value))} />
            </label>
          </div>
          <div className="metric-grid compact-grid">
            <MetricTile label="Tumor volume" value={`${summary.tumor_volume_cubic_centimeters} cm³`} />
            <MetricTile label="Tumor voxels" value={summary.tumor_voxel_count.toLocaleString()} />
            <MetricTile label="Tumor slices" value={summary.tumor_slice_range ? summary.tumor_slice_range.join(" – ") : "none"} />
            <MetricTile label="Processing time" value={`${summary.inference_seconds}s`} hint={summary.device} />
            {summary.dice_score !== null && <MetricTile label="Dice vs ground truth" value={formatNumber(summary.dice_score)} />}
            {summary.iou_score !== null && <MetricTile label="IoU vs ground truth" value={formatNumber(summary.iou_score)} />}
          </div>
        </div>
      </div>
      <LineChart title="Predicted tumor area per axial slice (click to jump)" xLabel="Slice" yLabel="Voxels" height={180}
        markerX={sliceIndex} onPointSelect={(index) => setSliceIndex(getSliceIndex(index))}
        series={[{ name: "Tumor area", color: CHART_COLORS.validation, points: tumorAreaPoints }]} />
    </Panel>
  );
}
