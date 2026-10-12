import { useState } from "react";
import { StateBadge } from "./components/common.jsx";
import { useTrainingStream } from "./hooks/useTrainingStream.js";
import { SegmentationPage } from "./pages/segmentation/SegmentationPage.jsx";
import { TrainingPage } from "./pages/training/TrainingPage.jsx";

const PAGES = [
  { key: "training", label: "Training" },
  { key: "segmentation", label: "Segmentation" },
];

export default function App() {
  const [activePageKey, setActivePageKey] = useState(PAGES[0].key);
  const { trainingState, isConnected } = useTrainingStream();

  return (
    <div className="application">
      <header className="application-header">
        <div className="brand">
          <span className="brand-title">Glioma Segmentation</span>
          <span className="brand-subtitle">Modality-Aware Attention U-Net · BraTS 2023</span>
        </div>
        <nav className="page-tabs">
          {PAGES.map((page) => (
            <button key={page.key} className={activePageKey === page.key ? "is-active" : ""} onClick={() => setActivePageKey(page.key)}>
              {page.label}
            </button>
          ))}
        </nav>
        <div className="header-status">
          <i className={isConnected ? "dot dot-online" : "dot dot-offline"} />
          Training: <StateBadge state={trainingState?.state} />
        </div>
      </header>
      <main className="page-content">
        {activePageKey === "training" ? <TrainingPage trainingState={trainingState} isConnected={isConnected} /> : <SegmentationPage />}
      </main>
    </div>
  );
}
