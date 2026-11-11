import React, { useState } from 'react';
import {
  Sliders,
  Award,
  Activity,
  Layers,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Eye,
  X,
  Play,
  RotateCcw,
} from 'lucide-react';
import { Hyperparameters, ModelMetrics, TrainedModelsState, EmailSample } from '../ml/types';

interface ModelLabViewProps {
  state: TrainedModelsState | null;
  hyperparameters: Hyperparameters;
  onUpdateHyperparameters: (newParams: Partial<Hyperparameters>) => void;
  onRetrain: () => void;
  isTraining: boolean;
}

export const ModelLabView: React.FC<ModelLabViewProps> = ({
  state,
  hyperparameters,
  onUpdateHyperparameters,
  onRetrain,
  isTraining,
}) => {
  const [selectedModelTab, setSelectedModelTab] = useState<'mnb' | 'svm' | 'cnb'>('mnb');
  const [inspectedSample, setInspectedSample] = useState<any | null>(null);

  if (!state) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-16 text-center text-slate-500">
        Loading models and evaluating benchmarks...
      </div>
    );
  }

  const activeMetrics: ModelMetrics =
    selectedModelTab === 'mnb'
      ? state.naiveBayesMetrics
      : selectedModelTab === 'svm'
      ? state.svmMetrics
      : state.complementNbMetrics;

  const activeModelTitle =
    selectedModelTab === 'mnb'
      ? 'Multinomial Naive Bayes'
      : selectedModelTab === 'svm'
      ? 'Linear Support Vector Machine (SGD)'
      : 'Complement Naive Bayes';

  // SVG dimensions for ROC Curve
  const svgWidth = 420;
  const svgHeight = 280;
  const padding = 36;
  const plotW = svgWidth - padding * 2;
  const plotH = svgHeight - padding * 2;

  // Convert ROC points to SVG coordinates
  // X: FPR (0 to 1) -> padding to padding + plotW
  // Y: TPR (0 to 1) -> padding + plotH to padding
  const points = activeMetrics.rocCurve.map(pt => {
    const x = padding + pt.fpr * plotW;
    const y = padding + plotH - pt.tpr * plotH;
    return `${x},${y}`;
  });
  const pathD = points.length > 0 ? `M ${points.join(' L ')}` : '';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Model Lab & Evaluation Suite
          </h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Holdout test evaluation, confusion matrices, ROC curves, and hyperparameter tuning.
          </p>
        </div>

        {/* Global Dataset Split Metadata (Zero-Pill text with separators) */}
        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
          <span>Corpus: {state.totalCorpusSize} docs</span>
          <span aria-hidden="true">·</span>
          <span>Train: {state.trainSize} docs</span>
          <span aria-hidden="true">·</span>
          <span>Test: {state.testSize} docs</span>
          <span aria-hidden="true">·</span>
          <span>Vocab: {state.vocabularySize} tokens</span>
        </div>
      </div>

      {/* Cross-Model Comparison Table */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-xs">
        <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Award className="w-4 h-4 text-rose-500" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
              Benchmark Evaluation on Holdout Test Set ({state.testSize} samples)
            </h2>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            Stratified 75/25 Split
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 text-slate-500 dark:text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-3 px-6">Algorithm Model</th>
                <th className="py-3 px-4 text-right">Accuracy</th>
                <th className="py-3 px-4 text-right">Precision (Spam)</th>
                <th className="py-3 px-4 text-right">Recall (Spam)</th>
                <th className="py-3 px-4 text-right">F1-Score</th>
                <th className="py-3 px-4 text-right">ROC AUC</th>
                <th className="py-3 px-4 text-right">Training Latency</th>
                <th className="py-3 px-6 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800 font-mono tabular-nums">
              {/* MNB */}
              <tr
                className={`hover:bg-slate-50/80 dark:hover:bg-slate-800/50 transition-colors ${
                  selectedModelTab === 'mnb' ? 'bg-rose-500/5 dark:bg-rose-500/10' : ''
                }`}
              >
                <td className="py-3.5 px-6 font-sans font-medium text-slate-900 dark:text-white flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-rose-500" />
                  Multinomial Naive Bayes (MNB)
                </td>
                <td className="py-3.5 px-4 text-right font-bold text-slate-900 dark:text-white">
                  {(state.naiveBayesMetrics.accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-4 text-right">
                  {(state.naiveBayesMetrics.precision * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-4 text-right">
                  {(state.naiveBayesMetrics.recall * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-4 text-right font-semibold text-rose-500">
                  {state.naiveBayesMetrics.f1.toFixed(3)}
                </td>
                <td className="py-3.5 px-4 text-right">
                  {state.naiveBayesMetrics.auc.toFixed(3)}
                </td>
                <td className="py-3.5 px-4 text-right text-slate-400">
                  {state.naiveBayesMetrics.trainTimeMs.toFixed(1)}ms
                </td>
                <td className="py-3.5 px-6 text-center font-sans">
                  <button
                    onClick={() => setSelectedModelTab('mnb')}
                    className="px-2.5 py-1 text-xs font-semibold rounded text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 transition-colors"
                  >
                    Inspect
                  </button>
                </td>
              </tr>

              {/* SVM */}
              <tr
                className={`hover:bg-slate-50/80 dark:hover:bg-slate-800/50 transition-colors ${
                  selectedModelTab === 'svm' ? 'bg-rose-500/5 dark:bg-rose-500/10' : ''
                }`}
              >
                <td className="py-3.5 px-6 font-sans font-medium text-slate-900 dark:text-white flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-sky-500" />
                  Linear Support Vector Machine (Pegasos SGD)
                </td>
                <td className="py-3.5 px-4 text-right font-bold text-slate-900 dark:text-white">
                  {(state.svmMetrics.accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-4 text-right">
                  {(state.svmMetrics.precision * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-4 text-right">
                  {(state.svmMetrics.recall * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-4 text-right font-semibold text-sky-500">
                  {state.svmMetrics.f1.toFixed(3)}
                </td>
                <td className="py-3.5 px-4 text-right">
                  {state.svmMetrics.auc.toFixed(3)}
                </td>
                <td className="py-3.5 px-4 text-right text-slate-400">
                  {state.svmMetrics.trainTimeMs.toFixed(1)}ms
                </td>
                <td className="py-3.5 px-6 text-center font-sans">
                  <button
                    onClick={() => setSelectedModelTab('svm')}
                    className="px-2.5 py-1 text-xs font-semibold rounded text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 transition-colors"
                  >
                    Inspect
                  </button>
                </td>
              </tr>

              {/* CNB */}
              <tr
                className={`hover:bg-slate-50/80 dark:hover:bg-slate-800/50 transition-colors ${
                  selectedModelTab === 'cnb' ? 'bg-rose-500/5 dark:bg-rose-500/10' : ''
                }`}
              >
                <td className="py-3.5 px-6 font-sans font-medium text-slate-900 dark:text-white flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  Complement Naive Bayes (CNB)
                </td>
                <td className="py-3.5 px-4 text-right font-bold text-slate-900 dark:text-white">
                  {(state.complementNbMetrics.accuracy * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-4 text-right">
                  {(state.complementNbMetrics.precision * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-4 text-right">
                  {(state.complementNbMetrics.recall * 100).toFixed(1)}%
                </td>
                <td className="py-3.5 px-4 text-right font-semibold text-emerald-500">
                  {state.complementNbMetrics.f1.toFixed(3)}
                </td>
                <td className="py-3.5 px-4 text-right">
                  {state.complementNbMetrics.auc.toFixed(3)}
                </td>
                <td className="py-3.5 px-4 text-right text-slate-400">
                  {state.complementNbMetrics.trainTimeMs.toFixed(1)}ms
                </td>
                <td className="py-3.5 px-6 text-center font-sans">
                  <button
                    onClick={() => setSelectedModelTab('cnb')}
                    className="px-2.5 py-1 text-xs font-semibold rounded text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 transition-colors"
                  >
                    Inspect
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Diagnostic Section: Confusion Matrix & ROC Curve */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Confusion Matrix & Misclassifications (6 cols) */}
        <div className="lg:col-span-6 space-y-6">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs space-y-5">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-rose-500" />
                <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                  Confusion Matrix: {activeModelTitle}
                </h3>
              </div>
              <span className="text-xs text-slate-400">
                Total Test: {activeMetrics.testCount}
              </span>
            </div>

            {/* Matrix Visual Grid */}
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                {/* Header Row */}
                <div />
                <div className="font-semibold text-slate-700 dark:text-slate-300 py-1 bg-slate-50 dark:bg-slate-950/40 rounded">
                  Pred: SPAM
                </div>
                <div className="font-semibold text-slate-700 dark:text-slate-300 py-1 bg-slate-50 dark:bg-slate-950/40 rounded">
                  Pred: HAM
                </div>

                {/* Actual Spam Row */}
                <div className="flex items-center justify-end pr-2 font-semibold text-slate-700 dark:text-slate-300">
                  Actual SPAM
                </div>
                {/* TP */}
                <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                  <span className="text-xs text-emerald-600 dark:text-emerald-400 block font-medium">
                    True Positives (TP)
                  </span>
                  <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400 tabular-nums">
                    {activeMetrics.confusionMatrix.tp}
                  </span>
                  <span className="text-[11px] text-slate-400 block mt-1">Spam caught</span>
                </div>
                {/* FN */}
                <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-lg">
                  <span className="text-xs text-rose-600 dark:text-rose-400 block font-medium">
                    False Negatives (FN)
                  </span>
                  <span className="text-2xl font-bold font-mono text-rose-600 dark:text-rose-400 tabular-nums">
                    {activeMetrics.confusionMatrix.fn}
                  </span>
                  <span className="text-[11px] text-slate-400 block mt-1">Spam missed</span>
                </div>

                {/* Actual Ham Row */}
                <div className="flex items-center justify-end pr-2 font-semibold text-slate-700 dark:text-slate-300">
                  Actual HAM
                </div>
                {/* FP */}
                <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                  <span className="text-xs text-amber-600 dark:text-amber-400 block font-medium">
                    False Positives (FP)
                  </span>
                  <span className="text-2xl font-bold font-mono text-amber-600 dark:text-amber-400 tabular-nums">
                    {activeMetrics.confusionMatrix.fp}
                  </span>
                  <span className="text-[11px] text-slate-400 block mt-1">Ham flagged (Critical!)</span>
                </div>
                {/* TN */}
                <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
                  <span className="text-xs text-emerald-600 dark:text-emerald-400 block font-medium">
                    True Negatives (TN)
                  </span>
                  <span className="text-2xl font-bold font-mono text-emerald-600 dark:text-emerald-400 tabular-nums">
                    {activeMetrics.confusionMatrix.tn}
                  </span>
                  <span className="text-[11px] text-slate-400 block mt-1">Ham passed</span>
                </div>
              </div>
            </div>

            {/* Error Audit / Misclassified List */}
            <div className="pt-2 border-t border-slate-100 dark:border-slate-800 space-y-3">
              <span className="text-xs font-semibold text-slate-700 dark:text-slate-300 block">
                Misclassified Samples Audit ({activeMetrics.misclassifiedSamples.length} errors):
              </span>

              {activeMetrics.misclassifiedSamples.length > 0 ? (
                <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                  {activeMetrics.misclassifiedSamples.map((item, idx) => (
                    <div
                      key={idx}
                      onClick={() => setInspectedSample(item)}
                      className="p-2.5 bg-slate-50 dark:bg-slate-950/60 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-800 cursor-pointer transition-colors flex items-center justify-between text-xs"
                    >
                      <div className="truncate pr-3">
                        <span className="font-semibold text-slate-900 dark:text-white mr-2">
                          {item.sample.subject}
                        </span>
                        <span className="text-slate-400">from {item.sample.sender}</span>
                      </div>
                      <div className="shrink-0 flex items-center gap-2">
                        <span className="text-[11px] font-mono text-slate-400">
                          True: {item.sample.label.toUpperCase()} → Pred: {item.predicted.toUpperCase()}
                        </span>
                        <Eye className="w-3.5 h-3.5 text-slate-400" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5 py-1">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Zero classification errors on this holdout test split!</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: ROC Curve & Top Discriminative Words (6 cols) */}
        <div className="lg:col-span-6 space-y-6">
          {/* ROC Curve Chart */}
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-sky-500" />
                <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                  Receiver Operating Characteristic (ROC Curve)
                </h3>
              </div>
              <div className="text-xs font-mono font-bold text-sky-500 tabular-nums">
                AUC = {activeMetrics.auc.toFixed(3)}
              </div>
            </div>

            {/* SVG Plot */}
            <div className="flex justify-center">
              <svg width={svgWidth} height={svgHeight} className="overflow-visible">
                {/* Gridlines */}
                {[0.25, 0.5, 0.75, 1.0].map((v, i) => {
                  const y = padding + plotH - v * plotH;
                  const x = padding + v * plotW;
                  return (
                    <g key={i} className="text-[10px] fill-slate-400 dark:fill-slate-500">
                      <line
                        x1={padding}
                        y1={y}
                        x2={padding + plotW}
                        y2={y}
                        stroke="currentColor"
                        strokeDasharray="2,2"
                        className="text-slate-200 dark:text-slate-800"
                      />
                      <line
                        x1={x}
                        y1={padding}
                        x2={x}
                        y2={padding + plotH}
                        stroke="currentColor"
                        strokeDasharray="2,2"
                        className="text-slate-200 dark:text-slate-800"
                      />
                      <text x={padding - 6} y={y + 3} textAnchor="end">
                        {v.toFixed(2)}
                      </text>
                      <text x={x} y={padding + plotH + 14} textAnchor="middle">
                        {v.toFixed(2)}
                      </text>
                    </g>
                  );
                })}

                {/* Axes */}
                <line
                  x1={padding}
                  y1={padding + plotH}
                  x2={padding + plotW}
                  y2={padding + plotH}
                  stroke="currentColor"
                  className="text-slate-400 dark:text-slate-600"
                />
                <line
                  x1={padding}
                  y1={padding}
                  x2={padding}
                  y2={padding + plotH}
                  stroke="currentColor"
                  className="text-slate-400 dark:text-slate-600"
                />

                {/* Random Guess Diagonal (AUC = 0.5) */}
                <line
                  x1={padding}
                  y1={padding + plotH}
                  x2={padding + plotW}
                  y2={padding}
                  stroke="currentColor"
                  strokeDasharray="4,4"
                  className="text-slate-300 dark:text-slate-700"
                />

                {/* ROC Curve Path */}
                {pathD && (
                  <path
                    d={pathD}
                    fill="none"
                    stroke="#0284c7"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                )}

                {/* Data Points */}
                {activeMetrics.rocCurve.map((pt, i) => {
                  const cx = padding + pt.fpr * plotW;
                  const cy = padding + plotH - pt.tpr * plotH;
                  return (
                    <circle
                      key={i}
                      cx={cx}
                      cy={cy}
                      r="3"
                      fill="#0284c7"
                      className="hover:r-5 transition-all"
                    >
                      <title>{`Threshold: ${pt.threshold.toFixed(2)} | FPR: ${pt.fpr.toFixed(2)} | TPR: ${pt.tpr.toFixed(2)}`}</title>
                    </circle>
                  );
                })}

                {/* Axis Labels */}
                <text
                  x={padding + plotW / 2}
                  y={padding + plotH + 30}
                  textAnchor="middle"
                  className="text-[11px] fill-slate-500 font-medium"
                >
                  False Positive Rate (1 - Specificity)
                </text>
                <text
                  x={-(padding + plotH / 2)}
                  y={12}
                  textAnchor="middle"
                  transform="rotate(-90)"
                  className="text-[11px] fill-slate-500 font-medium"
                >
                  True Positive Rate (Sensitivity / Recall)
                </text>
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Global Vocabulary Feature Insights */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
          <div>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
              Global Discriminative Vocabulary (Trained Weights)
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              The highest weighted tokens across the vocabulary influencing spam vs ham boundaries.
            </p>
          </div>
          <span className="text-xs text-slate-400 font-mono">
            Total Vocab: {state.vocabularySize} tokens
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Top Spam Vocab */}
          <div className="space-y-2">
            <span className="text-xs font-bold text-rose-500 uppercase tracking-wider block">
              Top 12 Spam Predictive Words (Positive Weights)
            </span>
            <div className="flex flex-wrap gap-1.5">
              {state.topSpamVocab.slice(0, 12).map((item, idx) => (
                <div
                  key={idx}
                  className="px-2.5 py-1 text-xs font-mono bg-rose-500/10 border border-rose-500/20 text-rose-600 dark:text-rose-300 rounded flex items-center gap-1.5"
                >
                  <span>{item.word}</span>
                  <span className="text-[10px] opacity-75 tabular-nums">+{item.score.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Top Ham Vocab */}
          <div className="space-y-2">
            <span className="text-xs font-bold text-emerald-500 uppercase tracking-wider block">
              Top 12 Ham Predictive Words (Negative Weights)
            </span>
            <div className="flex flex-wrap gap-1.5">
              {state.topHamVocab.slice(0, 12).map((item, idx) => (
                <div
                  key={idx}
                  className="px-2.5 py-1 text-xs font-mono bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-300 rounded flex items-center gap-1.5"
                >
                  <span>{item.word}</span>
                  <span className="text-[10px] opacity-75 tabular-nums">-{item.score.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Hyperparameter Tuning & Retraining Form */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs space-y-6">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-rose-500" />
            <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
              Hyperparameter Tuning & Feature Extraction Pipeline
            </h3>
          </div>
          <button
            onClick={onRetrain}
            disabled={isTraining}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-rose-600 hover:bg-rose-500 rounded-lg shadow-sm transition-colors disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5" />
            <span>{isTraining ? 'Training Models...' : 'Apply & Retrain Models'}</span>
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 text-xs">
          {/* Alpha Laplace Smoothing */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="font-semibold text-slate-700 dark:text-slate-300">
                Laplace Smoothing (α)
              </label>
              <span className="font-mono text-rose-500 font-bold tabular-nums">
                {hyperparameters.alpha.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="3.0"
              step="0.1"
              value={hyperparameters.alpha}
              onChange={e => onUpdateHyperparameters({ alpha: parseFloat(e.target.value) })}
              className="w-full accent-rose-500"
            />
            <p className="text-[11px] text-slate-400">
              Prevents zero-probability underflow in Naive Bayes.
            </p>
          </div>

          {/* SVM Regularization C */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="font-semibold text-slate-700 dark:text-slate-300">
                SVM Regularization (C)
              </label>
              <span className="font-mono text-sky-500 font-bold tabular-nums">
                {hyperparameters.svmC.toFixed(1)}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="5.0"
              step="0.1"
              value={hyperparameters.svmC}
              onChange={e => onUpdateHyperparameters({ svmC: parseFloat(e.target.value) })}
              className="w-full accent-sky-500"
            />
            <p className="text-[11px] text-slate-400">
              Balances margin width vs misclassification penalty.
            </p>
          </div>

          {/* SVM Epochs */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="font-semibold text-slate-700 dark:text-slate-300">
                SGD Epochs (SVM)
              </label>
              <span className="font-mono text-slate-900 dark:text-white font-bold tabular-nums">
                {hyperparameters.svmEpochs}
              </span>
            </div>
            <input
              type="range"
              min="10"
              max="50"
              step="5"
              value={hyperparameters.svmEpochs}
              onChange={e => onUpdateHyperparameters({ svmEpochs: parseInt(e.target.value) })}
              className="w-full accent-slate-600"
            />
            <p className="text-[11px] text-slate-400">
              Iterations through training vectors.
            </p>
          </div>

          {/* Train/Test Split */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="font-semibold text-slate-700 dark:text-slate-300">
                Test Split Ratio
              </label>
              <span className="font-mono text-slate-900 dark:text-white font-bold tabular-nums">
                {(hyperparameters.testSplitRatio * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min="0.15"
              max="0.35"
              step="0.05"
              value={hyperparameters.testSplitRatio}
              onChange={e => onUpdateHyperparameters({ testSplitRatio: parseFloat(e.target.value) })}
              className="w-full accent-slate-600"
            />
            <p className="text-[11px] text-slate-400">
              Holdout percentage reserved for evaluation.
            </p>
          </div>
        </div>

        {/* Feature Extraction Pipeline Toggles */}
        <div className="pt-4 border-t border-slate-100 dark:border-slate-800 grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
          {/* Vectorizer Type */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-700 dark:text-slate-300 block">
              Vectorization Scheme
            </label>
            <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg">
              {(['tfidf', 'bow', 'binary'] as const).map(type => (
                <button
                  key={type}
                  onClick={() => onUpdateHyperparameters({ vectorizer: type })}
                  className={`flex-1 py-1 text-center font-medium rounded-md transition-colors ${
                    hyperparameters.vectorizer === type
                      ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs font-bold'
                      : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  {type.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* N-gram Support */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-700 dark:text-slate-300 block">
              N-Gram Vocabulary
            </label>
            <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg">
              <button
                onClick={() => onUpdateHyperparameters({ useBigrams: false })}
                className={`flex-1 py-1 text-center font-medium rounded-md transition-colors ${
                  !hyperparameters.useBigrams
                    ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs font-bold'
                    : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                Unigrams Only
              </button>
              <button
                onClick={() => onUpdateHyperparameters({ useBigrams: true })}
                className={`flex-1 py-1 text-center font-medium rounded-md transition-colors ${
                  hyperparameters.useBigrams
                    ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs font-bold'
                    : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                Unigrams + Bigrams
              </button>
            </div>
          </div>

          {/* Stopwords Filtering */}
          <div className="space-y-1.5">
            <label className="font-semibold text-slate-700 dark:text-slate-300 block">
              Stopwords Preprocessing
            </label>
            <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg">
              <button
                onClick={() => onUpdateHyperparameters({ removeStopwords: true })}
                className={`flex-1 py-1 text-center font-medium rounded-md transition-colors ${
                  hyperparameters.removeStopwords
                    ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs font-bold'
                    : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                Remove Stopwords
              </button>
              <button
                onClick={() => onUpdateHyperparameters({ removeStopwords: false })}
                className={`flex-1 py-1 text-center font-medium rounded-md transition-colors ${
                  !hyperparameters.removeStopwords
                    ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs font-bold'
                    : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
                }`}
              >
                Keep All Tokens
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Inspect Misclassified Sample Modal */}
      {inspectedSample && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-xl space-y-4 animate-scale-in">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-amber-500" />
                <h4 className="text-base font-bold text-slate-900 dark:text-white">
                  Misclassified Sample Inspection
                </h4>
              </div>
              <button
                onClick={() => setInspectedSample(null)}
                className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-white rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 bg-slate-50 dark:bg-slate-950/60 rounded-lg">
                  <span className="text-slate-400 block text-[11px]">GROUND TRUTH LABEL</span>
                  <span className="font-bold text-sm text-slate-900 dark:text-white uppercase">
                    {inspectedSample.sample.label}
                  </span>
                </div>
                <div className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-lg">
                  <span className="text-rose-500 block text-[11px]">MODEL PREDICTION</span>
                  <span className="font-bold text-sm text-rose-500 uppercase">
                    {inspectedSample.predicted} ({(inspectedSample.confidence * 100).toFixed(1)}% confidence)
                  </span>
                </div>
              </div>

              <div>
                <span className="text-slate-400 block mb-1">Subject:</span>
                <p className="font-semibold text-slate-900 dark:text-white p-2.5 bg-slate-50 dark:bg-slate-950/60 rounded-lg">
                  {inspectedSample.sample.subject}
                </p>
              </div>

              <div>
                <span className="text-slate-400 block mb-1">Body Text:</span>
                <p className="text-slate-700 dark:text-slate-300 p-3 bg-slate-50 dark:bg-slate-950/60 rounded-lg max-h-48 overflow-y-auto leading-relaxed">
                  {inspectedSample.sample.body}
                </p>
              </div>

              <div className="pt-2 text-slate-500 dark:text-slate-400 italic">
                Tip: Adjust Laplace smoothing α or SVM regularization C, or add more balanced training samples for this category to eliminate this edge case.
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <button
                onClick={() => setInspectedSample(null)}
                className="px-4 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 dark:bg-slate-800 dark:hover:bg-slate-700 rounded-lg transition-colors"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
