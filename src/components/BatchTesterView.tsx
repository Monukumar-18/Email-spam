import React, { useState } from 'react';
import { Layers, Play, Download, CheckCircle, AlertTriangle, FileSpreadsheet } from 'lucide-react';
import { SpamClassifierManager } from '../ml/modelManager';
import { EmailLabel, EmailSample } from '../ml/types';

interface BatchTesterViewProps {
  classifier: SpamClassifierManager;
  dataset: EmailSample[];
}

interface BatchResult {
  id: number;
  snippet: string;
  predicted: EmailLabel;
  confidence: number;
  spamProbability: number;
  timeMs: number;
}

const SAMPLE_BATCH_INPUT = `1. URGENT: Your Apple iCloud ID is locked due to suspicious billing charges. Click here to confirm debit card.
2. Team standup tomorrow moved to 10:30 AM in Conference Room 4B to accommodate external client call.
3. Win $5,000,000 cash in UK Euro Millions sweepstakes! Reply with bank account and wire transfer details.
4. Your Amazon delivery #482910 has arrived at your front porch. Review your delivery photo.
5. Passive crypto income alert! Deposit 0.1 ETH and get 300% guaranteed returns within 24 hours. No risk!
6. Hey mom, can you send me grandma's banana bread recipe when you get a chance? Love you!
7. Final notice: Court citation subpoena for civil litigation attached in password-protected ZIP archive.
8. PR #294 merged: Optimized database indexing queries for Postgres user authentication sessions.`;

export const BatchTesterView: React.FC<BatchTesterViewProps> = ({ classifier, dataset }) => {
  const [inputText, setInputText] = useState(SAMPLE_BATCH_INPUT);
  const [selectedModel, setSelectedModel] = useState<'naive_bayes' | 'svm' | 'complement_nb'>('naive_bayes');
  const [results, setResults] = useState<BatchResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const handleRunBatch = () => {
    setIsRunning(true);
    setTimeout(() => {
      const lines = inputText
        .split('\n')
        .map(l => l.trim())
        .filter(l => l.length > 5);

      const batchOut: BatchResult[] = lines.map((line, idx) => {
        const pred = classifier.predict(line, selectedModel);
        return {
          id: idx + 1,
          snippet: line,
          predicted: pred.predictedLabel,
          confidence: pred.confidence,
          spamProbability: pred.spamProbability,
          timeMs: pred.inferenceTimeMs,
        };
      });

      setResults(batchOut);
      setIsRunning(false);
    }, 100);
  };

  const handleExportCsv = () => {
    if (results.length === 0) return;
    const headers = ['Index', 'Email Snippet', 'Predicted Label', 'Spam Probability', 'Confidence'];
    const rows = results.map(r => [
      r.id,
      `"${r.snippet.replace(/"/g, '""')}"`,
      r.predicted.toUpperCase(),
      (r.spamProbability * 100).toFixed(2) + '%',
      (r.confidence * 100).toFixed(2) + '%',
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `batch_predictions_${selectedModel}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const spamCount = results.filter(r => r.predicted === 'spam').length;
  const hamCount = results.filter(r => r.predicted === 'ham').length;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            High-Throughput Batch Classifier
          </h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Paste multiple email messages or subjects line-by-line to execute bulk spam classification.
          </p>
        </div>

        {/* Model Selector Tabs */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg text-xs">
          <button
            onClick={() => setSelectedModel('naive_bayes')}
            className={`px-3 py-1.5 font-semibold rounded-md transition-colors ${
              selectedModel === 'naive_bayes'
                ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Naive Bayes (MNB)
          </button>
          <button
            onClick={() => setSelectedModel('svm')}
            className={`px-3 py-1.5 font-semibold rounded-md transition-colors ${
              selectedModel === 'svm'
                ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Linear SVM
          </button>
          <button
            onClick={() => setSelectedModel('complement_nb')}
            className={`px-3 py-1.5 font-semibold rounded-md transition-colors ${
              selectedModel === 'complement_nb'
                ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Complement NB
          </button>
        </div>
      </div>

      {/* Input Box and Actions */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-700 dark:text-slate-300">
            Batch Input (One email or subject per line)
          </label>
          <span className="text-xs text-slate-400 font-mono">
            {inputText.split('\n').filter(l => l.trim().length > 5).length} lines ready
          </span>
        </div>

        <textarea
          rows={7}
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          placeholder="Paste email snippets here (one per line)..."
          className="w-full px-3 py-2 text-xs font-mono bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white leading-relaxed resize-y"
        />

        <div className="flex items-center justify-between pt-2">
          <button
            onClick={() => setInputText('')}
            className="text-xs text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
          >
            Clear Text
          </button>

          <button
            onClick={handleRunBatch}
            disabled={isRunning || !inputText.trim()}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 dark:bg-rose-600 dark:hover:bg-rose-500 rounded-lg shadow-sm transition-colors disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5" />
            <span>{isRunning ? 'Processing Batch...' : 'Run Batch Classification'}</span>
          </button>
        </div>
      </div>

      {/* Batch Results Output */}
      {results.length > 0 && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-xs space-y-4">
          <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <FileSpreadsheet className="w-4 h-4 text-rose-500" />
              <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
                Batch Inference Report ({results.length} emails evaluated)
              </h2>
            </div>

            <div className="flex items-center gap-4 text-xs font-mono tabular-nums">
              <span className="text-rose-500 font-bold">
                Spam: {spamCount} ({((spamCount / results.length) * 100).toFixed(0)}%)
              </span>
              <span aria-hidden="true" className="text-slate-400">·</span>
              <span className="text-emerald-500 font-bold">
                Ham: {hamCount} ({((hamCount / results.length) * 100).toFixed(0)}%)
              </span>
              <button
                onClick={handleExportCsv}
                className="inline-flex items-center gap-1.5 px-3 py-1 text-xs font-medium bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-900 dark:text-white rounded-md transition-colors"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export CSV</span>
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 text-slate-500 dark:text-slate-400 uppercase tracking-wider font-semibold">
                  <th className="py-2.5 px-4 w-12 text-center">#</th>
                  <th className="py-2.5 px-4 w-28">Prediction</th>
                  <th className="py-2.5 px-4 w-36 text-right">Spam Prob</th>
                  <th className="py-2.5 px-4">Email Snippet</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {results.map(r => (
                  <tr
                    key={r.id}
                    className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    <td className="py-2.5 px-4 text-center font-mono text-slate-400">
                      {r.id}
                    </td>
                    <td className="py-2.5 px-4 whitespace-nowrap">
                      <span
                        className={`font-semibold uppercase tracking-wider font-mono text-[11px] ${
                          r.predicted === 'spam'
                            ? 'text-rose-600 dark:text-rose-400'
                            : 'text-emerald-600 dark:text-emerald-400'
                        }`}
                      >
                        {r.predicted}
                      </span>
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                      <span
                        className={r.spamProbability > 0.5 ? 'text-rose-500 font-bold' : 'text-emerald-500'}
                      >
                        {(r.spamProbability * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-2.5 px-4 text-slate-800 dark:text-slate-200 font-sans">
                      {r.snippet}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
