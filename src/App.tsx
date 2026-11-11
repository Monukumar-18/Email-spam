import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Header } from './components/Header';
import { PredictorView } from './components/PredictorView';
import { ModelLabView } from './components/ModelLabView';
import { DatasetView } from './components/DatasetView';
import { BatchTesterView } from './components/BatchTesterView';
import { AlgorithmGuideView } from './components/AlgorithmGuideView';
import { SpamClassifierManager } from './ml/modelManager';
import { loadDataset, resetDataset, saveDataset } from './ml/dataset';
import { EmailSample, Hyperparameters, TrainedModelsState } from './ml/types';

export default function App() {
  const [activeTab, setActiveTab] = useState<
    'predictor' | 'model_lab' | 'dataset' | 'batch_test' | 'algorithm_guide'
  >('predictor');

  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      return document.documentElement.classList.contains('dark') || true;
    }
    return true;
  });

  const [dataset, setDataset] = useState<EmailSample[]>(() => loadDataset());
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [modelState, setModelState] = useState<TrainedModelsState | null>(null);

  // Instantiate classifier manager
  const classifier = useMemo(() => new SpamClassifierManager(), []);

  // Sync dark class on html
  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDark]);

  const toggleTheme = useCallback(() => {
    setIsDark(prev => !prev);
  }, []);

  // Train models on mount and whenever dataset or hyperparameters update
  const executeTrain = useCallback(
    (currentDataset: EmailSample[]) => {
      setIsTraining(true);
      setTimeout(() => {
        try {
          const newState = classifier.trainAll(currentDataset);
          setModelState(newState);
        } catch (err) {
          console.error('Model training error:', err);
        } finally {
          setIsTraining(false);
        }
      }, 50);
    },
    [classifier]
  );

  // Initial train
  useEffect(() => {
    executeTrain(dataset);
  }, [executeTrain, dataset]);

  const handleRetrain = useCallback(() => {
    executeTrain(dataset);
  }, [executeTrain, dataset]);

  const handleAddEmail = useCallback(
    (newSample: Omit<EmailSample, 'id'>) => {
      const emailWithId: EmailSample = {
        ...newSample,
        id: `custom-${Date.now()}`,
      };
      const updated = [emailWithId, ...dataset];
      setDataset(updated);
      saveDataset(updated);
      executeTrain(updated);
    },
    [dataset, executeTrain]
  );

  const handleDeleteEmail = useCallback(
    (id: string) => {
      const updated = dataset.filter(d => d.id !== id);
      setDataset(updated);
      saveDataset(updated);
      executeTrain(updated);
    },
    [dataset, executeTrain]
  );

  const handleResetBenchmark = useCallback(() => {
    const fresh = resetDataset();
    setDataset(fresh);
    executeTrain(fresh);
  }, [executeTrain]);

  const handleImportDataset = useCallback(
    (imported: EmailSample[]) => {
      setDataset(imported);
      saveDataset(imported);
      executeTrain(imported);
    },
    [executeTrain]
  );

  const handleUpdateHyperparameters = useCallback(
    (newParams: Partial<Hyperparameters>) => {
      classifier.setHyperparameters(newParams);
      executeTrain(dataset);
    },
    [classifier, dataset, executeTrain]
  );

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans transition-colors duration-150">
      {/* Top Bar Contract compliant Header */}
      <Header
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        isTraining={isTraining}
        onRetrain={handleRetrain}
        isDark={isDark}
        onToggleTheme={toggleTheme}
        totalEmails={dataset.length}
      />

      {/* Main View Area */}
      <main className="flex-1">
        {activeTab === 'predictor' && (
          <PredictorView
            classifier={classifier}
            onAddEmailToDataset={handleAddEmail}
            onNavigateToModelLab={() => setActiveTab('model_lab')}
          />
        )}

        {activeTab === 'model_lab' && (
          <ModelLabView
            state={modelState}
            hyperparameters={classifier.getHyperparameters()}
            onUpdateHyperparameters={handleUpdateHyperparameters}
            onRetrain={handleRetrain}
            isTraining={isTraining}
          />
        )}

        {activeTab === 'dataset' && (
          <DatasetView
            dataset={dataset}
            onAddEmail={handleAddEmail}
            onDeleteEmail={handleDeleteEmail}
            onResetBenchmark={handleResetBenchmark}
            onImportDataset={handleImportDataset}
          />
        )}

        {activeTab === 'batch_test' && (
          <BatchTesterView classifier={classifier} dataset={dataset} />
        )}

        {activeTab === 'algorithm_guide' && <AlgorithmGuideView />}
      </main>

      {/* Quiet, Clean Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800/80 py-6 px-6 bg-white dark:bg-slate-900/60 text-xs text-slate-500 dark:text-slate-400">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-700 dark:text-slate-300">
              SpamGuard ML
            </span>
            <span aria-hidden="true">·</span>
            <span>Multinomial Naive Bayes & Linear Support Vector Machine Engine</span>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setActiveTab('algorithm_guide')}
              className="hover:text-slate-900 dark:hover:text-white transition-colors"
            >
              Theory & Derivations
            </button>
            <span aria-hidden="true">·</span>
            <button
              onClick={() => setActiveTab('model_lab')}
              className="hover:text-slate-900 dark:hover:text-white transition-colors"
            >
              Benchmark Metrics
            </button>
            <span aria-hidden="true">·</span>
            <span>In-Browser Client-Side ML</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
