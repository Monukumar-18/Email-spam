import React from 'react';
import { ShieldCheck, Moon, Sun, RefreshCw, Cpu } from 'lucide-react';

interface HeaderProps {
  activeTab: 'predictor' | 'model_lab' | 'dataset' | 'batch_test' | 'algorithm_guide';
  onSelectTab: (tab: 'predictor' | 'model_lab' | 'dataset' | 'batch_test' | 'algorithm_guide') => void;
  isTraining: boolean;
  onRetrain: () => void;
  isDark: boolean;
  onToggleTheme: () => void;
  totalEmails: number;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onSelectTab,
  isTraining,
  onRetrain,
  isDark,
  onToggleTheme,
  totalEmails,
}) => {
  return (
    <header className="flex items-center justify-between gap-8 px-6 py-3.5 border-b border-slate-200 dark:border-slate-800/80 bg-white dark:bg-slate-900/90 backdrop-blur-md sticky top-0 z-40">
      {/* Zone 1: Single text element wordmark */}
      <div className="flex items-center gap-2.5 shrink-0">
        <div className="w-8 h-8 rounded-lg bg-rose-500/10 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400 flex items-center justify-center border border-rose-500/20">
          <ShieldCheck className="w-4 h-4" />
        </div>
        <div className="flex items-baseline gap-2">
          <span className="text-base font-bold tracking-tight text-slate-900 dark:text-white whitespace-nowrap">
            SpamGuard ML
          </span>
          <span className="text-xs text-slate-500 dark:text-slate-400 font-normal hidden sm:inline">
            Naive Bayes & SVM Engine
          </span>
        </div>
      </div>

      {/* Zone 2: 4-5 concise single-line text navigation links */}
      <nav className="hidden lg:flex items-center gap-6 text-sm font-medium text-slate-600 dark:text-slate-400">
        <button
          onClick={() => onSelectTab('predictor')}
          className={`hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0 pb-0.5 border-b-2 ${
            activeTab === 'predictor'
              ? 'border-rose-500 text-slate-900 dark:text-white font-semibold'
              : 'border-transparent'
          }`}
        >
          Classifier Playground
        </button>
        <button
          onClick={() => onSelectTab('model_lab')}
          className={`hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0 pb-0.5 border-b-2 ${
            activeTab === 'model_lab'
              ? 'border-rose-500 text-slate-900 dark:text-white font-semibold'
              : 'border-transparent'
          }`}
        >
          Model Lab & Metrics
        </button>
        <button
          onClick={() => onSelectTab('dataset')}
          className={`hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0 pb-0.5 border-b-2 ${
            activeTab === 'dataset'
              ? 'border-rose-500 text-slate-900 dark:text-white font-semibold'
              : 'border-transparent'
          }`}
        >
          Corpus & Dataset ({totalEmails})
        </button>
        <button
          onClick={() => onSelectTab('batch_test')}
          className={`hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0 pb-0.5 border-b-2 ${
            activeTab === 'batch_test'
              ? 'border-rose-500 text-slate-900 dark:text-white font-semibold'
              : 'border-transparent'
          }`}
        >
          Batch Evaluator
        </button>
        <button
          onClick={() => onSelectTab('algorithm_guide')}
          className={`hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0 pb-0.5 border-b-2 ${
            activeTab === 'algorithm_guide'
              ? 'border-rose-500 text-slate-900 dark:text-white font-semibold'
              : 'border-transparent'
          }`}
        >
          Math & Theory
        </button>
      </nav>

      {/* Zone 3: 1 primary action + subtle theme toggle */}
      <div className="flex items-center gap-3 shrink-0">
        <button
          onClick={onToggleTheme}
          aria-label="Toggle color theme"
          className="p-2 text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
        >
          {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        <button
          onClick={onRetrain}
          disabled={isTraining}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 dark:bg-rose-600 dark:hover:bg-rose-500 rounded-lg shadow-sm transition-all whitespace-nowrap shrink-0 disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isTraining ? 'animate-spin' : ''}`} />
          <span>{isTraining ? 'Training...' : 'Retrain Models'}</span>
        </button>
      </div>
    </header>
  );
};
