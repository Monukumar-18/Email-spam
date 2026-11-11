import React, { useState, useMemo } from 'react';
import {
  Search,
  Filter,
  Plus,
  Trash2,
  Download,
  Upload,
  RotateCcw,
  Tag,
  CheckCircle2,
  AlertTriangle,
  X,
  FileText,
} from 'lucide-react';
import { EmailCategory, EmailLabel, EmailSample } from '../ml/types';

interface DatasetViewProps {
  dataset: EmailSample[];
  onAddEmail: (email: Omit<EmailSample, 'id'>) => void;
  onDeleteEmail: (id: string) => void;
  onResetBenchmark: () => void;
  onImportDataset: (emails: EmailSample[]) => void;
}

const CATEGORY_NAMES: Record<EmailCategory, string> = {
  phishing: 'Phishing & Credentials',
  lottery_prize: 'Lottery & Advance Fee',
  crypto_scam: 'Crypto / Fast Yield',
  pharma_counterfeit: 'Spam Products & Backlinks',
  urgency_extortion: 'Urgent Extortion',
  work_collaborative: 'Work Collaboration',
  personal_social: 'Personal & Social',
  transactional_receipt: 'Transactional Receipt',
  security_notification: 'Security Alerts',
  newsletter_update: 'Newsletters & Digests',
};

export const DatasetView: React.FC<DatasetViewProps> = ({
  dataset,
  onAddEmail,
  onDeleteEmail,
  onResetBenchmark,
  onImportDataset,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [labelFilter, setLabelFilter] = useState<'all' | 'spam' | 'ham'>('all');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);

  // New sample form state
  const [newSender, setNewSender] = useState('');
  const [newSubject, setNewSubject] = useState('');
  const [newBody, setNewBody] = useState('');
  const [newLabel, setNewLabel] = useState<EmailLabel>('spam');
  const [newCategory, setNewCategory] = useState<EmailCategory>('phishing');

  // Filtered dataset
  const filteredDataset = useMemo(() => {
    return dataset.filter(item => {
      if (labelFilter !== 'all' && item.label !== labelFilter) return false;
      if (selectedCategory !== 'all' && item.category !== selectedCategory) return false;
      if (searchTerm.trim()) {
        const q = searchTerm.toLowerCase();
        const matchesSubject = item.subject.toLowerCase().includes(q);
        const matchesBody = item.body.toLowerCase().includes(q);
        const matchesSender = item.sender.toLowerCase().includes(q);
        if (!matchesSubject && !matchesBody && !matchesSender) return false;
      }
      return true;
    });
  }, [dataset, labelFilter, selectedCategory, searchTerm]);

  const spamCount = dataset.filter(d => d.label === 'spam').length;
  const hamCount = dataset.filter(d => d.label === 'ham').length;

  const handleExportJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(dataset, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `spamguard_dataset_${dataset.length}_samples.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleExportCsv = () => {
    const headers = ['id', 'label', 'category', 'sender', 'subject', 'body'];
    const rows = dataset.map(d => [
      d.id,
      d.label,
      d.category,
      `"${d.sender.replace(/"/g, '""')}"`,
      `"${d.subject.replace(/"/g, '""')}"`,
      `"${d.body.replace(/"/g, '""')}"`,
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `spamguard_dataset_${dataset.length}_samples.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = event => {
      try {
        const text = event.target?.result as string;
        if (file.name.endsWith('.json')) {
          const parsed = JSON.parse(text);
          if (Array.isArray(parsed)) {
            onImportDataset(parsed);
          }
        } else if (file.name.endsWith('.csv')) {
          // Basic CSV parser
          const lines = text.split('\n').filter(Boolean);
          if (lines.length > 1) {
            const imported: EmailSample[] = [];
            for (let i = 1; i < lines.length; i++) {
              const line = lines[i];
              // Match CSV columns with quote support
              const regex = /(?:^|,)(?:"([^"]*(?:""[^"]*)*)"|([^,]*))/g;
              const matches: string[] = [];
              let match;
              while ((match = regex.exec(line)) !== null) {
                if (match[1] !== undefined) {
                  matches.push(match[1].replace(/""/g, '"'));
                } else {
                  matches.push(match[2] || '');
                }
              }
              if (matches.length >= 6) {
                imported.push({
                  id: `csv-${Date.now()}-${i}`,
                  label: matches[1].toLowerCase() === 'spam' ? 'spam' : 'ham',
                  category: (matches[2] as EmailCategory) || 'phishing',
                  sender: matches[3] || 'unknown@domain.com',
                  subject: matches[4] || '(No Subject)',
                  body: matches[5] || '',
                  isCustom: true,
                });
              }
            }
            if (imported.length > 0) {
              onImportDataset([...dataset, ...imported]);
            }
          }
        }
      } catch (err) {
        console.error('Failed to parse file', err);
      }
    };
    reader.readAsText(file);
  };

  const handleAddNewSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newSubject.trim() && !newBody.trim()) return;

    onAddEmail({
      sender: newSender.trim() || 'custom-sample@local.dev',
      subject: newSubject.trim() || '(No Subject)',
      body: newBody.trim(),
      label: newLabel,
      category: newCategory,
      date: new Date().toISOString().split('T')[0],
      isCustom: true,
    });

    setNewSender('');
    setNewSubject('');
    setNewBody('');
    setIsAddModalOpen(false);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header and Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Training Corpus & Benchmark Dataset
          </h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Curated repository of realistic emails used to train and test the Naive Bayes and SVM models.
          </p>
        </div>

        {/* Global Dataset Metrics (Zero-Pill formatting) */}
        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
          <span>Total: {dataset.length} samples</span>
          <span aria-hidden="true">·</span>
          <span className="text-rose-500 font-medium">Spam: {spamCount} ({((spamCount / (dataset.length || 1)) * 100).toFixed(1)}%)</span>
          <span aria-hidden="true">·</span>
          <span className="text-emerald-500 font-medium">Ham: {hamCount} ({((hamCount / (dataset.length || 1)) * 100).toFixed(1)}%)</span>
        </div>
      </div>

      {/* Action Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Search Input */}
        <div className="relative min-w-[280px] flex-1 max-w-md">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            placeholder="Search keywords, subjects, senders..."
            className="w-full pl-9 pr-3 py-1.5 text-xs bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white focus:outline-hidden focus:ring-2 focus:ring-rose-500/30"
          />
        </div>

        {/* Filters and Actions */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Label Filter segmented tabs */}
          <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg text-xs">
            <button
              onClick={() => setLabelFilter('all')}
              className={`px-2.5 py-1 font-medium rounded-md transition-colors ${
                labelFilter === 'all'
                  ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs font-semibold'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              All ({dataset.length})
            </button>
            <button
              onClick={() => setLabelFilter('spam')}
              className={`px-2.5 py-1 font-medium rounded-md transition-colors ${
                labelFilter === 'spam'
                  ? 'bg-white dark:bg-slate-900 text-rose-500 shadow-xs font-semibold'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Spam ({spamCount})
            </button>
            <button
              onClick={() => setLabelFilter('ham')}
              className={`px-2.5 py-1 font-medium rounded-md transition-colors ${
                labelFilter === 'ham'
                  ? 'bg-white dark:bg-slate-900 text-emerald-500 shadow-xs font-semibold'
                  : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Ham ({hamCount})
            </button>
          </div>

          {/* Add Sample Button */}
          <button
            onClick={() => setIsAddModalOpen(true)}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 dark:bg-rose-600 dark:hover:bg-rose-500 rounded-lg shadow-xs transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add Sample</span>
          </button>

          {/* Export / Import Dropdowns or Buttons */}
          <button
            onClick={handleExportCsv}
            title="Download Dataset as CSV"
            className="p-1.5 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-xs font-medium flex items-center gap-1"
          >
            <Download className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">CSV</span>
          </button>

          <label
            title="Upload CSV or JSON file"
            className="p-1.5 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-xs font-medium flex items-center gap-1 cursor-pointer"
          >
            <Upload className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Import</span>
            <input
              type="file"
              accept=".csv,.json"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>

          <button
            onClick={onResetBenchmark}
            title="Reset dataset back to original 70 benchmark emails"
            className="p-1.5 text-slate-500 hover:text-rose-500 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 text-xs transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Dataset Table */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-xs">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/60 text-slate-500 dark:text-slate-400 uppercase tracking-wider font-semibold">
                <th className="py-3 px-4 w-20">Class</th>
                <th className="py-3 px-4 w-44">Category</th>
                <th className="py-3 px-4 w-56">Sender</th>
                <th className="py-3 px-4 min-w-[240px]">Subject & Preview</th>
                <th className="py-3 px-4 w-16 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {filteredDataset.length > 0 ? (
                filteredDataset.map(item => (
                  <tr
                    key={item.id}
                    className="hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    {/* Label */}
                    <td className="py-3 px-4 whitespace-nowrap">
                      <span
                        className={`font-semibold uppercase tracking-wider font-mono text-[11px] ${
                          item.label === 'spam'
                            ? 'text-rose-600 dark:text-rose-400'
                            : 'text-emerald-600 dark:text-emerald-400'
                        }`}
                      >
                        {item.label}
                      </span>
                    </td>

                    {/* Category */}
                    <td className="py-3 px-4 text-slate-500 dark:text-slate-400 whitespace-nowrap">
                      {CATEGORY_NAMES[item.category] || item.category}
                    </td>

                    {/* Sender */}
                    <td className="py-3 px-4 font-mono text-slate-700 dark:text-slate-300 truncate max-w-[200px]">
                      {item.sender}
                    </td>

                    {/* Subject & Preview */}
                    <td className="py-3 px-4">
                      <div className="font-semibold text-slate-900 dark:text-white line-clamp-1">
                        {item.subject}
                      </div>
                      <div className="text-slate-500 dark:text-slate-400 line-clamp-1 text-[11px] mt-0.5 font-normal">
                        {item.body}
                      </div>
                    </td>

                    {/* Action */}
                    <td className="py-3 px-4 text-center whitespace-nowrap">
                      <button
                        onClick={() => onDeleteEmail(item.id)}
                        className="p-1 text-slate-400 hover:text-rose-500 transition-colors rounded"
                        title="Delete email sample"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="py-12 text-center text-slate-400">
                    No emails match your filter criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Sample Modal */}
      {isAddModalOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl max-w-xl w-full p-6 shadow-xl space-y-4 animate-scale-in">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800">
              <h3 className="text-base font-bold text-slate-900 dark:text-white">
                Add New Training Email Sample
              </h3>
              <button
                onClick={() => setIsAddModalOpen(false)}
                className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-white rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleAddNewSubmit} className="space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="font-medium text-slate-700 dark:text-slate-300 block mb-1">
                    Classification Ground Truth
                  </label>
                  <select
                    value={newLabel}
                    onChange={e => setNewLabel(e.target.value as EmailLabel)}
                    className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white"
                  >
                    <option value="spam">SPAM (Unwanted / Malicious)</option>
                    <option value="ham">HAM (Legitimate)</option>
                  </select>
                </div>

                <div>
                  <label className="font-medium text-slate-700 dark:text-slate-300 block mb-1">
                    Category Taxonomy
                  </label>
                  <select
                    value={newCategory}
                    onChange={e => setNewCategory(e.target.value as EmailCategory)}
                    className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white"
                  >
                    {Object.entries(CATEGORY_NAMES).map(([key, label]) => (
                      <option key={key} value={key}>
                        {label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="font-medium text-slate-700 dark:text-slate-300 block mb-1">
                  Sender Address
                </label>
                <input
                  type="text"
                  required
                  value={newSender}
                  onChange={e => setNewSender(e.target.value)}
                  placeholder="e.g. sender@example.com"
                  className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white"
                />
              </div>

              <div>
                <label className="font-medium text-slate-700 dark:text-slate-300 block mb-1">
                  Subject Line
                </label>
                <input
                  type="text"
                  required
                  value={newSubject}
                  onChange={e => setNewSubject(e.target.value)}
                  placeholder="Subject line of sample..."
                  className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white"
                />
              </div>

              <div>
                <label className="font-medium text-slate-700 dark:text-slate-300 block mb-1">
                  Email Body Text
                </label>
                <textarea
                  required
                  rows={6}
                  value={newBody}
                  onChange={e => setNewBody(e.target.value)}
                  placeholder="Type the full message body text..."
                  className="w-full px-3 py-2 bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white leading-relaxed resize-y"
                />
              </div>

              <div className="flex justify-end gap-2 pt-2 border-t border-slate-100 dark:border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsAddModalOpen(false)}
                  className="px-4 py-2 font-medium text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 font-semibold text-white bg-rose-600 hover:bg-rose-500 rounded-lg transition-colors"
                >
                  Save Sample
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
