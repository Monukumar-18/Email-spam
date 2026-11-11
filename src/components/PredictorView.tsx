import React, { useState, useMemo } from 'react';
import {
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  Sparkles,
  Flame,
  ShieldCheck,
  Send,
  PlusCircle,
  RotateCcw,
  Sliders,
  Info,
} from 'lucide-react';
import { PredictionResult, EmailLabel, EmailCategory, EmailSample } from '../ml/types';
import { SpamClassifierManager } from '../ml/modelManager';

interface PredictorViewProps {
  classifier: SpamClassifierManager;
  onAddEmailToDataset: (email: Omit<EmailSample, 'id'>) => void;
  onNavigateToModelLab: () => void;
}

const PRESET_EMAILS = [
  {
    name: 'Phishing (Bank Suspended)',
    sender: 'security@chase-urgent-verify.net',
    subject: 'URGENT: Unauthorized login detected - Account Suspended',
    body: 'Dear customer, We detected 3 unauthorized login attempts from a foreign IP address. Your checking account and online banking access have been suspended immediately. Click here to confirm your password, SSN, and card PIN within 12 hours or your account will be permanently closed.',
  },
  {
    name: 'Crypto Scam (Bitcoin AI)',
    sender: 'wealth@crypto-trader-signals.io',
    subject: 'Guaranteed 500% ROI in 7 Days with Automated Bitcoin Robot',
    body: 'Passive cryptocurrency income! Join 10,000+ investors making $5,000 daily profits. Deposit minimum 0.05 BTC today and watch your balance multiply. 100% risk free guaranteed payout. Limited slots remaining. Act now before enrollment closes!',
  },
  {
    name: 'Lottery Prize (Sweepstakes)',
    sender: 'payouts@international-lottery-board.org',
    subject: 'CONGRATULATIONS: You won $1,000,000 cash prize payout!',
    body: 'Winning notification! Your email was selected in our international lucky draw. To claim your $1,000,000 USD prize, reply with your full address, telephone number, and bank account details for immediate western union wire transfer.',
  },
  {
    name: 'Work Ham (Sprint Retro)',
    sender: 'david.miller@engineering-hub.com',
    subject: 'Sprint 42 Retrospective notes and action items',
    body: 'Hi everyone, thanks for joining the retro today. Main takeaways: we will increase code review coverage for backend APIs, Dave will handle database index optimization, and our next planning call is Monday at 10 AM. Let me know if you have feedback on the meeting notes.',
  },
  {
    name: 'Personal Ham (Dinner Plans)',
    sender: 'sarah.k@gmail.com',
    subject: 'Dinner this Friday downtown?',
    body: 'Hey! Hope your week is going smoothly. Are we still on for Italian dinner this Friday night? I checked that new pizza place near Pike Market and they have open reservations at 7 PM. Let me know if that works for you!',
  },
  {
    name: 'Transactional Ham (Flight Ticket)',
    sender: 'reservations@delta.com',
    subject: 'Your Delta flight confirmation: Seattle to Boston (DL 894)',
    body: 'Thank you for booking with Delta Air Lines. Confirmation code: G74K9P. Your flight departs Seattle-Tacoma on April 14 at 9:30 AM. Baggage allowance: 1 carry-on item. You may check in online 24 hours prior to departure.',
  },
];

export const PredictorView: React.FC<PredictorViewProps> = ({
  classifier,
  onAddEmailToDataset,
  onNavigateToModelLab,
}) => {
  const [sender, setSender] = useState(PRESET_EMAILS[0].sender);
  const [subject, setSubject] = useState(PRESET_EMAILS[0].subject);
  const [body, setBody] = useState(PRESET_EMAILS[0].body);
  const [selectedModel, setSelectedModel] = useState<'naive_bayes' | 'svm' | 'complement_nb' | 'compare'>(
    'naive_bayes'
  );
  const [hoveredToken, setHoveredToken] = useState<any | null>(null);
  const [saveSuccessMsg, setSaveSuccessMsg] = useState(false);

  // Compute prediction dynamically
  const fullText = `${subject} ${body}`;

  const primaryPrediction: PredictionResult | null = useMemo(() => {
    if (!fullText.trim()) return null;
    const modelToUse = selectedModel === 'compare' ? 'naive_bayes' : selectedModel;
    try {
      return classifier.predict(fullText, modelToUse);
    } catch {
      return null;
    }
  }, [classifier, fullText, selectedModel]);

  const comparisonPredictions = useMemo(() => {
    if (selectedModel !== 'compare' || !fullText.trim()) return null;
    try {
      const nb = classifier.predict(fullText, 'naive_bayes');
      const svm = classifier.predict(fullText, 'svm');
      const cnb = classifier.predict(fullText, 'complement_nb');
      return { nb, svm, cnb };
    } catch {
      return null;
    }
  }, [classifier, fullText, selectedModel]);

  const handleSelectPreset = (preset: typeof PRESET_EMAILS[0]) => {
    setSender(preset.sender);
    setSubject(preset.subject);
    setBody(preset.body);
  };

  const handleReset = () => {
    setSender('');
    setSubject('');
    setBody('');
  };

  const handleSaveToCorpus = (label: EmailLabel) => {
    if (!subject.trim() && !body.trim()) return;
    onAddEmailToDataset({
      sender: sender.trim() || 'user-input@local.dev',
      subject: subject.trim() || '(No Subject)',
      body: body.trim(),
      label,
      category: label === 'spam' ? 'phishing' : 'work_collaborative',
      date: new Date().toISOString().split('T')[0],
      isCustom: true,
    });
    setSaveSuccessMsg(true);
    setTimeout(() => setSaveSuccessMsg(false), 3000);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Page Header / Context */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Email Spam Prediction Playground
          </h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Real-time inference with Multinomial Naive Bayes, Linear Support Vector Machines, and XAI token attribution.
          </p>
        </div>

        {/* Model Mode Selector Tabs */}
        <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800/80 rounded-lg shrink-0">
          <button
            onClick={() => setSelectedModel('naive_bayes')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
              selectedModel === 'naive_bayes'
                ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Naive Bayes (MNB)
          </button>
          <button
            onClick={() => setSelectedModel('svm')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
              selectedModel === 'svm'
                ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Linear SVM
          </button>
          <button
            onClick={() => setSelectedModel('complement_nb')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
              selectedModel === 'complement_nb'
                ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Complement NB
          </button>
          <button
            onClick={() => setSelectedModel('compare')}
            className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${
              selectedModel === 'compare'
                ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
            }`}
          >
            Side-by-Side
          </button>
        </div>
      </div>

      {/* Preset Selector Bar */}
      <div className="space-y-2">
        <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
          Load Sample Scenarios
        </div>
        <div className="flex flex-wrap gap-2">
          {PRESET_EMAILS.map((preset, idx) => (
            <button
              key={idx}
              onClick={() => handleSelectPreset(preset)}
              className="px-3 py-1.5 text-xs font-medium bg-white dark:bg-slate-900 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-800 rounded-md transition-colors whitespace-nowrap"
            >
              {preset.name}
            </button>
          ))}
          <button
            onClick={handleReset}
            className="px-3 py-1.5 text-xs font-medium text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white rounded-md transition-colors inline-flex items-center gap-1.5"
          >
            <RotateCcw className="w-3 h-3" />
            Clear
          </button>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Email Input Editor (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 dark:border-slate-800/80">
              <span className="text-sm font-semibold text-slate-900 dark:text-white">
                Email Message Inspector
              </span>
              <span className="text-xs text-slate-400">
                Type or paste any email below
              </span>
            </div>

            {/* Sender */}
            <div className="space-y-1">
              <label className="text-xs font-medium text-slate-600 dark:text-slate-400">
                From / Sender Address
              </label>
              <input
                type="text"
                value={sender}
                onChange={e => setSender(e.target.value)}
                placeholder="e.g. notifications@service.com"
                className="w-full px-3 py-2 text-sm bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white focus:outline-hidden focus:ring-2 focus:ring-rose-500/30"
              />
            </div>

            {/* Subject */}
            <div className="space-y-1">
              <label className="text-xs font-medium text-slate-600 dark:text-slate-400">
                Subject Line
              </label>
              <input
                type="text"
                value={subject}
                onChange={e => setSubject(e.target.value)}
                placeholder="Email Subject..."
                className="w-full px-3 py-2 text-sm font-medium bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white focus:outline-hidden focus:ring-2 focus:ring-rose-500/30"
              />
            </div>

            {/* Body */}
            <div className="space-y-1">
              <label className="text-xs font-medium text-slate-600 dark:text-slate-400">
                Email Body Text
              </label>
              <textarea
                rows={9}
                value={body}
                onChange={e => setBody(e.target.value)}
                placeholder="Type or paste the full message body here..."
                className="w-full px-3 py-2 text-sm bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 rounded-lg text-slate-900 dark:text-white focus:outline-hidden focus:ring-2 focus:ring-rose-500/30 font-normal leading-relaxed resize-y"
              />
            </div>

            {/* Action Bar */}
            <div className="pt-2 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 dark:border-slate-800/80">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">Add to Training Set as:</span>
                <button
                  onClick={() => handleSaveToCorpus('spam')}
                  className="px-2.5 py-1 text-xs font-semibold text-rose-600 dark:text-rose-400 bg-rose-50 dark:bg-rose-950/40 hover:bg-rose-100 dark:hover:bg-rose-900/40 rounded-md border border-rose-200 dark:border-rose-900/50 transition-colors"
                >
                  + Spam
                </button>
                <button
                  onClick={() => handleSaveToCorpus('ham')}
                  className="px-2.5 py-1 text-xs font-semibold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/40 hover:bg-emerald-100 dark:hover:bg-emerald-900/40 rounded-md border border-emerald-200 dark:border-emerald-900/50 transition-colors"
                >
                  + Ham
                </button>
                {saveSuccessMsg && (
                  <span className="text-xs text-emerald-600 dark:text-emerald-400 font-medium animate-fade-in">
                    Saved to dataset!
                  </span>
                )}
              </div>

              <div className="text-xs text-slate-400 font-mono">
                {fullText.trim().split(/\s+/).filter(Boolean).length} words
              </div>
            </div>
          </div>

          {/* Explainable AI: Token Attribution Visualizer */}
          {primaryPrediction && primaryPrediction.tokens.length > 0 && (
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-slate-800">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-rose-500" />
                  <span className="text-sm font-semibold text-slate-900 dark:text-white">
                    Explainable AI (XAI): Token Attribution Map
                  </span>
                </div>
                <div className="text-xs text-slate-500 flex items-center gap-3">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-xs bg-rose-500/70 inline-block" /> Spam Indicator
                  </span>
                  <span className="inline-flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-xs bg-emerald-500/70 inline-block" /> Ham Indicator
                  </span>
                </div>
              </div>

              <p className="text-xs text-slate-500 dark:text-slate-400">
                Hover or click any token to inspect its log-odds statistical contribution to the classification decision.
              </p>

              {/* Token Flow Display */}
              <div className="p-4 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800 leading-loose text-sm min-h-[100px]">
                {primaryPrediction.tokens.map((tok, idx) => {
                  const isSpam = tok.rawWeight > 0.05;
                  const isHam = tok.rawWeight < -0.05;

                  let bgColor = 'bg-transparent text-slate-600 dark:text-slate-400';
                  if (isSpam) {
                    const intensity = Math.min(1, Math.abs(tok.spamAffinity));
                    bgColor =
                      intensity > 0.6
                        ? 'bg-rose-500/25 text-rose-900 dark:text-rose-200 font-medium'
                        : 'bg-rose-500/15 text-rose-800 dark:text-rose-300';
                  } else if (isHam) {
                    const intensity = Math.min(1, Math.abs(tok.spamAffinity));
                    bgColor =
                      intensity > 0.6
                        ? 'bg-emerald-500/25 text-emerald-900 dark:text-emerald-200 font-medium'
                        : 'bg-emerald-500/15 text-emerald-800 dark:text-emerald-300';
                  }

                  return (
                    <span
                      key={idx}
                      onMouseEnter={() => setHoveredToken(tok)}
                      onMouseLeave={() => setHoveredToken(null)}
                      className={`inline-block px-1.5 py-0.5 mx-0.5 rounded cursor-pointer transition-all hover:scale-105 hover:ring-1 hover:ring-slate-400 ${bgColor}`}
                    >
                      {tok.token}
                    </span>
                  );
                })}
              </div>

              {/* Hovered Token Inspection Card */}
              {hoveredToken ? (
                <div className="p-3 bg-slate-100 dark:bg-slate-800/90 rounded-lg border border-slate-300 dark:border-slate-700 flex flex-wrap items-center justify-between gap-4 text-xs">
                  <div>
                    <span className="font-mono font-bold text-sm text-slate-900 dark:text-white mr-2">
                      "{hoveredToken.token}"
                    </span>
                    <span className="text-slate-500 dark:text-slate-400">
                      {hoveredToken.inVocab ? 'Recognized Vocabulary Feature' : 'Out of Vocabulary / Stopword'}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 font-mono tabular-nums">
                    <div>
                      <span className="text-slate-400 mr-1">Weight:</span>
                      <span className={hoveredToken.rawWeight > 0 ? 'text-rose-500 font-bold' : 'text-emerald-500 font-bold'}>
                        {hoveredToken.rawWeight > 0 ? '+' : ''}{hoveredToken.rawWeight.toFixed(4)}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 mr-1">Count:</span>
                      <span className="text-slate-900 dark:text-white">{hoveredToken.count}</span>
                    </div>
                    <div>
                      <span className="text-slate-400 mr-1">Affinity:</span>
                      <span className={hoveredToken.spamAffinity > 0 ? 'text-rose-400' : 'text-emerald-400'}>
                        {hoveredToken.spamAffinity > 0 ? 'Spam' : 'Ham'} ({(Math.abs(hoveredToken.spamAffinity) * 100).toFixed(1)}%)
                      </span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-slate-400 italic">
                  Hover over words above to inspect individual feature impact.
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Prediction Results & Model Comparison (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Main Verdict Card */}
          {primaryPrediction && (
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs space-y-5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Model Verdict ({primaryPrediction.modelDisplayName})
                </span>
                <span className="text-xs font-mono text-slate-400">
                  Latency: {primaryPrediction.inferenceTimeMs.toFixed(2)}ms
                </span>
              </div>

              {/* Big Status Badge */}
              <div
                className={`p-5 rounded-xl border flex items-center justify-between ${
                  primaryPrediction.predictedLabel === 'spam'
                    ? 'bg-rose-500/10 border-rose-500/30 text-rose-500'
                    : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-500'
                }`}
              >
                <div className="flex items-center gap-3.5">
                  {primaryPrediction.predictedLabel === 'spam' ? (
                    <AlertTriangle className="w-8 h-8 text-rose-600 dark:text-rose-400 shrink-0" />
                  ) : (
                    <CheckCircle className="w-8 h-8 text-emerald-600 dark:text-emerald-400 shrink-0" />
                  )}
                  <div>
                    <h2 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
                      {primaryPrediction.predictedLabel === 'spam' ? 'Flagged as Spam' : 'Legitimate (Ham)'}
                    </h2>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                      {primaryPrediction.predictedLabel === 'spam'
                        ? 'High probability of unwanted, phishing, or scam content.'
                        : 'Matches normal personal, transactional, or workplace email.'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Probability Meter */}
              <div className="space-y-2">
                <div className="flex justify-between items-baseline text-xs font-mono">
                  <span className="text-emerald-600 dark:text-emerald-400 font-medium">
                    Ham {(primaryPrediction.hamProbability * 100).toFixed(1)}%
                  </span>
                  <span className="text-slate-500">
                    Decision Threshold: 50.0%
                  </span>
                  <span className="text-rose-600 dark:text-rose-400 font-bold">
                    Spam {(primaryPrediction.spamProbability * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="w-full h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden flex">
                  <div
                    style={{ width: `${primaryPrediction.hamProbability * 100}%` }}
                    className="bg-emerald-500 transition-all duration-300"
                  />
                  <div
                    style={{ width: `${primaryPrediction.spamProbability * 100}%` }}
                    className="bg-rose-500 transition-all duration-300"
                  />
                </div>
              </div>

              {/* Decision Function Raw Score */}
              <div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-100 dark:border-slate-800 text-xs">
                <div>
                  <span className="text-slate-400 block">Decision Score:</span>
                  <span className="font-mono font-semibold text-slate-900 dark:text-white tabular-nums">
                    {primaryPrediction.rawScore > 0 ? '+' : ''}{primaryPrediction.rawScore.toFixed(3)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400 block">Confidence Level:</span>
                  <span className="font-mono font-semibold text-slate-900 dark:text-white tabular-nums">
                    {(primaryPrediction.confidence * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Side-by-Side Model Comparison (when selected) */}
          {selectedModel === 'compare' && comparisonPredictions && (
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-4">
              <div className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Algorithm Consensus Comparison
              </div>

              <div className="space-y-3">
                {/* Multinomial Naive Bayes */}
                <div className="p-3 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-bold text-slate-900 dark:text-white block">
                      Multinomial Naive Bayes
                    </span>
                    <span className="text-[11px] text-slate-500">
                      Log-Odds Ratio: {comparisonPredictions.nb.rawScore.toFixed(2)}
                    </span>
                  </div>
                  <div className="text-right font-mono tabular-nums">
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded ${
                        comparisonPredictions.nb.predictedLabel === 'spam'
                          ? 'bg-rose-500/20 text-rose-500'
                          : 'bg-emerald-500/20 text-emerald-500'
                      }`}
                    >
                      {comparisonPredictions.nb.predictedLabel.toUpperCase()}
                    </span>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {(comparisonPredictions.nb.spamProbability * 100).toFixed(1)}% Spam
                    </div>
                  </div>
                </div>

                {/* Linear Support Vector Machine */}
                <div className="p-3 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-bold text-slate-900 dark:text-white block">
                      Linear Support Vector Machine
                    </span>
                    <span className="text-[11px] text-slate-500">
                      Hyperplane Margin: {comparisonPredictions.svm.rawScore.toFixed(3)}
                    </span>
                  </div>
                  <div className="text-right font-mono tabular-nums">
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded ${
                        comparisonPredictions.svm.predictedLabel === 'spam'
                          ? 'bg-rose-500/20 text-rose-500'
                          : 'bg-emerald-500/20 text-emerald-500'
                      }`}
                    >
                      {comparisonPredictions.svm.predictedLabel.toUpperCase()}
                    </span>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {(comparisonPredictions.svm.spamProbability * 100).toFixed(1)}% Spam
                    </div>
                  </div>
                </div>

                {/* Complement Naive Bayes */}
                <div className="p-3 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800 flex items-center justify-between">
                  <div>
                    <span className="text-xs font-bold text-slate-900 dark:text-white block">
                      Complement Naive Bayes
                    </span>
                    <span className="text-[11px] text-slate-500">
                      Inverse Class Log Score
                    </span>
                  </div>
                  <div className="text-right font-mono tabular-nums">
                    <span
                      className={`text-xs font-bold px-2 py-0.5 rounded ${
                        comparisonPredictions.cnb.predictedLabel === 'spam'
                          ? 'bg-rose-500/20 text-rose-500'
                          : 'bg-emerald-500/20 text-emerald-500'
                      }`}
                    >
                      {comparisonPredictions.cnb.predictedLabel.toUpperCase()}
                    </span>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      {(comparisonPredictions.cnb.spamProbability * 100).toFixed(1)}% Spam
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Top Contributing Features Bars */}
          {primaryPrediction && (
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-100 dark:border-slate-800">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                  Dominant Feature Drivers
                </span>
                <span className="text-[11px] text-slate-400">TF-IDF & Weights</span>
              </div>

              {/* Spam driving features */}
              <div className="space-y-2">
                <span className="text-xs font-medium text-rose-500 flex items-center gap-1">
                  <Flame className="w-3.5 h-3.5" /> Strongest Spam Signals
                </span>
                {primaryPrediction.topSpamTokens.length > 0 ? (
                  <div className="space-y-1.5">
                    {primaryPrediction.topSpamTokens.map((item, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-xs">
                        <span className="font-mono text-slate-800 dark:text-slate-200 w-28 truncate">
                          {item.token}
                        </span>
                        <div className="flex-1 bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                          <div
                            style={{ width: `${Math.min(100, item.weight * 60)}%` }}
                            className="bg-rose-500 h-full rounded-full"
                          />
                        </div>
                        <span className="font-mono text-rose-500 tabular-nums w-12 text-right">
                          +{item.weight.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-slate-400 italic">No significant spam keywords detected.</div>
                )}
              </div>

              {/* Ham driving features */}
              <div className="space-y-2 pt-2 border-t border-slate-100 dark:border-slate-800">
                <span className="text-xs font-medium text-emerald-500 flex items-center gap-1">
                  <ShieldCheck className="w-3.5 h-3.5" /> Strongest Ham Signals
                </span>
                {primaryPrediction.topHamTokens.length > 0 ? (
                  <div className="space-y-1.5">
                    {primaryPrediction.topHamTokens.map((item, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-xs">
                        <span className="font-mono text-slate-800 dark:text-slate-200 w-28 truncate">
                          {item.token}
                        </span>
                        <div className="flex-1 bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                          <div
                            style={{ width: `${Math.min(100, item.weight * 60)}%` }}
                            className="bg-emerald-500 h-full rounded-full"
                          />
                        </div>
                        <span className="font-mono text-emerald-500 tabular-nums w-12 text-right">
                          -{item.weight.toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-xs text-slate-400 italic">No strong legitimate anchor tokens detected.</div>
                )}
              </div>
            </div>
          )}

          {/* Structural & Heuristic Signals */}
          {primaryPrediction && (
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 shadow-xs space-y-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400">
                Structural & Syntactic Heuristics
              </span>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-2.5 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800">
                  <span className="text-slate-400 block text-[11px]">ALL CAPS RATIO</span>
                  <span className="font-mono font-bold text-sm text-slate-900 dark:text-white tabular-nums">
                    {(primaryPrediction.structuralSignals.capsRatio * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="p-2.5 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800">
                  <span className="text-slate-400 block text-[11px]">EXCLAMATION (!)</span>
                  <span className="font-mono font-bold text-sm text-slate-900 dark:text-white tabular-nums">
                    {primaryPrediction.structuralSignals.exclamationCount}
                  </span>
                </div>
                <div className="p-2.5 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800">
                  <span className="text-slate-400 block text-[11px]">CURRENCY SYMBOLS ($)</span>
                  <span className="font-mono font-bold text-sm text-slate-900 dark:text-white tabular-nums">
                    {primaryPrediction.structuralSignals.dollarCount}
                  </span>
                </div>
                <div className="p-2.5 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800">
                  <span className="text-slate-400 block text-[11px]">URGENT TRIGGERS</span>
                  <span className="font-mono font-bold text-sm text-slate-900 dark:text-white tabular-nums">
                    {primaryPrediction.structuralSignals.urgentKeywordCount}
                  </span>
                </div>
              </div>

              {primaryPrediction.structuralSignals.detectedUrgentKeywords.length > 0 && (
                <div className="text-xs text-rose-500 dark:text-rose-400 font-medium pt-1">
                  Urgency triggers detected:{' '}
                  {primaryPrediction.structuralSignals.detectedUrgentKeywords.join(', ')}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
