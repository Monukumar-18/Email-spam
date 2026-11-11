export type EmailLabel = 'spam' | 'ham';

export type EmailCategory =
  | 'phishing'
  | 'lottery_prize'
  | 'crypto_scam'
  | 'pharma_counterfeit'
  | 'urgency_extortion'
  | 'work_collaborative'
  | 'personal_social'
  | 'transactional_receipt'
  | 'security_notification'
  | 'newsletter_update';

export interface EmailSample {
  id: string;
  sender: string;
  subject: string;
  body: string;
  label: EmailLabel;
  category: EmailCategory;
  date?: string;
  isCustom?: boolean;
}

export type VectorizerType = 'tfidf' | 'bow' | 'binary';

export interface Hyperparameters {
  alpha: number; // Laplace smoothing for Naive Bayes (e.g. 1.0)
  svmC: number; // Regularization parameter C for SVM (e.g. 1.0)
  svmEpochs: number; // SGD epochs (e.g. 20)
  svmLearningRate: number; // Initial learning rate (e.g. 0.05)
  vectorizer: VectorizerType;
  useBigrams: boolean;
  removeStopwords: boolean;
  minWordFreq: number; // Minimum occurrences across corpus
  testSplitRatio: number; // 0.20 to 0.35
}

export interface TokenAttribution {
  token: string;
  count: number;
  inVocab: boolean;
  spamAffinity: number; // -1 (strong ham) to +1 (strong spam)
  logRatio: number; // Log odds ratio or SVM weight
  rawWeight: number;
}

export interface StructuralSignals {
  capsRatio: number;
  exclamationCount: number;
  dollarCount: number;
  suspiciousUrlCount: number;
  urgentKeywordCount: number;
  detectedUrgentKeywords: string[];
}

export interface PredictionResult {
  predictedLabel: EmailLabel;
  confidence: number; // Calibrated probability for predicted class (0.5 to 1.0)
  spamProbability: number; // Exact P(Spam) (0.0 to 1.0)
  hamProbability: number; // Exact P(Ham) (0.0 to 1.0)
  rawScore: number; // Decision function score (log-odds or hyperplane distance)
  modelType: 'naive_bayes' | 'svm' | 'complement_nb';
  modelDisplayName: string;
  tokens: TokenAttribution[];
  topSpamTokens: Array<{ token: string; weight: number }>;
  topHamTokens: Array<{ token: string; weight: number }>;
  structuralSignals: StructuralSignals;
  inferenceTimeMs: number;
}

export interface ConfusionMatrix {
  tp: number; // True Positives (Spam predicted as Spam)
  fp: number; // False Positives (Ham predicted as Spam)
  tn: number; // True Negatives (Ham predicted as Ham)
  fn: number; // False Negatives (Spam predicted as Ham)
}

export interface RocPoint {
  fpr: number; // False Positive Rate (1 - Specificity)
  tpr: number; // True Positive Rate (Recall / Sensitivity)
  threshold: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  specificity: number;
  auc: number;
  confusionMatrix: ConfusionMatrix;
  rocCurve: RocPoint[];
  trainCount: number;
  testCount: number;
  trainTimeMs: number;
  misclassifiedSamples: Array<{
    sample: EmailSample;
    predicted: EmailLabel;
    confidence: number;
    rawScore: number;
  }>;
}

export interface TrainedModelsState {
  naiveBayesMetrics: ModelMetrics;
  svmMetrics: ModelMetrics;
  complementNbMetrics: ModelMetrics;
  vocabularySize: number;
  totalCorpusSize: number;
  trainSize: number;
  testSize: number;
  topSpamVocab: Array<{ word: string; score: number }>;
  topHamVocab: Array<{ word: string; score: number }>;
  hyperparameters: Hyperparameters;
  lastTrainedAt: number;
}
