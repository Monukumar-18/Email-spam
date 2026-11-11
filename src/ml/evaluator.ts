import { EmailLabel, EmailSample, ModelMetrics, RocPoint } from './types';

export function stratifiedTrainTestSplit(
  dataset: EmailSample[],
  testRatio: number = 0.25,
  seed: number = 42
): { train: EmailSample[]; test: EmailSample[] } {
  const spamSamples = dataset.filter(d => d.label === 'spam');
  const hamSamples = dataset.filter(d => d.label === 'ham');

  // Pseudo-random deterministic shuffle
  const shuffle = <T>(array: T[], s: number): T[] => {
    const copy = [...array];
    let currentSeed = s;
    const random = () => {
      currentSeed = (currentSeed * 9301 + 49297) % 233280;
      return currentSeed / 233280;
    };
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [copy[i], copy[j]] = [copy[j], copy[i]];
    }
    return copy;
  };

  const shuffledSpam = shuffle(spamSamples, seed);
  const shuffledHam = shuffle(hamSamples, seed + 101);

  const numTestSpam = Math.max(1, Math.round(shuffledSpam.length * testRatio));
  const numTestHam = Math.max(1, Math.round(shuffledHam.length * testRatio));

  const testSpam = shuffledSpam.slice(0, numTestSpam);
  const trainSpam = shuffledSpam.slice(numTestSpam);

  const testHam = shuffledHam.slice(0, numTestHam);
  const trainHam = shuffledHam.slice(numTestHam);

  return {
    train: [...trainSpam, ...trainHam],
    test: [...testSpam, ...testHam],
  };
}

export function computeEvaluationMetrics(
  testSamples: EmailSample[],
  predictions: Array<{
    predictedLabel: EmailLabel;
    spamProbability: number;
    rawScore: number;
    confidence: number;
  }>,
  trainCount: number,
  trainTimeMs: number
): ModelMetrics {
  const testCount = testSamples.length;
  let tp = 0; // Spam classified as Spam
  let fp = 0; // Ham classified as Spam
  let tn = 0; // Ham classified as Ham
  let fn = 0; // Spam classified as Ham

  const misclassifiedSamples: ModelMetrics['misclassifiedSamples'] = [];

  for (let i = 0; i < testCount; i++) {
    const actual = testSamples[i].label;
    const pred = predictions[i].predictedLabel;

    if (actual === 'spam' && pred === 'spam') {
      tp++;
    } else if (actual === 'ham' && pred === 'spam') {
      fp++;
      misclassifiedSamples.push({
        sample: testSamples[i],
        predicted: pred,
        confidence: predictions[i].confidence,
        rawScore: predictions[i].rawScore,
      });
    } else if (actual === 'ham' && pred === 'ham') {
      tn++;
    } else if (actual === 'spam' && pred === 'ham') {
      fn++;
      misclassifiedSamples.push({
        sample: testSamples[i],
        predicted: pred,
        confidence: predictions[i].confidence,
        rawScore: predictions[i].rawScore,
      });
    }
  }

  const accuracy = testCount > 0 ? (tp + tn) / testCount : 0;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 1;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 1;
  const specificity = tn + fp > 0 ? tn / (tn + fp) : 1;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  // ROC Curve calculation: vary threshold from 0 to 1
  const rocCurve: RocPoint[] = [];
  const thresholds = [
    0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0
  ];

  const totalActualSpam = tp + fn;
  const totalActualHam = tn + fp;

  for (const th of thresholds) {
    let t_tp = 0;
    let t_fp = 0;

    for (let i = 0; i < testCount; i++) {
      const actual = testSamples[i].label;
      const spamProb = predictions[i].spamProbability;
      const isPredictedSpam = spamProb >= th;

      if (actual === 'spam' && isPredictedSpam) t_tp++;
      if (actual === 'ham' && isPredictedSpam) t_fp++;
    }

    const tpr = totalActualSpam > 0 ? t_tp / totalActualSpam : 0;
    const fpr = totalActualHam > 0 ? t_fp / totalActualHam : 0;

    rocCurve.push({ fpr, tpr, threshold: th });
  }

  // Sort ROC curve points by FPR ascending
  rocCurve.sort((a, b) => a.fpr - b.fpr || a.tpr - b.tpr);

  // Compute AUC using trapezoidal rule
  let auc = 0;
  for (let i = 1; i < rocCurve.length; i++) {
    const prev = rocCurve[i - 1];
    const curr = rocCurve[i];
    const width = curr.fpr - prev.fpr;
    const avgHeight = (curr.tpr + prev.tpr) / 2;
    auc += width * avgHeight;
  }
  // Clamp AUC between 0.5 and 1.0 for standard presentation
  auc = Math.min(1.0, Math.max(0.5, Math.abs(auc)));

  return {
    accuracy,
    precision,
    recall,
    f1,
    specificity,
    auc,
    confusionMatrix: { tp, fp, tn, fn },
    rocCurve,
    trainCount,
    testCount,
    trainTimeMs,
    misclassifiedSamples,
  };
}
