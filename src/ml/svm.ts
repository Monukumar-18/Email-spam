import { EmailLabel, TokenAttribution } from './types';
import { dotProduct, SparseVector, VectorizerModel } from './vectorizer';

export interface LinearSvmModel {
  weights: number[]; // featureIndex -> weight w_j
  bias: number; // bias b
  c: number; // regularization parameter
  epochs: number;
  plattScale: { a: number; b: number }; // Platt calibration parameters
  vocabularySize: number;
}

export function trainLinearSvm(
  trainingVectors: SparseVector[],
  trainingLabels: EmailLabel[],
  vocabularySize: number,
  cParam: number = 1.0,
  epochs: number = 20,
  initialLearningRate: number = 0.05
): LinearSvmModel {
  const numSamples = trainingVectors.length;
  const weights = new Array<number>(vocabularySize).fill(0);
  let bias = 0;

  // Convert labels to binary {-1, +1}
  // +1 = spam, -1 = ham
  const y = trainingLabels.map(l => (l === 'spam' ? 1 : -1));

  // Shuffled index order for SGD
  const indices = Array.from({ length: numSamples }, (_, i) => i);

  // Subgradient SGD with L2 regularization
  // Minimize (1/2) * ||w||^2 + C * sum(hinge_loss)
  const lambda = 1.0 / (cParam * numSamples);

  for (let epoch = 1; epoch <= epochs; epoch++) {
    // Learning rate with scheduled decay
    const lr = initialLearningRate / (1 + 0.05 * epoch);

    // Shuffle samples
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    for (const i of indices) {
      const vec = trainingVectors[i];
      const target = y[i];

      // Compute decision value: w^T x + b
      const margin = dotProduct(vec, weights) + bias;

      // Hinge loss condition: target * (w^T x + b) < 1
      if (target * margin < 1.0) {
        // Gradient update with misclassification / margin violation penalty:
        // w <- w - lr * (lambda * w - target * x)
        // b <- b + lr * target * 0.1
        const regScale = 1.0 - lr * lambda;

        // Regularize all weights touched or slightly decay
        for (let k = 0; k < vec.indices.length; k++) {
          const idx = vec.indices[k];
          const xVal = vec.values[k];
          weights[idx] = weights[idx] * regScale + lr * target * xVal;
        }

        bias += lr * target * 0.1;
      } else {
        // Gradient only from regularization term: w <- (1 - lr * lambda) * w
        for (let k = 0; k < vec.indices.length; k++) {
          const idx = vec.indices[k];
          weights[idx] *= 1.0 - lr * lambda;
        }
      }
    }
  }

  // Calibrate Platt scaling parameters (A, B) on training margins
  // f(x) -> P(y = 1 | x) = 1 / (1 + exp(A * f(x) + B))
  // A standard initialization for linear margins:
  let meanPosMargin = 0;
  let countPos = 0;
  let meanNegMargin = 0;
  let countNeg = 0;

  for (let i = 0; i < numSamples; i++) {
    const margin = dotProduct(trainingVectors[i], weights) + bias;
    if (y[i] === 1) {
      meanPosMargin += margin;
      countPos++;
    } else {
      meanNegMargin += margin;
      countNeg++;
    }
  }

  const avgPos = countPos > 0 ? meanPosMargin / countPos : 1.0;
  const avgNeg = countNeg > 0 ? meanNegMargin / countNeg : -1.0;
  const separation = Math.max(0.5, avgPos - avgNeg);
  const aParam = -2.5 / separation; // Negative scale so positive margin gives high spam prob
  const bParam = -aParam * ((avgPos + avgNeg) / 2);

  return {
    weights,
    bias,
    c: cParam,
    epochs,
    plattScale: { a: aParam, b: bParam },
    vocabularySize,
  };
}

export function predictLinearSvm(
  vector: SparseVector,
  model: LinearSvmModel,
  tokens: string[],
  vectorizer: VectorizerModel
): {
  predictedLabel: EmailLabel;
  spamProbability: number;
  hamProbability: number;
  confidence: number;
  rawScore: number;
  tokenAttributions: TokenAttribution[];
} {
  // Margin distance: f(x) = w^T x + b
  const margin = dotProduct(vector, model.weights) + model.bias;

  // Platt scaling probability calibration
  const exponent = model.plattScale.a * margin + model.plattScale.b;
  let spamProb: number;
  if (exponent > 35) {
    spamProb = 0.0001;
  } else if (exponent < -35) {
    spamProb = 0.9999;
  } else {
    spamProb = 1.0 / (1.0 + Math.exp(exponent));
  }
  const hamProb = 1.0 - spamProb;

  // Binary decision boundary at hyperplane margin = 0 (or spamProb >= 0.5)
  const predictedLabel: EmailLabel = margin >= 0 ? 'spam' : 'ham';
  const confidence = predictedLabel === 'spam' ? spamProb : hamProb;

  // Build token-level attribution using SVM feature weights w_j
  const tokenMap = new Map<string, number>();
  for (const t of tokens) {
    tokenMap.set(t, (tokenMap.get(t) || 0) + 1);
  }

  const tokenAttributions: TokenAttribution[] = [];
  for (const [token, count] of tokenMap.entries()) {
    const idx = vectorizer.vocabulary.get(token);
    if (idx !== undefined) {
      const weight = model.weights[idx] || 0;
      // Affinity bounded between -1 and +1
      const affinity = Math.tanh(weight * 2.0);

      tokenAttributions.push({
        token,
        count,
        inVocab: true,
        spamAffinity: affinity,
        logRatio: weight,
        rawWeight: weight * count,
      });
    } else {
      tokenAttributions.push({
        token,
        count,
        inVocab: false,
        spamAffinity: 0,
        logRatio: 0,
        rawWeight: 0,
      });
    }
  }

  return {
    predictedLabel,
    spamProbability: spamProb,
    hamProbability: hamProb,
    confidence,
    rawScore: margin,
    tokenAttributions,
  };
}
