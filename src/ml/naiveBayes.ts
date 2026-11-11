import { EmailLabel, TokenAttribution } from './types';
import { SparseVector, VectorizerModel } from './vectorizer';

export interface NaiveBayesModel {
  type: 'naive_bayes' | 'complement_nb';
  vocabularySize: number;
  classPriors: { spam: number; ham: number }; // P(y)
  logClassPriors: { spam: number; ham: number }; // ln P(y)
  // Conditional log probabilities: ln P(w_k | y)
  spamFeatureLogProbs: number[]; // index -> ln P(w | spam)
  hamFeatureLogProbs: number[]; // index -> ln P(w | ham)
  // CNB weights: index -> weight
  cnbWeights?: number[];
  alpha: number;
}

export function trainMultinomialNaiveBayes(
  trainingVectors: SparseVector[],
  trainingLabels: EmailLabel[],
  vocabularySize: number,
  alpha: number = 1.0
): NaiveBayesModel {
  const numSamples = trainingVectors.length;
  let spamCount = 0;
  let hamCount = 0;

  const spamTokenSums = new Array<number>(vocabularySize).fill(0);
  const hamTokenSums = new Array<number>(vocabularySize).fill(0);

  let totalSpamTokens = 0;
  let totalHamTokens = 0;

  for (let i = 0; i < numSamples; i++) {
    const vec = trainingVectors[i];
    const isSpam = trainingLabels[i] === 'spam';

    if (isSpam) {
      spamCount++;
      for (let j = 0; j < vec.indices.length; j++) {
        const idx = vec.indices[j];
        const val = vec.values[j];
        spamTokenSums[idx] += val;
        totalSpamTokens += val;
      }
    } else {
      hamCount++;
      for (let j = 0; j < vec.indices.length; j++) {
        const idx = vec.indices[j];
        const val = vec.values[j];
        hamTokenSums[idx] += val;
        totalHamTokens += val;
      }
    }
  }

  // Calculate priors with Laplace smoothing on class counts
  const spamPrior = (spamCount + 1) / (numSamples + 2);
  const hamPrior = (hamCount + 1) / (numSamples + 2);

  const logSpamPrior = Math.log(spamPrior);
  const logHamPrior = Math.log(hamPrior);

  // Compute feature log-probabilities with Laplace smoothing alpha
  // P(w | y) = (sum(x_w) + alpha) / (total_tokens_in_y + alpha * |V|)
  const spamDenominator = totalSpamTokens + alpha * vocabularySize;
  const hamDenominator = totalHamTokens + alpha * vocabularySize;

  const spamFeatureLogProbs = new Array<number>(vocabularySize);
  const hamFeatureLogProbs = new Array<number>(vocabularySize);

  for (let i = 0; i < vocabularySize; i++) {
    spamFeatureLogProbs[i] = Math.log((spamTokenSums[i] + alpha) / spamDenominator);
    hamFeatureLogProbs[i] = Math.log((hamTokenSums[i] + alpha) / hamDenominator);
  }

  return {
    type: 'naive_bayes',
    vocabularySize,
    classPriors: { spam: spamPrior, ham: hamPrior },
    logClassPriors: { spam: logSpamPrior, ham: logHamPrior },
    spamFeatureLogProbs,
    hamFeatureLogProbs,
    alpha,
  };
}

export function trainComplementNaiveBayes(
  trainingVectors: SparseVector[],
  trainingLabels: EmailLabel[],
  vocabularySize: number,
  alpha: number = 1.0
): NaiveBayesModel {
  // Complement Naive Bayes estimates weights from the complement of each class
  // Ideal for skewed / imbalanced text corpora
  const numSamples = trainingVectors.length;
  let spamCount = 0;
  let hamCount = 0;

  // We accumulate complement sums: for spam, complement is ham tokens; for ham, complement is spam tokens
  const notSpamSums = new Array<number>(vocabularySize).fill(0); // Actually Ham tokens
  const notHamSums = new Array<number>(vocabularySize).fill(0); // Actually Spam tokens
  let totalNotSpam = 0;
  let totalNotHam = 0;

  for (let i = 0; i < numSamples; i++) {
    const vec = trainingVectors[i];
    const isSpam = trainingLabels[i] === 'spam';

    if (isSpam) {
      spamCount++;
      for (let j = 0; j < vec.indices.length; j++) {
        const idx = vec.indices[j];
        const val = vec.values[j];
        notHamSums[idx] += val;
        totalNotHam += val;
      }
    } else {
      hamCount++;
      for (let j = 0; j < vec.indices.length; j++) {
        const idx = vec.indices[j];
        const val = vec.values[j];
        notSpamSums[idx] += val;
        totalNotSpam += val;
      }
    }
  }

  const spamDenominator = totalNotSpam + alpha * vocabularySize;
  const hamDenominator = totalNotHam + alpha * vocabularySize;

  const spamFeatureLogProbs = new Array<number>(vocabularySize);
  const hamFeatureLogProbs = new Array<number>(vocabularySize);

  // CNB assigns weight w_{c, j} = - log ( (N_{~c, j} + alpha) / (N_{~c} + alpha * |V|) )
  // and normalizes weights
  for (let i = 0; i < vocabularySize; i++) {
    // Ham complement -> inverted log prob for spam
    spamFeatureLogProbs[i] = -Math.log((notSpamSums[i] + alpha) / spamDenominator);
    // Spam complement -> inverted log prob for ham
    hamFeatureLogProbs[i] = -Math.log((notHamSums[i] + alpha) / hamDenominator);
  }

  const spamPrior = (spamCount + 1) / (numSamples + 2);
  const hamPrior = (hamCount + 1) / (numSamples + 2);

  return {
    type: 'complement_nb',
    vocabularySize,
    classPriors: { spam: spamPrior, ham: hamPrior },
    logClassPriors: { spam: Math.log(spamPrior), ham: Math.log(hamPrior) },
    spamFeatureLogProbs,
    hamFeatureLogProbs,
    alpha,
  };
}

export function predictNaiveBayes(
  vector: SparseVector,
  model: NaiveBayesModel,
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
  let logSpamScore = model.logClassPriors.spam;
  let logHamScore = model.logClassPriors.ham;

  for (let i = 0; i < vector.indices.length; i++) {
    const idx = vector.indices[i];
    const val = vector.values[i];
    logSpamScore += val * model.spamFeatureLogProbs[idx];
    logHamScore += val * model.hamFeatureLogProbs[idx];
  }

  // Decision score: difference in log-likelihoods
  const logOdds = logSpamScore - logHamScore;

  // Calibrated probability via numerically stable sigmoid
  // P(Spam) = 1 / (1 + exp(-logOdds))
  let spamProb: number;
  if (logOdds > 35) {
    spamProb = 0.9999;
  } else if (logOdds < -35) {
    spamProb = 0.0001;
  } else {
    spamProb = 1.0 / (1.0 + Math.exp(-logOdds));
  }
  const hamProb = 1.0 - spamProb;

  const predictedLabel: EmailLabel = spamProb >= 0.5 ? 'spam' : 'ham';
  const confidence = predictedLabel === 'spam' ? spamProb : hamProb;

  // Build token-level attribution
  const tokenMap = new Map<string, number>();
  for (const t of tokens) {
    tokenMap.set(t, (tokenMap.get(t) || 0) + 1);
  }

  const tokenAttributions: TokenAttribution[] = [];
  for (const [token, count] of tokenMap.entries()) {
    const idx = vectorizer.vocabulary.get(token);
    if (idx !== undefined) {
      const spamLog = model.spamFeatureLogProbs[idx];
      const hamLog = model.hamFeatureLogProbs[idx];
      const diff = spamLog - hamLog;

      // Bound affinity between -1 and +1
      const affinity = Math.tanh(diff * 0.8);

      tokenAttributions.push({
        token,
        count,
        inVocab: true,
        spamAffinity: affinity,
        logRatio: diff,
        rawWeight: diff * count,
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
    rawScore: logOdds,
    tokenAttributions,
  };
}
