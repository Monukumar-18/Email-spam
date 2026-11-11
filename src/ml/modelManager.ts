import { computeEvaluationMetrics, stratifiedTrainTestSplit } from './evaluator';
import { NaiveBayesModel, predictNaiveBayes, trainComplementNaiveBayes, trainMultinomialNaiveBayes } from './naiveBayes';
import { LinearSvmModel, predictLinearSvm, trainLinearSvm } from './svm';
import { extractStructuralSignals, tokenizeText } from './tokenizer';
import {
  EmailSample,
  Hyperparameters,
  ModelMetrics,
  PredictionResult,
  TrainedModelsState,
} from './types';
import { buildVocabulary, SparseVector, transformDocument, VectorizerModel } from './vectorizer';

export class SpamClassifierManager {
  private hyperparameters: Hyperparameters;
  private vectorizer: VectorizerModel | null = null;
  private multinomialNb: NaiveBayesModel | null = null;
  private complementNb: NaiveBayesModel | null = null;
  private linearSvm: LinearSvmModel | null = null;
  private state: TrainedModelsState | null = null;

  constructor(hyperparameters?: Partial<Hyperparameters>) {
    this.hyperparameters = {
      alpha: 1.0,
      svmC: 1.0,
      svmEpochs: 25,
      svmLearningRate: 0.05,
      vectorizer: 'tfidf',
      useBigrams: true,
      removeStopwords: true,
      minWordFreq: 2,
      testSplitRatio: 0.25,
      ...hyperparameters,
    };
  }

  public getHyperparameters(): Hyperparameters {
    return { ...this.hyperparameters };
  }

  public setHyperparameters(newParams: Partial<Hyperparameters>): void {
    this.hyperparameters = { ...this.hyperparameters, ...newParams };
  }

  public getState(): TrainedModelsState | null {
    return this.state;
  }

  public trainAll(dataset: EmailSample[]): TrainedModelsState {
    const startTime = performance.now();

    // 1. Stratified Train / Test Split
    const { train, test } = stratifiedTrainTestSplit(
      dataset,
      this.hyperparameters.testSplitRatio
    );

    // 2. Tokenize all training documents
    const trainDocsTokens: string[][] = train.map(doc => {
      const fullText = `${doc.subject} ${doc.body}`;
      return tokenizeText(fullText, {
        useBigrams: this.hyperparameters.useBigrams,
        removeStopwords: this.hyperparameters.removeStopwords,
      });
    });

    // 3. Build Vocabulary and IDF weights
    const { vocabulary, inverseVocabulary, idf } = buildVocabulary(
      trainDocsTokens,
      this.hyperparameters.minWordFreq
    );

    const vectorizer: VectorizerModel = {
      vocabulary,
      inverseVocabulary,
      idf,
      docCount: train.length,
      type: this.hyperparameters.vectorizer,
    };
    this.vectorizer = vectorizer;

    // 4. Transform training set into Sparse Vectors
    const trainVectors: SparseVector[] = trainDocsTokens.map(tokens =>
      transformDocument(tokens, vectorizer)
    );
    const trainLabels = train.map(d => d.label);

    // 5. Train Multinomial Naive Bayes
    const mnbStartTime = performance.now();
    const mnbModel = trainMultinomialNaiveBayes(
      trainVectors,
      trainLabels,
      inverseVocabulary.length,
      this.hyperparameters.alpha
    );
    const mnbTrainTime = performance.now() - mnbStartTime;
    this.multinomialNb = mnbModel;

    // 6. Train Complement Naive Bayes
    const cnbStartTime = performance.now();
    const cnbModel = trainComplementNaiveBayes(
      trainVectors,
      trainLabels,
      inverseVocabulary.length,
      this.hyperparameters.alpha
    );
    const cnbTrainTime = performance.now() - cnbStartTime;
    this.complementNb = cnbModel;

    // 7. Train Linear Support Vector Machine
    const svmStartTime = performance.now();
    const svmModel = trainLinearSvm(
      trainVectors,
      trainLabels,
      inverseVocabulary.length,
      this.hyperparameters.svmC,
      this.hyperparameters.svmEpochs,
      this.hyperparameters.svmLearningRate
    );
    const svmTrainTime = performance.now() - svmStartTime;
    this.linearSvm = svmModel;

    // 8. Transform Test Set for Evaluation
    const testDocsTokens: string[][] = test.map(doc => {
      const fullText = `${doc.subject} ${doc.body}`;
      return tokenizeText(fullText, {
        useBigrams: this.hyperparameters.useBigrams,
        removeStopwords: this.hyperparameters.removeStopwords,
      });
    });
    const testVectors: SparseVector[] = testDocsTokens.map(tokens =>
      transformDocument(tokens, vectorizer)
    );

    // 9. Predict on Test Set for MNB
    const mnbPredictions = testVectors.map((vec, i) =>
      predictNaiveBayes(vec, mnbModel, testDocsTokens[i], vectorizer)
    );
    const mnbMetrics = computeEvaluationMetrics(
      test,
      mnbPredictions,
      train.length,
      mnbTrainTime
    );

    // 10. Predict on Test Set for CNB
    const cnbPredictions = testVectors.map((vec, i) =>
      predictNaiveBayes(vec, cnbModel, testDocsTokens[i], vectorizer)
    );
    const cnbMetrics = computeEvaluationMetrics(
      test,
      cnbPredictions,
      train.length,
      cnbTrainTime
    );

    // 11. Predict on Test Set for SVM
    const svmPredictions = testVectors.map((vec, i) =>
      predictLinearSvm(vec, svmModel, testDocsTokens[i], vectorizer)
    );
    const svmMetrics = computeEvaluationMetrics(
      test,
      svmPredictions,
      train.length,
      svmTrainTime
    );

    // 12. Extract top global spam/ham tokens from SVM & MNB
    const tokenRankings: Array<{ word: string; score: number }> = [];
    for (let i = 0; i < inverseVocabulary.length; i++) {
      const word = inverseVocabulary[i];
      const svmW = svmModel.weights[i] || 0;
      const nbDiff = mnbModel.spamFeatureLogProbs[i] - mnbModel.hamFeatureLogProbs[i];
      // Combined normalized score
      const combinedScore = (svmW * 1.5) + (nbDiff * 0.5);
      tokenRankings.push({ word, score: combinedScore });
    }

    tokenRankings.sort((a, b) => b.score - a.score);
    const topSpamVocab = tokenRankings.slice(0, 15);
    const topHamVocab = tokenRankings
      .slice(-15)
      .reverse()
      .map(t => ({ word: t.word, score: -t.score }));

    const trainedState: TrainedModelsState = {
      naiveBayesMetrics: mnbMetrics,
      svmMetrics: svmMetrics,
      complementNbMetrics: cnbMetrics,
      vocabularySize: inverseVocabulary.length,
      totalCorpusSize: dataset.length,
      trainSize: train.length,
      testSize: test.length,
      topSpamVocab,
      topHamVocab,
      hyperparameters: { ...this.hyperparameters },
      lastTrainedAt: Date.now(),
    };

    this.state = trainedState;
    return trainedState;
  }

  public predict(
    text: string,
    modelType: 'naive_bayes' | 'svm' | 'complement_nb' = 'naive_bayes'
  ): PredictionResult {
    const t0 = performance.now();

    if (!this.vectorizer || !this.multinomialNb || !this.linearSvm || !this.complementNb) {
      throw new Error('Models have not been trained yet. Please call trainAll() first.');
    }

    const structuralSignals = extractStructuralSignals(text);
    const tokens = tokenizeText(text, {
      useBigrams: this.hyperparameters.useBigrams,
      removeStopwords: this.hyperparameters.removeStopwords,
    });

    const vector = transformDocument(tokens, this.vectorizer);

    let rawPrediction: {
      predictedLabel: 'spam' | 'ham';
      spamProbability: number;
      hamProbability: number;
      confidence: number;
      rawScore: number;
      tokenAttributions: any[];
    };
    let modelDisplayName = 'Multinomial Naive Bayes';

    if (modelType === 'svm') {
      rawPrediction = predictLinearSvm(vector, this.linearSvm, tokens, this.vectorizer);
      modelDisplayName = 'Linear Support Vector Machine';
    } else if (modelType === 'complement_nb') {
      rawPrediction = predictNaiveBayes(vector, this.complementNb, tokens, this.vectorizer);
      modelDisplayName = 'Complement Naive Bayes';
    } else {
      rawPrediction = predictNaiveBayes(vector, this.multinomialNb, tokens, this.vectorizer);
      modelDisplayName = 'Multinomial Naive Bayes';
    }

    // Heuristic boost if intense structural signals are present (e.g. 3+ urgent triggers and multiple dollar signs)
    let adjustedSpamProb = rawPrediction.spamProbability;
    if (structuralSignals.urgentKeywordCount >= 2 || (structuralSignals.capsRatio > 0.25 && structuralSignals.dollarCount > 1)) {
      adjustedSpamProb = Math.min(0.999, adjustedSpamProb + 0.05);
    }
    const adjustedHamProb = 1 - adjustedSpamProb;
    const finalPredictedLabel = adjustedSpamProb >= 0.5 ? 'spam' : 'ham';
    const finalConfidence = finalPredictedLabel === 'spam' ? adjustedSpamProb : adjustedHamProb;

    // Sort tokens by spam contribution
    const sortedTokens = [...rawPrediction.tokenAttributions].sort(
      (a, b) => Math.abs(b.rawWeight) - Math.abs(a.rawWeight)
    );

    const topSpamTokens = sortedTokens
      .filter(t => t.rawWeight > 0.01)
      .slice(0, 6)
      .map(t => ({ token: t.token, weight: t.rawWeight }));

    const topHamTokens = sortedTokens
      .filter(t => t.rawWeight < -0.01)
      .slice(0, 6)
      .map(t => ({ token: t.token, weight: Math.abs(t.rawWeight) }));

    const inferenceTimeMs = performance.now() - t0;

    return {
      predictedLabel: finalPredictedLabel,
      confidence: finalConfidence,
      spamProbability: adjustedSpamProb,
      hamProbability: adjustedHamProb,
      rawScore: rawPrediction.rawScore,
      modelType,
      modelDisplayName,
      tokens: rawPrediction.tokenAttributions,
      topSpamTokens,
      topHamTokens,
      structuralSignals,
      inferenceTimeMs,
    };
  }
}
