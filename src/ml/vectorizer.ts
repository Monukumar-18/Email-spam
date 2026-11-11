import { VectorizerType } from './types';

export interface VectorizerModel {
  vocabulary: Map<string, number>; // token -> featureIndex
  inverseVocabulary: string[]; // featureIndex -> token
  idf: number[]; // featureIndex -> IDF weight
  docCount: number;
  type: VectorizerType;
}

export interface SparseVector {
  indices: number[];
  values: number[];
}

export function buildVocabulary(
  documentsTokens: string[][],
  minWordFreq: number = 2
): { vocabulary: Map<string, number>; inverseVocabulary: string[]; idf: number[] } {
  const docFreq = new Map<string, number>();
  const totalDocs = documentsTokens.length;

  // Compute document frequency for each token
  for (const tokens of documentsTokens) {
    const uniqueTokens = new Set(tokens);
    for (const token of uniqueTokens) {
      docFreq.set(token, (docFreq.get(token) || 0) + 1);
    }
  }

  // Filter by minimum frequency
  const vocabulary = new Map<string, number>();
  const inverseVocabulary: string[] = [];
  const idf: number[] = [];

  let featureIdx = 0;
  for (const [token, df] of docFreq.entries()) {
    if (df >= minWordFreq) {
      vocabulary.set(token, featureIdx);
      inverseVocabulary.push(token);
      // Smooth IDF: ln((1 + N) / (1 + df)) + 1
      const idfValue = Math.log((1 + totalDocs) / (1 + df)) + 1.0;
      idf.push(idfValue);
      featureIdx++;
    }
  }

  return { vocabulary, inverseVocabulary, idf };
}

export function transformDocument(
  tokens: string[],
  vectorizer: VectorizerModel
): SparseVector {
  const { vocabulary, idf, type } = vectorizer;
  const counts = new Map<number, number>();

  for (const token of tokens) {
    const idx = vocabulary.get(token);
    if (idx !== undefined) {
      counts.set(idx, (counts.get(idx) || 0) + 1);
    }
  }

  const indices: number[] = [];
  const values: number[] = [];
  const totalTokens = tokens.length || 1;

  for (const [idx, count] of counts.entries()) {
    indices.push(idx);

    if (type === 'binary') {
      values.push(1.0);
    } else if (type === 'bow') {
      values.push(count);
    } else {
      // tfidf: (count / totalTokens) * idf[idx] or standard sublinear TF
      const tf = count / totalTokens;
      const tfidfVal = tf * idf[idx];
      values.push(tfidfVal);
    }
  }

  // Apply L2 normalization for TF-IDF to standard vector space length 1.0
  if (type === 'tfidf' && values.length > 0) {
    let normSq = 0;
    for (const v of values) {
      normSq += v * v;
    }
    const norm = Math.sqrt(normSq) || 1e-9;
    for (let i = 0; i < values.length; i++) {
      values[i] = values[i] / norm;
    }
  }

  return { indices, values };
}

export function dotProduct(v: SparseVector, weights: number[]): number {
  let sum = 0;
  for (let i = 0; i < v.indices.length; i++) {
    const idx = v.indices[i];
    sum += v.values[i] * (weights[idx] || 0);
  }
  return sum;
}
