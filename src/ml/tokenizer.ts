import { STOP_WORDS } from './stopwords';
import { StructuralSignals } from './types';

// Common high-priority urgency and scam triggers in email headers & text
export const URGENT_TRIGGERS = [
  'urgent', 'act now', 'immediately', 'verify your account', 'account suspended',
  'winner', 'congratulations', 'lottery', 'claim prize', 'million dollars',
  'wire transfer', 'cryptocurrency', 'bitcoin', 'guaranteed profit', 'no risk',
  'exclusive deal', 'risk free', 'click here', 'confirm password', 'unauthorized access',
  'security alert', 'final notice', 'action required', 'bank alert', 'dear beneficiary',
  'inheritance', 'western union', 'limited time offer', 'instant payout', 'free money'
];

export function extractStructuralSignals(text: string): StructuralSignals {
  if (!text) {
    return {
      capsRatio: 0,
      exclamationCount: 0,
      dollarCount: 0,
      suspiciousUrlCount: 0,
      urgentKeywordCount: 0,
      detectedUrgentKeywords: [],
    };
  }

  // Count uppercase letters vs total alphabetic letters
  const letters = text.replace(/[^a-zA-Z]/g, '');
  const uppercaseLetters = text.replace(/[^A-Z]/g, '');
  const capsRatio = letters.length > 0 ? uppercaseLetters.length / letters.length : 0;

  // Exclamation and dollar counts
  const exclamationMatches = text.match(/!/g);
  const exclamationCount = exclamationMatches ? exclamationMatches.length : 0;

  const dollarMatches = text.match(/\$|€|£|USD|cash/gi);
  const dollarCount = dollarMatches ? dollarMatches.length : 0;

  // Suspicious IP addresses or obscure URL patterns
  const urlMatches = text.match(/https?:\/\/[^\s]+|www\.[^\s]+|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/gi);
  const suspiciousUrlCount = urlMatches ? urlMatches.length : 0;

  // Urgent keywords
  const lowerText = text.toLowerCase();
  const detectedUrgentKeywords: string[] = [];
  for (const trigger of URGENT_TRIGGERS) {
    if (lowerText.includes(trigger)) {
      detectedUrgentKeywords.push(trigger);
    }
  }

  return {
    capsRatio,
    exclamationCount,
    dollarCount,
    suspiciousUrlCount,
    urgentKeywordCount: detectedUrgentKeywords.length,
    detectedUrgentKeywords,
  };
}

export interface TokenizeOptions {
  useBigrams?: boolean;
  removeStopwords?: boolean;
  minTokenLength?: number;
}

export function tokenizeText(
  text: string,
  options: TokenizeOptions = {}
): string[] {
  const { useBigrams = false, removeStopwords = true, minTokenLength = 2 } = options;

  if (!text) return [];

  // Normalize: lower case and replace punctuation with space, but keep characters and currency indicators
  const normalized = text
    .toLowerCase()
    .replace(/[^\w\s$!]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  // Extract base words
  const rawTokens = normalized.split(/\s+/).filter(token => {
    // Keep special signals like "$" or long enough words
    if (token === '$') return true;
    if (token.length < minTokenLength) return false;
    // Check if purely numeric
    if (/^\d+$/.test(token) && token.length > 6) return false; // Ignore large random numbers
    return true;
  });

  const filteredTokens = removeStopwords
    ? rawTokens.filter(t => !STOP_WORDS.has(t) || t === '$')
    : rawTokens;

  if (!useBigrams) {
    return filteredTokens;
  }

  // Generate bigrams
  const result: string[] = [...filteredTokens];
  for (let i = 0; i < filteredTokens.length - 1; i++) {
    const bigram = `${filteredTokens[i]}_${filteredTokens[i + 1]}`;
    result.push(bigram);
  }

  return result;
}
