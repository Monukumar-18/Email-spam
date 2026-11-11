import React from 'react';
import { Shield, Cpu, Zap } from 'lucide-react';

export const AlgorithmGuideView: React.FC = () => {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-10">
      {/* Page Title */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-5">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Mathematical Foundations & Machine Learning Theory
        </h1>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          First-principles derivations for Multinomial Naive Bayes, Linear Support Vector Machines, and TF-IDF Vectorization.
        </p>
      </div>

      {/* Section 1: Naive Bayes */}
      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-slate-100 dark:border-slate-800">
          <div className="w-7 h-7 rounded-md bg-rose-500/10 text-rose-500 flex items-center justify-center font-bold text-xs">
            01
          </div>
          <h2 className="text-base font-bold text-slate-900 dark:text-white">
            Multinomial Naive Bayes Classifier (MNB)
          </h2>
        </div>

        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          Naive Bayes is a probabilistic classifier founded on <strong>Bayes' Theorem</strong>. Given an email document represented as a sequence of words D = (w₁, w₂, ..., wₙ), we seek the class y ∈ &#123;Spam, Ham&#125; that maximizes the posterior probability:
        </p>

        {/* Math Callout */}
        <div className="p-4 bg-slate-50 dark:bg-slate-950/80 rounded-lg border border-slate-200 dark:border-slate-800 font-mono text-xs sm:text-sm text-slate-800 dark:text-slate-200 overflow-x-auto space-y-2">
          <div>P(y | D) = [ P(D | y) · P(y) ] / P(D)</div>
          <div className="text-slate-500 text-xs">Since P(D) is constant across classes:</div>
          <div className="text-rose-600 dark:text-rose-400 font-bold">
            y* = argmax_y [ P(y) · ∏ P(w_i | y) ]
          </div>
        </div>

        <div className="space-y-3 text-xs sm:text-sm text-slate-600 dark:text-slate-300">
          <h3 className="font-bold text-slate-900 dark:text-white text-xs uppercase tracking-wider">
            Critical Mathematical Mechanisms:
          </h3>
          <ul className="space-y-2 list-disc pl-5">
            <li>
              <strong>Class-Conditional Independence Assumption ("Naive"):</strong> Assumes that the presence of word w_i is conditionally independent of any other word w_j given the class y. Despite this strong simplification, it performs exceptionally well on text classification.
            </li>
            <li>
              <strong>Laplace (Additive) Smoothing (α):</strong> If a word in the test document never appeared in class y during training, P(w | y) = 0, which would zero out the entire product. We introduce smoothing:
              <div className="p-2.5 my-1.5 bg-slate-100 dark:bg-slate-950 font-mono text-xs rounded border border-slate-200 dark:border-slate-800">
                P(w_i | y) = ( Count(w_i, y) + α ) / ( TotalTokensIn(y) + α · |V| )
              </div>
              Where |V| is the total vocabulary size and α = 1.0 is standard Laplace smoothing.
            </li>
            <li>
              <strong>Log-Space Computation (Underflow Prevention):</strong> Multiplying hundreds of small probabilities leads to floating-point underflow. We compute in log space using summation:
              <div className="p-2.5 my-1.5 bg-slate-100 dark:bg-slate-950 font-mono text-xs rounded border border-slate-200 dark:border-slate-800">
                ln P(y | D) = ln P(y) + ∑ x_i · ln P(w_i | y)
              </div>
            </li>
          </ul>
        </div>
      </section>

      {/* Section 2: Support Vector Machines */}
      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-slate-100 dark:border-slate-800">
          <div className="w-7 h-7 rounded-md bg-sky-500/10 text-sky-500 flex items-center justify-center font-bold text-xs">
            02
          </div>
          <h2 className="text-base font-bold text-slate-900 dark:text-white">
            Linear Support Vector Machine (Pegasos SGD)
          </h2>
        </div>

        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          Support Vector Machines construct a maximum-margin hyperplane wᵀ x + b = 0 in the high-dimensional feature space that maximizes the geometric margin (distance 2 / ||w||) to the nearest training samples from either class (the support vectors).
        </p>

        {/* Math Callout */}
        <div className="p-4 bg-slate-50 dark:bg-slate-950/80 rounded-lg border border-slate-200 dark:border-slate-800 font-mono text-xs sm:text-sm text-slate-800 dark:text-slate-200 overflow-x-auto space-y-2">
          <div>Primal Objective (Soft-Margin Hinge Loss):</div>
          <div className="text-sky-600 dark:text-sky-400 font-bold">
            min_(w, b) [ (1/2) ||w||² + C · ∑ max(0, 1 - y_i(wᵀ x_i + b)) ]
          </div>
          <div className="text-slate-500 text-xs">where y_i ∈ &#123;-1, +1&#125; and C is the regularization penalty.</div>
        </div>

        <div className="space-y-3 text-xs sm:text-sm text-slate-600 dark:text-slate-300">
          <h3 className="font-bold text-slate-900 dark:text-white text-xs uppercase tracking-wider">
            Optimization & Calibration:
          </h3>
          <ul className="space-y-2 list-disc pl-5">
            <li>
              <strong>Pegasos Stochastic Subgradient Descent:</strong> We iterate over training vectors x_i. If y_i(wᵀ x_i + b) &lt; 1 (margin violation), weights are updated along the subgradient: w ← (1 - ηλ)w + η C y_i x_i. This provides ultra-fast convergence for sparse text features.
            </li>
            <li>
              <strong>Platt Scaling for Calibrated Probabilities:</strong> Standard SVM outputs a signed geometric distance f(x) = wᵀ x + b. To turn this into a true probability P(Spam | x), we fit a logistic sigmoid:
              <div className="p-2.5 my-1.5 bg-slate-100 dark:bg-slate-950 font-mono text-xs rounded border border-slate-200 dark:border-slate-800">
                P(y = 1 | x) = 1 / [ 1 + exp(A · f(x) + B) ]
              </div>
            </li>
          </ul>
        </div>
      </section>

      {/* Section 3: TF-IDF Vectorization */}
      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex items-center gap-2 pb-2 border-b border-slate-100 dark:border-slate-800">
          <div className="w-7 h-7 rounded-md bg-emerald-500/10 text-emerald-500 flex items-center justify-center font-bold text-xs">
            03
          </div>
          <h2 className="text-base font-bold text-slate-900 dark:text-white">
            TF-IDF Feature Representation
          </h2>
        </div>

        <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
          Raw word counts can disproportionately weight common words. <strong>TF-IDF</strong> (Term Frequency - Inverse Document Frequency) downweights words that appear across all documents while emphasizing terms that uniquely characterize specific emails.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="p-3 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800">
            <span className="font-bold text-slate-900 dark:text-white block mb-1">
              Term Frequency (TF):
            </span>
            TF(t, d) = count(t, d) / total_tokens(d)
          </div>
          <div className="p-3 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800">
            <span className="font-bold text-slate-900 dark:text-white block mb-1">
              Inverse Document Frequency (IDF):
            </span>
            IDF(t) = ln((1 + N) / (1 + DF(t))) + 1
          </div>
        </div>
      </section>

      {/* Section 4: Architecture Comparison */}
      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 shadow-xs space-y-4">
        <h3 className="text-base font-bold text-slate-900 dark:text-white">
          Why Linear Models Excel in Production Email Filters
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800 space-y-2">
            <span className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-amber-500" /> Microsecond Latency
            </span>
            <p className="text-slate-500 dark:text-slate-400 leading-relaxed">
              Inference requires only a sparse dot product (wᵀ x + b), taking under 0.5 milliseconds per email.
            </p>
          </div>

          <div className="p-4 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800 space-y-2">
            <span className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-sky-500" /> High Dimensionality
            </span>
            <p className="text-slate-500 dark:text-slate-400 leading-relaxed">
              Text classification features live in high-dimensional spaces where classes are naturally linearly separable.
            </p>
          </div>

          <div className="p-4 bg-slate-50 dark:bg-slate-950/60 rounded-lg border border-slate-200 dark:border-slate-800 space-y-2">
            <span className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
              <Shield className="w-4 h-4 text-emerald-500" /> Full Explainability
            </span>
            <p className="text-slate-500 dark:text-slate-400 leading-relaxed">
              Every prediction can be audited directly by inspecting exact feature log-odds ratios and weights for compliance.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};
