"""Latent trust-state decoder: a Gaussian Hidden Markov Model trained by
Expectation-Maximization (Baum-Welch) with Viterbi MAP decoding.

Hand-rolled in NumPy (no hmmlearn dependency) with a numerically stable
log-space forward-backward recursion. Observations are the 4-channel kinematic
telemetry; the K=4 hidden states correspond to the latent behavioral regimes
(Calibrated_Compliance, Skeptical_Delay, Herd_Following, Advisory_Ignore).

Because EM is invariant to state relabeling, `align_states` recovers the best
permutation against ground-truth labels for honest accuracy reporting.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

try:                                   # optional, only for smarter init
    from sklearn.cluster import KMeans
    _HAVE_SK = True
except Exception:                      # pragma: no cover
    _HAVE_SK = False

_LOG_EPS = -1e300


class GaussianHMM:
    def __init__(self, n_states: int = 4, n_iter: int = 40, tol: float = 1e-4,
                 reg: float = 1e-3, seed: int = 0):
        self.K = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.reg = reg
        self.rng = np.random.default_rng(seed)
        self.pi = None          # [K]
        self.A = None           # [K,K] row-stochastic transition matrix
        self.mu = None          # [K,d]
        self.cov = None         # [K,d,d]
        self.loglik_history_ = []

    # ---- emissions -------------------------------------------------------
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """[T,K] log N(x_t | mu_k, cov_k)."""
        T = X.shape[0]
        logB = np.empty((T, self.K))
        for k in range(self.K):
            logB[:, k] = multivariate_normal.logpdf(
                X, mean=self.mu[k], cov=self.cov[k], allow_singular=True)
        return logB

    # ---- initialization --------------------------------------------------
    def _init(self, pooled: np.ndarray):
        d = pooled.shape[1]
        if _HAVE_SK:
            km = KMeans(n_clusters=self.K, n_init=5,
                        random_state=0).fit(pooled)
            labels = km.labels_
            self.mu = km.cluster_centers_.copy()
        else:                                           # pragma: no cover
            idx = self.rng.choice(len(pooled), self.K, replace=False)
            self.mu = pooled[idx].copy()
            labels = np.argmin(
                ((pooled[:, None, :] - self.mu[None]) ** 2).sum(-1), axis=1)
        self.cov = np.empty((self.K, d, d))
        for k in range(self.K):
            pts = pooled[labels == k]
            c = np.cov(pts.T) if len(pts) > d else np.eye(d)
            self.cov[k] = np.atleast_2d(c) + self.reg * np.eye(d)
        self.pi = np.full(self.K, 1.0 / self.K)
        self.A = np.full((self.K, self.K), 1.0 / self.K)

    # ---- forward-backward (log space) ------------------------------------
    def _forward_backward(self, logB: np.ndarray):
        T = logB.shape[0]
        logpi, logA = np.log(self.pi + 1e-300), np.log(self.A + 1e-300)
        la = np.empty((T, self.K))
        la[0] = logpi + logB[0]
        for t in range(1, T):
            la[t] = logB[t] + logsumexp(la[t - 1][:, None] + logA, axis=0)
        lb = np.empty((T, self.K))
        lb[-1] = 0.0
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(logA + logB[t + 1][None, :] + lb[t + 1][None, :],
                              axis=1)
        ll = logsumexp(la[-1])
        log_gamma = la + lb - ll
        return la, lb, ll, log_gamma, logA

    # ---- fit (multiple sequences) ----------------------------------------
    def fit(self, sequences: list[np.ndarray]):
        pooled = np.vstack(sequences)
        self._init(pooled)
        prev_ll = -np.inf
        for _ in range(self.n_iter):
            # accumulators
            pi_acc = np.zeros(self.K)
            A_num = np.zeros((self.K, self.K))
            gsum = np.zeros(self.K)
            mu_num = np.zeros_like(self.mu)
            cov_num = np.zeros_like(self.cov)
            total_ll = 0.0

            for X in sequences:
                T = X.shape[0]
                logB = self._log_emission(X)
                la, lb, ll, log_gamma, logA = self._forward_backward(logB)
                total_ll += ll
                gamma = np.exp(log_gamma)                 # [T,K]
                pi_acc += gamma[0]
                gsum += gamma.sum(0)
                mu_num += gamma.T @ X
                # xi (summed over t)
                for t in range(T - 1):
                    lxi = (la[t][:, None] + logA
                           + logB[t + 1][None, :] + lb[t + 1][None, :] - ll)
                    A_num += np.exp(lxi)
                # covariance accumulation
                for k in range(self.K):
                    diff = X - self.mu[k]
                    cov_num[k] += (gamma[:, k][:, None, None]
                                   * (diff[:, :, None] * diff[:, None, :])).sum(0)

            # M-step
            self.pi = pi_acc / pi_acc.sum()
            self.A = A_num / A_num.sum(1, keepdims=True).clip(min=1e-300)
            self.mu = mu_num / gsum[:, None].clip(min=1e-300)
            d = pooled.shape[1]
            for k in range(self.K):
                self.cov[k] = (cov_num[k] / max(gsum[k], 1e-300)
                               + self.reg * np.eye(d))

            self.loglik_history_.append(total_ll)
            if abs(total_ll - prev_ll) < self.tol * (1 + abs(prev_ll)):
                break
            prev_ll = total_ll
        return self

    # ---- Viterbi MAP decode ----------------------------------------------
    def decode(self, X: np.ndarray) -> np.ndarray:
        T = X.shape[0]
        logB = self._log_emission(X)
        logpi, logA = np.log(self.pi + 1e-300), np.log(self.A + 1e-300)
        delta = np.empty((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)
        delta[0] = logpi + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = logB[t] + np.max(scores, axis=0)
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def posterior_trust(self, X: np.ndarray, calibrated_state: int) -> float:
        """Mean posterior probability of the Calibrated_Compliance state."""
        logB = self._log_emission(X)
        _, _, _, log_gamma, _ = self._forward_backward(logB)
        return float(np.exp(log_gamma[:, calibrated_state]).mean())


def align_states(pred: np.ndarray, truth: np.ndarray, K: int) -> np.ndarray:
    """Return a mapping pred->truth maximising agreement (Hungarian on the
    contingency table). Used only for honest accuracy reporting."""
    from scipy.optimize import linear_sum_assignment
    M = np.zeros((K, K))
    for p, t in zip(pred, truth):
        M[p, t] += 1
    row, col = linear_sum_assignment(-M)
    mapping = np.zeros(K, dtype=int)
    for r, c in zip(row, col):
        mapping[r] = c
    return mapping


def decoding_accuracy(model: GaussianHMM, sequences: list[dict]) -> float:
    preds = np.concatenate([model.decode(s["obs"]) for s in sequences])
    truth = np.concatenate([s["states"] for s in sequences])
    mapping = align_states(preds, truth, model.K)
    return float((mapping[preds] == truth).mean())


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sim.heterogeneous_driver_sim import simulate_study

    out = simulate_study(n_episodes=50)
    seqs = out.sequences[:400]
    hmm = GaussianHMM(n_states=4, n_iter=40).fit([s["obs"] for s in seqs])
    acc = decoding_accuracy(hmm, seqs)
    print(f"HMM latent-state decoding accuracy: {acc:.3f}")
    print("final log-lik:", round(hmm.loglik_history_[-1], 1),
          "| iters:", len(hmm.loglik_history_))
