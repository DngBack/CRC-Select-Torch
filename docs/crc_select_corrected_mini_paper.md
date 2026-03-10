# CRC-Select: Learning Rejection Policies Optimized for Conformal Risk Control

## Motivation

Modern classifiers can achieve strong in-distribution accuracy yet behave unreliably under shift or on hard examples. In safety-sensitive applications, a natural deployment rule is **selective prediction**: predict on examples judged reliable and abstain otherwise. The practical objective is to maximize coverage while keeping the risk of accepted predictions low.

Classical learning-to-reject methods, including confidence thresholding and SelectiveNet-style architectures, optimize an empirical risk-coverage trade-off but typically do not provide an auditable finite-sample guarantee on the deployed rule. Conformal Risk Control (CRC), by contrast, gives finite-sample control of the expected value of a bounded monotone loss under exchangeability via a held-out calibration set. However, standard CRC treats the score function as fixed; it does not explain how to **learn** a selector that makes the final conformal threshold less conservative.

This paper studies that missing interaction. The core idea is to train the selector so that the family of CRC-calibrated acceptance rules achieves **higher coverage at the same audited risk target**. Rather than claiming that standard CRC directly controls conditional selective risk, we formulate CRC-Select around a quantity that CRC can actually certify: the **accepted-loss mass**.

## Problem Setup

Let \((X,Y) \sim P\) with input space \(\mathcal X\) and label space \(\mathcal Y\). We learn:

- a predictor \(f_\theta: \mathcal X \to \mathbb R^K\), with class probabilities
  \[
  p_\theta(y\mid x) = \mathrm{softmax}(f_\theta(x))_y,
  \]
- a selector \(g_\phi: \mathcal X \to [0,1]\), interpreted as an acceptance score.

For a threshold \(\tau \in [0,1]\), the deployment policy is
\[
\pi_{\theta,\phi,\tau}(x)=
\begin{cases}
\hat y_\theta(x), & g_\phi(x) \ge \tau,\\
\text{abstain}, & g_\phi(x) < \tau,
\end{cases}
\]
where \(\hat y_\theta(x)=\arg\max_y p_\theta(y\mid x)\).

We track three quantities:

1. **Coverage**
\[
C(\tau)=\mathbb P(g_\phi(X)\ge \tau).
\]

2. **Accepted-loss mass**
\[
A(\tau)=\mathbb E\big[r(X,Y;\theta)\,\mathbf 1\{g_\phi(X)\ge \tau\}\big].
\]

3. **Conditional selective risk**
\[
R_{\mathrm{sel}}(\tau)=\mathbb E[r(X,Y;\theta)\mid g_\phi(X)\ge \tau]
=\frac{A(\tau)}{C(\tau)}
\quad \text{when } C(\tau)>0.
\]

The key distinction is that **CRC directly controls \(A(\tau)\), not \(R_{\mathrm{sel}}(\tau)\)** in the standard split setting.

## Risk Definition

For the vision setting, define the bounded example-wise loss
\[
r(x,y;\theta)=1-p_\theta(y\mid x) \in [0,1].
\]
This loss is small when the model assigns high probability to the true label. The framework also permits cost-sensitive replacements, such as false-negative-weighted or clinically weighted losses, as long as the resulting loss remains bounded.

Now define the CRC loss family
\[
L_i(\tau)=r(X_i,Y_i;\theta)\,\mathbf 1\{g_\phi(X_i)\ge \tau\}.
\]
For each fixed sample \((X_i,Y_i)\), \(L_i(\tau)\) is nonincreasing in \(\tau\), since increasing the acceptance threshold can only remove accepted examples. Moreover, \(0\le L_i(\tau)\le 1\).

These are exactly the properties required by standard CRC.

## CRC Calibration Target

Suppose we have a calibration set \(\mathcal D_{\mathrm{cal}}=\{(X_i,Y_i)\}_{i=1}^n\) independent of training and exchangeable with the future test point. For any threshold \(\tau\), define the empirical accepted-loss mass
\[
\widehat A_n(\tau)=\frac{1}{n}\sum_{i=1}^n L_i(\tau)
=\frac{1}{n}\sum_{i=1}^n r(X_i,Y_i;\theta)\,\mathbf 1\{g_\phi(X_i)\ge \tau\}.
\]

Given a target risk level \(\alpha\in(0,1)\), CRC selects
\[
\hat \tau = \inf\Bigl\{\tau:\; \frac{n}{n+1}\widehat A_n(\tau)+\frac{1}{n+1}\le \alpha\Bigr\}.
\]

Because \(L_i(\tau)\) is bounded in \([0,1]\) and monotone in \(\tau\), standard CRC implies the following.

## Theorem 1 (Finite-Sample Accepted-Loss Control)

Assume \((X_1,Y_1),\dots,(X_n,Y_n),(X_{n+1},Y_{n+1})\) are exchangeable, and fix trained parameters \((\theta,\phi)\). Let
\[
L_i(\tau)=r(X_i,Y_i;\theta)\,\mathbf 1\{g_\phi(X_i)\ge \tau\},
\qquad 0\le L_i(\tau)\le 1.
\]
If \(\hat \tau\) is chosen by the CRC rule above, then the deployed policy satisfies
\[
\mathbb E\big[r(X_{n+1},Y_{n+1};\theta)\,\mathbf 1\{g_\phi(X_{n+1})\ge \hat \tau\}\big] \le \alpha.
\]

### Interpretation

The theorem certifies the expected **loss mass on accepted predictions**. It does **not** directly certify the conditional selective risk \(R_{\mathrm{sel}}(\hat\tau)\). However, since
\[
A(\tau)=C(\tau)\,R_{\mathrm{sel}}(\tau),
\]
improving coverage at a fixed certified accepted-loss budget yields a better risk-coverage operating point.

## Why Post-hoc CRC Can Be Inefficient

Post-hoc CRC takes the learned selector as fixed and calibrates only the final threshold \(\tau\). If the selector assigns high acceptance scores to many difficult or error-prone examples, then the curve \(A(\tau)\) decreases slowly as \(\tau\) increases. Consequently, the CRC-calibrated threshold \(\hat\tau\) must be large to satisfy the target \(\alpha\), which reduces coverage.

CRC-Select targets this inefficiency directly: it learns \(g_\phi\) so that accepted-loss mass falls quickly as the threshold tightens, making the conformally calibrated threshold less conservative.

## CRC-Select Training Objective

We split the data into:

- \(\mathcal D_{\mathrm{train}}\) for gradient-based training,
- \(\mathcal D_{\mathrm{cal}}\) for held-out CRC calibration,
- \(\mathcal D_{\mathrm{test}}\) for final evaluation.

Training uses a differentiable surrogate aligned with the certified quantity \(A(\tau)\).

For a minibatch \(\{(x_i,y_i)\}_{i=1}^m\), let \(g_i=g_\phi(x_i)\) and \(r_i=r(x_i,y_i;\theta)\). We use:

### Prediction loss
\[
\mathcal L_{\mathrm{pred}}=\frac{\sum_{i=1}^m g_i\,\mathrm{CE}(f_\theta(x_i),y_i)}{\sum_{i=1}^m g_i + \varepsilon},
\]
which encourages good predictive performance on examples likely to be accepted.

### Coverage regularizer
\[
\widehat C=\frac{1}{m}\sum_{i=1}^m g_i,
\qquad
\mathcal L_{\mathrm{cov}}=(\widehat C-c_0)^2,
\]
where \(c_0\in(0,1)\) is a target soft coverage level.

### CRC-aligned accepted-loss penalty
\[
\widehat A_{\mathrm{mb}}=\frac{1}{m}\sum_{i=1}^m g_i r_i,
\qquad
\mathcal L_{\mathrm{risk}}=\max\{0,\widehat A_{\mathrm{mb}}-\alpha\}.
\]

Unlike a minibatch conditional-risk ratio, \(\widehat A_{\mathrm{mb}}\) matches the quantity that the final CRC step certifies.

### Total objective
\[
\mathcal L = \mathcal L_{\mathrm{pred}} + \beta \mathcal L_{\mathrm{cov}} + \mu \mathcal L_{\mathrm{risk}},
\]
with trade-off parameters \(\beta,\mu>0\).

## Training Algorithm

Because CRC calibration involves order statistics and threshold selection, it is generally non-smooth. We therefore use alternating optimization.

### Algorithm: CRC-Select

1. **Warm start.** Train \((\theta,\phi)\) with a standard selective prediction objective.
2. **Calibration step (no gradient).** Using the current \((\theta,\phi)\), compute the CRC threshold \(\hat\tau\) on \(\mathcal D_{\mathrm{cal}}\).
3. **Training step.** Holding \(\hat\tau\) fixed, update \((\theta,\phi)\) on \(\mathcal D_{\mathrm{train}}\) for \(T\) gradient steps using the loss above.
4. Repeat steps 2–3 until convergence.
5. **Final calibration.** After training is complete, freeze \((\theta,\phi)\) and perform one final CRC calibration on \(\mathcal D_{\mathrm{cal}}\) to obtain the deployed threshold.

The formal guarantee attaches to the **final frozen model and final held-out calibration step**.

## What CRC-Select Claims

CRC-Select makes two distinct claims:

1. **Finite-sample guarantee.** Under exchangeability, the final deployed policy controls accepted-loss mass at the target level \(\alpha\).
2. **Learning claim.** By training the selector to shape the accepted-loss curve, CRC-Select can achieve higher coverage than post-hoc CRC at the same audited target.

This separation is important: the first claim is theorem-backed, while the second is an empirical or optimization claim.

## Evaluation Protocol

We recommend reporting:

- **Coverage vs. accepted-loss mass** on in-distribution data,
- **Coverage at fixed target \(\alpha\)** after final CRC calibration,
- **Conditional selective risk** \(R_{\mathrm{sel}}\) as a descriptive metric,
- **Violation frequency** across repeated splits and random seeds,
- **OOD dangerous acceptance rate**: the fraction of shifted or OOD points accepted.

The main audited metric should match the theorem, namely accepted-loss mass.

## Baselines

To isolate the contribution, compare against:

1. **SelectiveNet / learning-to-reject baseline** without conformal calibration,
2. **Post-hoc CRC on a fixed selector**,
3. **OOD-aware gating baseline** such as energy-based rejection,
4. **If available, recent selective conformal baselines** that combine selection and conformal calibration but do not explicitly optimize CRC-tightness during training.

## Scope and Limitations

The finite-sample theorem relies on exchangeability between calibration and test data. Therefore, under distribution shift or OOD settings, CRC-Select does not automatically retain the same formal guarantee unless one uses an extension such as weighted or non-exchangeable CRC. In those settings, experiments should be described as robustness evaluations rather than covered by the main theorem.

## Summary

CRC-Select couples selective prediction with conformal risk control by training the selector to improve the efficiency of the final CRC calibration step. The mathematically correct object certified by standard CRC is the accepted-loss mass
\[
\mathbb E[r(X,Y;\theta)\mathbf 1\{g_\phi(X)\ge \hat\tau\}],
\]
not the conditional selective risk directly. This formulation preserves a valid finite-sample guarantee while still targeting the practical goal of obtaining higher coverage at the same audited risk level.

