<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# still not very clear to me. Do it but step by step

Let me walk through it very carefully, one step at a time.

***

## Start: What we know

We have two quantities:

- $FP_{\text{base}}$ = false positives from baseline
- $N_{\text{skip\_neg}}$ = number of negative frames skipped

And we know:

$$
FP(\theta) = FP_{\text{base}} - a, \quad 0 \leq a \leq \min(FP_{\text{base}},\ N_{\text{skip\_neg}})
$$

***

## Step 1: Find the range of $FP(\theta)$

Since $a$ ranges from $0$ to $\min(FP_{\text{base}},\ N_{\text{skip\_neg}})$, $FP(\theta)$ ranges from:

**Maximum** (when $a = 0$):

$$
FP(\theta) = FP_{\text{base}}
$$

**Minimum** (when $a$ is as large as possible):

$$
FP(\theta) = FP_{\text{base}} - \min(FP_{\text{base}},\ N_{\text{skip\_neg}})
$$

So:

$$
FP(\theta) \in \Big[FP_{\text{base}} - \min(FP_{\text{base}},\ N_{\text{skip\_neg}}),\ \ FP_{\text{base}}\Big]
$$

***

## Step 2: Divide everything by $N_{\text{neg}}$

We divide all three terms by $N_{\text{neg}}$ to convert counts into rates:

$$
\frac{FP(\theta)}{N_{\text{neg}}} \in \left[\frac{FP_{\text{base}} - \min(FP_{\text{base}},\ N_{\text{skip\_neg}})}{N_{\text{neg}}},\ \ \frac{FP_{\text{base}}}{N_{\text{neg}}}\right]
$$

The left side is just $\mathrm{FAR}(\theta)$ by definition.

***

## Step 3: Substitute known definitions

We substitute:

- $\mathrm{FAR}_{\text{base}} = FP_{\text{base}} / N_{\text{neg}}$
- $S_r(\theta) = N_{\text{skip\_neg}} / N_{\text{neg}}$

**Upper bound** is easy:

$$
\frac{FP_{\text{base}}}{N_{\text{neg}}} = \mathrm{FAR}_{\text{base}}
$$

**Lower bound** — split the fraction:

$$
\frac{FP_{\text{base}} - \min(FP_{\text{base}},\ N_{\text{skip\_neg}})}{N_{\text{neg}}} = \frac{FP_{\text{base}}}{N_{\text{neg}}} - \frac{\min(FP_{\text{base}},\ N_{\text{skip\_neg}})}{N_{\text{neg}}}
$$

$$
= \mathrm{FAR}_{\text{base}} - \frac{\min(FP_{\text{base}},\ N_{\text{skip\_neg}})}{N_{\text{neg}}}
$$

***

## Step 4: Simplify the $\min$ term

Using the identity $\min(A, B) / C = \min(A/C,\ B/C)$:

$$
\frac{\min(FP_{\text{base}},\ N_{\text{skip\_neg}})}{N_{\text{neg}}} = \min\!\left(\frac{FP_{\text{base}}}{N_{\text{neg}}},\ \frac{N_{\text{skip\_neg}}}{N_{\text{neg}}}\right) = \min(\mathrm{FAR}_{\text{base}},\ S_r(\theta))
$$

So the lower bound becomes:

$$
\mathrm{FAR}_{\text{base}} - \min(\mathrm{FAR}_{\text{base}},\ S_r(\theta))
$$

***

## Step 5: Simplify using two cases

**Case 1** — $S_r(\theta) \leq \mathrm{FAR}_{\text{base}}$:

$$
\min(\mathrm{FAR}_{\text{base}},\ S_r) = S_r(\theta)
$$

$$
\Rightarrow \text{lower bound} = \mathrm{FAR}_{\text{base}} - S_r(\theta) \geq 0
$$

**Case 2** — $S_r(\theta) > \mathrm{FAR}_{\text{base}}$:

$$
\min(\mathrm{FAR}_{\text{base}},\ S_r) = \mathrm{FAR}_{\text{base}}
$$

$$
\Rightarrow \text{lower bound} = \mathrm{FAR}_{\text{base}} - \mathrm{FAR}_{\text{base}} = 0
$$

Both cases are captured by:

$$
\max(0,\ \mathrm{FAR}_{\text{base}} - S_r(\theta))
$$

***

## Final Result

$$
\mathrm{FAR}(\theta) \in \left[\max\!\left(0,\ \mathrm{FAR}_{\text{base}} - S_r(\theta)\right),\ \mathrm{FAR}_{\text{base}}\right]
$$

