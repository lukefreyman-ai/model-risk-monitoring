# Model Validation Report - Retail Application Scorecard (GMSC)

**Prepared by:** Luke Freyman, Analyst, Model Risk (first-year)   **Date:** 2026-09-13
**Model:** WOE / logistic application scorecard, 7 characteristics, points scaled 600 @ 50:1 odds, PDO 20
**Scope:** ongoing-monitoring review over five monitoring windows (Q1-Q5) against the development sample (dev)
**Overall rating:** **RED** - the model remains fit for purpose as a ranking tool; population stability breaches require a cut-off review.

## 1. Summary

The scorecard still separates good and bad borrowers as well as it did at development (KS 0.555 at dev,
0.518 in the latest window; Gini 0.716 -> 0.666, a 7.1 % relative drop, inside the 10 % tolerance).
Portfolio-level calibration also holds: actual minus predicted default rate is within 0.48 pp in every window.

What has changed is **who is being scored**. The score PSI rises from 0.044 (Q3) to 0.171 (Q4, amber) and
0.356 (Q5, red). The characteristic stability table attributes the shift to revolving utilisation (CSI 0.403 in Q5)
and, secondarily, age. Bad rates rose from 6.78% (dev) to 12.35% (Q5), and the model's average predicted PD moved with them
(6.78% -> 11.87%), which is why calibration is still green: the model is correctly predicting that a riskier
intake is riskier.

## 2. What I checked and how

| check | method | result |
|---|---|---|
| Population stability | PSI of scores vs. dev deciles, per window | Q4 0.171 amber, Q5 0.356 red |
| Characteristic stability | CSI per scorecard bin, per characteristic | util 0.403 (Q5), age 0.103 (Q5); all others < 0.10 |
| Discrimination | KS and Gini per window vs. dev | KS 0.517-0.555; max Gini drop 7.1 % |
| Calibration | decile actual vs. predicted; portfolio gap; H-L | gap <= 0.48 pp; H-L p-values not relied on at this n |
| Challenger | gradient boosting on same dev window | +2.8 Gini pts on average, decays faster (14 % vs 7 %) |

The thresholds (PSI 0.10 / 0.25, CSI 0.10, KS >= 0.30, Gini drop < 10 %, calibration gap <= 2 pp) are the industry
rules of thumb in `config.yaml`; they are not calibrated to this portfolio's economics.

## 3. Findings

1. **HIGH** - Score PSI reaches 0.356 in Q5 (> 0.25): the scored population has shifted materially from development. Investigate acquisition mix before trusting cut-offs.
2. **MEDIUM** - Characteristic drift: 3 feature-window pairs breach CSI 0.1; largest is util in Q5 (CSI 0.403). These characteristics drive the score shift.
3. **LOW** - Champion KS stays above 0.3 (range 0.517-0.555); Gini relative drop vs development at most 7.1 %.
4. **LOW** - Portfolio-level calibration holds: |actual - predicted| default rate <= 2 pp in every window.
5. **INFO** - Challenger (gradient boosting) Gini exceeds the champion by 2.8 pts on average (+0.9 to +8.1) and its relative decay across windows is 14.4 % vs 7.1 % for the champion. Lift alone does not justify replacement: the challenger is not reason-code explainable and would need a full validation cycle.

## 4. Recommended actions

1. **Model owner (within 30 days):** explain the change in acquisition mix behind the utilisation shift (new
   channel? marketing? policy change?) and confirm it is intended.
2. **Credit policy:** re-run the approval-rate and expected-loss analysis at the current cut-off on the Q5 population.
   With a higher-risk intake the same cut-off approves a higher share of bads even though the score is accurate.
3. **Model risk:** keep the model in production; move monitoring of PSI/CSI to monthly until PSI returns below 0.25
   for two consecutive periods. No recalibration is required while the portfolio gap stays inside 2 pp.
4. **No action on the challenger.** Its lift does not offset the loss of reason-code explainability and it decays
   faster under the same drift.

## 5. Limitations of this review

- The five windows are a documented simulation on one snapshot (see README). Direction and magnitude of drift are
  by construction; the metrics and the way they are read are what this report demonstrates.
- Outcomes are known immediately here; in production a 24-month window would delay the KS/Gini/calibration checks.
- Override analysis, reject inference and fair-lending impact were out of scope and would be required for a full validation.

## 6. Things I would want a senior reviewer to check

- Whether 0.10 / 0.25 PSI bands are appropriate for a portfolio of this size, or whether a statistical test on
  decile shares would be more defensible.
- Whether the smoothing in the WOE calculation (adding 0.5 to each bin) is acceptable under the bank's standards.
- Whether the "cut-off review" recommendation should be a hard requirement rather than a suggestion given a red PSI.

*All figures come from `results/performance.csv`, `results/csi.csv` and `results/metrics.json`, produced by `python run.py`.*
