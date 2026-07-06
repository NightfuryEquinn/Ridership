# Chapter 5: Discussion and Conclusion

> **Change log (2026-07-06):** Updated for `--no-x-future` ablation (Section 4.7.1). **Revised §5.1**, **§5.2.1**, **§5.2.3**, **§5.2.4**, **§5.3.2**, **§5.4** (new eighth limitation), **§5.5** (new symmetric-`X_future` direction), **§5.6**. With-`X_future` headline numbers retained where they describe the deployable full system.

---

## 5.1 Introduction

This chapter interprets the empirical findings reported in Chapter 4, situates them within recent traffic and ridership forecasting literature, and draws out their implications for Malaysian public transit planning. The discussion follows the five thematic conclusions of Section 4.10 — now distinguishing the full deployable system (with known-future calendar conditioning) from architecture-only performance measured in the `X_future` ablation (Section 4.7.1): HMT-TSF's superiority over single-paradigm baselines, the dissociation between test accuracy and generalisation in recurrent models, the optimality of the 14-day look-back window, robustness under the MCO structural break, and the interpretability of the learned feature hierarchy. Practical implications, study limitations, future research directions, and the overall conclusion complete the chapter.

---

## 5.2 Discussion of Key Findings

### 5.2.1 Hybrid Multi-Modal Architectures Outperform Single-Paradigm Models

The central finding is that principled fusion of complementary inductive biases outperforms any single architectural paradigm for seven-day Malaysian transit ridership forecasting when the full deployable system is evaluated: HMT-TSF achieved Combined% 85.78 and R² 0.897 at `nomco_lb14` (86.59 / 0.906 feature-reduced), leading Informer (79.13%) by 6.65 Combined% and cutting daily MAE by 28.2% — with the best tuned baseline (Informer, 79.99%) still 5.79 Combined% behind. HMT-TSF alone receives known-future calendar conditioning (`X_future`), legitimately available at deployment but absent from baseline inputs; the July 2026 ablation (Section 4.7.1) shows that disabling `X_future` reduces Combined% to 80.89 (81.23 FR) while retaining a modest +1.76 / +2.10 point lead over Informer and roughly 10% MAE reduction — within the 1.5 Combined% baseline saturation band (Section 4.2). Mean `X_future` contribution across ten configurations is +4.17 / +3.68 Combined%, so roughly half to three-quarters of the headline margin over Informer reflects future-calendar conditioning rather than hybrid architecture alone. Consistent with Chen et al. (2022) and Wu et al. (2025), the top eight baselines compressed within 1.5 Combined%, a saturated regime no single mechanism escaped; HMT-TSF broke saturation through parallel multi-scale TCN, GCN message passing, regime gating, and future-calendar projection fused via a learned softmax gate over five semantic feature groups — implying that marginal returns lie in multi-modal fusion plus task-appropriate input design rather than incremental paradigm refinement, a finding that directly motivates the generalisation and robustness analyses that follow.

### 5.2.2 Test-Set Accuracy Conceals Systematic Overfitting in Recurrent Baselines

Nominal test-set accuracy is an unreliable indicator of deployment readiness because the strongest recurrent baselines achieve their rankings partly through memorisation: Section 4.6 records zero good-fit verdicts for all six LSTM-family and both simple-convolution graph models across twelve configurations (train/validation gaps 3.3×–13.6×; PDR-STGCN 38.61×) despite BiLSTM and TPA-LSTM ranking second and third (R² 0.776). This dissociation echoes Topilin et al. (2025) on memorisation at the expense of adaptability and contrasts Informer's unique 6/6 clean-fit record via ProbSparse regularisation (Song et al., 2024) — had only test Combined% been reported, BiLSTM would appear nearly equivalent to Informer, yet diagnostics place them at opposite ends of the generalisation spectrum — while hyperparameter tuning further eroded fit quality (good-fit rate 26% to 14%) for small, inconsistent test gains. For deployment, fit diagnostics must therefore be evaluated alongside accuracy — a practice that directly motivates the robustness analysis below.

### 5.2.3 The 14-Day Look-Back Window Is Decisively Optimal Under Normal Conditions

For a seven-day forecast horizon under normal operating conditions, a 14-day historical window is optimal and longer context actively harms most architectures: across Section 4.4, lb14 was best for ten of sixteen baseline variants (lb28 six, lb56 none), and HMT-TSF with `X_future` was single-peaked at lb14 (85.78%), declining to 77.68% at lb84 — its only overfit verdict with `X_future` enabled. Without `X_future`, lb14 remains optimal (80.89 / 81.23) but nomco_lb56 becomes overfit for both full and FR variants (Section 4.7.1), suggesting calendar conditioning stabilises intermediate look-backs. STFGNN collapsed from 73.94% to 43.25% (R² −0.112) as longer profiles misled temporal adjacency construction (Chang et al., 2025), whereas MTGNN's near-perfect stability (77.93 / 78.11 / 77.76) confirms Wu et al.'s (2025) account of multi-scale dilated inception; ridership predictability concentrates in two weekly cycles, with slower seasonal structure — dominant in SHAP attribution (Section 4.8) — supplied through calendar features rather than extended history (Ma and Zhang, 2025). HMT-TSF retains its lb14 optimum in both MCO and nomco regimes under both `X_future` settings — which transitions naturally to structural-break robustness.

### 5.2.4 Structural Inductive Biases Determine Robustness Under Distributional Shift

Robustness to structural breaks is governed less by nominal accuracy than by the rigidity or adaptivity of a model's structural assumptions: Section 4.5 documents Δ from HMT-TSF's −3.79 (with `X_future`) to STFGNN's −32.94, with CNN-LSTM sequential and STFGNN falling below naive mean-prediction (R² −0.050 and −0.306) despite strong default-configuration rankings. Disabling `X_future` worsens full-model MCO degradation at lb14 to Δ −5.40 (FR −7.22), indicating that future-known holiday and weekend structure contributes to structural-break transfer, not only to nominal nomco accuracy. Constrained designs proved disproportionately robust — STGCN's fixed Chebyshev graph limited drift (Deng, 2025), Informer's sparse attention engaged the most predictive query-key pairs across the regime boundary (Song et al., 2024) — revealing an expressiveness–stability trade-off in which flexible learners memorised pre-break regularities while constrained or regime-aware designs transferred; HMT-TSF with `X_future` resolved this through Reversible Instance Normalisation and Regime Gating Embedding (81.99% MCO accuracy, clean fit under break), with feature-reduced FR markedly less MCO-robust (−7.85 vs −3.79 at lb14 with `X_future`), indicating that feature redundancy itself contributes to the robustness budget. For Malaysian transit — whose ridership history contains an 85–95% demand collapse — production forecasting systems must be evaluated against, and architecturally prepared for, regime discontinuity.

### 5.2.5 The Learned Feature Hierarchy Is Domain-Consistent and Enables Principled Feature Reduction

The model's learned feature importance is interpretable, theoretically coherent, and actionable: SHAP analysis (Section 4.8) ranks seasonal calendar position highest, aggregate ridership autocorrelation as the most consistent driver across ten configurations, holiday anticipation as the leading discrete-event signal, and unfrozen fuel grades plus monsoon-corridor rainfall as leading external covariates — quantifying calendar-locked cycles, fuel-price mode substitution, and weather-induced accessibility effects consistent with Cui et al. (2025). The practical consequence is HMT-TSF-FR: removing 26 zero- or near-zero-SHAP features improved accuracy at every nomco lookback ≥14 (86.59 Combined%, 53×53 graph), as static spatial descriptors are redundant given time-invariant capacity encoded in historical ridership — though FR's consistent MCO losses (−0.85 to −5.22) reinforce that feature decisions validated under normal conditions must be re-validated under regime shift. Together, these five findings ground the implications discussed below.

---

## 5.3 Implications of the Study

### 5.3.1 Theoretical Implications

This study contributes three generalisable insights to spatio-temporal forecasting research. First, it provides systematic evidence — across sixteen baseline variants, three look-back windows, two distributional regimes, and tuned/untuned conditions — that the features-as-nodes graph formulation transfers graph-based traffic methods (Chang et al., 2025; Deng, 2025; Wu et al., 2025) from sensor-network settings to multivariate national ridership data, while exposing where transfer breaks down (temporal-adjacency construction at long windows). Second, it demonstrates that fit diagnostics constitute a necessary complement to test-set accuracy, materially reordering practical model rankings relative to accuracy-only evaluation. Third, it establishes regime gating combined with reversible instance normalisation as an effective mechanism for structural-break robustness, extending regime-shift observations from attention sparsity (Song et al., 2024) into an explicit, learnable regime representation.

### 5.3.2 Practical Implications for Malaysian Transit

For established networks such as the Klang Valley and Penang systems, the findings support phased integration: HMT-TSF can run in parallel with existing planning processes, with forecast-informed decisions adopted incrementally as walk-forward validation (Section 4.7) accumulates confidence. For developing corridors in Johor and East Malaysia, where new services can be planned around demand forecasts from the outset, the system serves as a proof of concept for demand-responsive network design.

From an operational perspective, accurate seven-day demand forecasts enable frequency setting, rolling-stock allocation, and staffing to match predicted demand. The full deployable system (with `X_future`) achieves 28.2% MAE reduction over the best baseline at nomco_lb14 (33.3% for FR); without `X_future`, MAE reduction falls to approximately 10% — still positive but within the saturated baseline band — so capacity-planning gains should be quoted against the configuration intended for deployment. The 28.2% figure translates into tighter capacity planning: operators can scale service by procuring additional cabins for existing lines rather than committing capital to new infrastructure until forecast demand justifies it. Hassan et al. (2025) demonstrate through a data-driven assessment of Johor Bahru and Penang networks that matching service provision to spatial demand patterns is central to transit efficiency in Malaysian cities; the present MAE reductions operationalise that principle at weekly forecasting horizons by giving planners a quantitative basis for capacity decisions before inefficiency materialises as overcrowding or empty-running. From a sustainability perspective, demand-matched service levels reduce empty-running and associated energy cost; SHAP-detected fuel-price signal provides an initial quantified link between subsidy policy and transit uptake, extending the fuel-elasticity patterns identified in Chapter 3's EDA into a model-attributed causal hierarchy. The features-as-nodes design scales by adding new service lines as feature nodes with recomputed correlation adjacency, and HMT-TSF-FR demonstrates that the feature set can remain lean as the network grows — a property increasingly important as Malaysian operators expand inter-city and regional services under the National Transit Policy framework.

---

## 5.4 Limitations of the Study

Eight limitations qualify the findings. First, the study operates at national daily granularity with service lines as features, so it does not address station-level or intra-day crowd management (Wei et al., 2023; Cui et al., 2025). Second, evaluation uses one national dataset; walk-forward validation supports temporal stability, but cross-network external validity is still untested. Third, the data are batch-historical, meaning collection delays and source errors can be imputed downstream but not prevented upstream.

Fourth, the MCO period provides only one observed structural break, so the K=3 Regime Gating Embedding is calibrated to a single shock type. Fifth, all headline results are single-seed point estimates; without repeated-seed variance, bootstrap intervals, or Diebold-Mariano-style tests, sub-percentage gaps within the 1.5 Combined% saturation band should be treated as indicative rather than conclusive, especially for Informer versus BiLSTM and the no-`X_future` HMT-TSF margin over Informer. Sixth, the study models demand-side unreliability only and excludes supply-side disruptions such as delays, breakdowns, and cancellations because no public Malaysian daily dataset is available. Seventh, the MCO-excluded setting still carries indirect MCO information through `ridership_lag_{7,14,28}` at the start of the post-MCO sample, so no-MCO results are not fully isolated from the removed period. Eighth, HMT-TSF alone receives per-horizon known-future calendar inputs (`X_future`); although Section 4.7.1 quantifies the effect, no baseline was given equivalent future-calendar tensors, so strict architecture comparison remains partially confounded.

---

## 5.5 Future Works

Eight directions follow from the findings and limitations.

**Statistical validation of model rankings.** Re-running headline configurations across multiple seeds, then adding bootstrap intervals or Diebold-Mariano tests, would turn indicative rankings into statistically defensible comparative claims.

**Symmetric known-future calendar conditioning.** To remove the remaining comparison asymmetry, future work should either give top baselines the same per-horizon future calendar tensors used by HMT-TSF or retrain all models under a shared no-`X_future` protocol, then repeat both arms across multiple seeds. This would separate input-design gains from architecture gains and test whether HMT-TSF's smaller no-`X_future` lead over Informer is statistically robust.

**Supply-side reliability data integration.** Adding AVL traces, on-time-performance records, and disruption logs as covariates or parallel targets would connect demand forecasting to service reliability management.

**Real-time data integration.** Ingesting service alerts, trip updates, and vehicle positions would move the system from offline planning support toward operational decision support, including passenger-facing wait-time estimation.

**Data integrity at the source.** Upstream validation, anomaly flagging, and reconciliation across fare-collection channels would reduce imputation burden and strengthen the autocorrelation signals identified as most predictive.

**Spatial and temporal disaggregation.** Extending the features-as-nodes design to station-level nodes with geographic adjacency would enable intra-day forecasting and align the framework with station-level passenger flow research.

**Generalisation to new service lines and networks.** The claimed scalability should be tested on genuinely new lines and new networks, including cold-start performance and transfer-learning warm starts from established corridors.

**Regime modelling beyond the MCO.** Replacing fixed K=3 embeddings with an open-set regime detector that can recognise novel shocks online, such as fare reforms or new-line openings, would turn retrospective robustness into a prospective deployment capability.

---

## 5.6 Conclusion

This study set out to determine whether a purpose-built hybrid architecture could outperform established deep-learning paradigms for seven-day Malaysian transit ridership forecasting, and to characterise the conditions under which each paradigm succeeds or fails. The answer is affirmative and well-bounded. The full deployable HMT-TSF system (with `X_future`) leads all sixteen baseline variants on every headline metric (Combined% 85.78, R² 0.897, MAE 49,897 — 86.59 / 0.906 / 46,323 in its feature-reduced form), is the most robust model under the MCO structural break with `X_future` enabled (Δ −3.79, retaining a study-best 81.99%), and posts a 19-of-20 good-fit record — generalisation quality no baseline family approaches. Without `X_future`, HMT-TSF retains a modest architecture-only lead over Informer (+1.76 / +2.10 Combined%, ~10% MAE reduction), confirming that hybrid design contributes positively but that future-calendar conditioning accounts for the majority of the headline margin (Section 4.7.1).

The comparative evaluation additionally yields durable design guidance: a 14-day look-back window is decisively optimal for a seven-day horizon under normal conditions; test-set accuracy must be read jointly with fit diagnostics, since the strongest recurrent baselines are chronic overfitters; and structural inductive biases, not nominal accuracy, determine survival under distributional shift. SHAP analysis confirms that predictive reasoning — ridership autocorrelation, weekly and holiday calendar structure, and secondary fuel-price and weather signals — is consistent with transit demand theory, and the 53-feature reduced variant shows the system can be made leaner while gaining accuracy under normal conditions. Taken together, these results establish a credible, interpretable foundation for demand-responsive transit planning in Malaysia — immediately as a proof of concept for emerging networks, and through phased integration for established Klang Valley and Penang systems, with real-time data integration as the clearest path from planning tool to operational system.

---

## References

Chang, J., Yin, J., Hao, Y., & Gao, C. (2025). STFDSGCN: Spatio-temporal fusion graph neural network based on dynamic sparse graph convolution GRU for traffic flow forecast. *Sensors*, *25*(11), 3446. https://doi.org/10.3390/s25113446

Chen, W., Yang, Z., Xu, G., & Sun, Y. (2022). Short-term traffic flow prediction based on CNN-BiLSTM with multicomponent information. *Applied Sciences*, *12*(17), 8714. https://doi.org/10.3390/app12178714

Cui, H., Si, B., Chi, D., Li, Y., Li, G., & Chen, Y. (2025). Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data. *PLOS ONE*, *20*(1), e0333094. https://doi.org/10.1371/journal.pone.0333094

Deng, H. (2025). Traffic-forecasting model with spatio-temporal kernel. *Electronics*, *14*(7), 1410. https://doi.org/10.3390/electronics14071410

Ma, X., & Zhang, H. (2025). Time series forecasting method based on multi-scale feature fusion and Autoformer. *Applied Sciences*, *15*(7), 3768. https://doi.org/10.3390/app15073768

Song, Y., Luo, R., Zhou, T., Zhou, C., & Su, R. (2024). Graph attention Informer for long-term traffic flow prediction under the impact of sports events. *Sensors*, *24*(15), 4796. https://doi.org/10.3390/s24154796

Topilin, I., Jiang, J., Feofilova, A., & Beskopylny, N. (2025). Traffic flow prediction via a hybrid CPO-CNN-LSTM-attention architecture. *Smart Cities*, *8*(5), 148. https://doi.org/10.3390/smartcities8050148

Wei, L., Guo, D., Chen, Z., Yang, J., & Feng, T. (2023). Forecasting short-term passenger flow of subway stations based on the temporal pattern attention mechanism and the long short-term memory network. *ISPRS International Journal of Geo-Information*, *12*(1), 25. https://doi.org/10.3390/ijgi12010025

Wu, Z., Liu, X., & Zhang, X. (2025). Multi dynamic temporal representation graph convolutional network for traffic flow prediction. *Scientific Reports*, *15*, 16734. https://doi.org/10.1038/s41598-025-01157-1

Hassan, M., Mahin, H. D., Ahmed, F., Hassan, M. M., Rahaman, A., & Abdullah, M. (2025). Assessing public transit network efficiency and accessibility in Johor Bahru and Penang, Malaysia: A data-driven approach. *Results in Engineering*, *27*, 106126. https://doi.org/10.1016/j.rineng.2025.106126

Amir, N. N., Anuar, N. S., Ismail, B., Azreen, N. A., & Ramli, N. A. (2025). Ridership prediction system for Rapid Bus Kuantan and Penang using multi-feature analysis. *Journal of the Malaysian Institute of Planners*, *23*(6), 332–347. https://doi.org/10.21837/pm.v23i39.1913
