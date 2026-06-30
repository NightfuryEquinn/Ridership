# Chapter 5: Discussion and Conclusion

---

## 5.1 Introduction

This chapter interprets the empirical findings reported in Chapter 4, situates them within recent traffic and ridership forecasting literature, and draws out their implications for Malaysian public transit planning. The discussion follows the five thematic conclusions of Section 4.10: HMT-TSF's superiority over single-paradigm baselines, the dissociation between test accuracy and generalisation in recurrent models, the optimality of the 14-day look-back window, robustness under the MCO structural break, and the interpretability of the learned feature hierarchy. Practical implications, study limitations, future research directions, and the overall conclusion complete the chapter.

---

## 5.2 Discussion of Key Findings

### 5.2.1 Hybrid Multi-Modal Architectures Outperform Single-Paradigm Models

The central finding of this study is that principled fusion of complementary inductive biases outperforms any single architectural paradigm for seven-day Malaysian transit ridership forecasting. HMT-TSF achieved Combined% of 85.78 and R² of 0.897 at `nomco_lb14` (86.59 / 0.906 in its feature-reduced form), leading the best baseline — Informer at 79.13% — by 6.65 Combined% and reducing daily ridership MAE by 28.2%. Even the best tuned baseline (Informer, 79.99%) remained 5.79 Combined% behind. One qualification applies: HMT-TSF alone receives known-future calendar conditioning (`X_future`), so the margin reflects architecture plus input advantage rather than architecture alone.

This result is consistent with a broader pattern in recent spatio-temporal forecasting research. Chen et al. (2022) demonstrate that coupling CNN local pattern extraction with bidirectional recurrence reduces traffic prediction error beyond either mechanism in isolation; Wu et al. (2025) show that combining learned graph adjacency with multi-scale dilated temporal convolution yields more stable predictions than fixed-structure alternatives. The analysis in Section 4.2 sharpens the interpretation: the top eight baselines compressed within a 1.5 Combined% band, indicating a saturated accuracy regime in which no single mechanism — bidirectionality, pattern attention, graph convolution, or sparse attention — could break away. HMT-TSF escaped saturation not by deepening one mechanism but by operating three in parallel and fusing them through a learned softmax gate over five semantic feature groups. The marginal returns to architectural innovation in transit forecasting therefore lie in principled multi-modal fusion rather than incremental refinement of individual paradigms — a finding that directly motivates the generalisation and robustness analyses that follow.

### 5.2.2 Test-Set Accuracy Conceals Systematic Overfitting in Recurrent Baselines

Nominal test-set accuracy is an unreliable indicator of deployment readiness because the strongest recurrent baselines achieve their rankings partly through memorisation. Section 4.6 showed that all six LSTM-family models and both simple-convolution graph models recorded zero good-fit verdicts across all twelve evaluated configurations, with train/validation gap ratios spanning 3.3× to 13.6× for the LSTM family and reaching 38.61× for PDR-STGCN — even though BiLSTM and TPA-LSTM ranked second and third on test accuracy with R² of 0.776.

This dissociation echoes Topilin et al. (2025), who observe that sequential CNN-LSTM architectures memorise characteristic local temporal patterns at the expense of adaptability, and contrasts with the implicit regularisation Song et al. (2024) attribute to ProbSparse attention, which restricts active queries to a logarithmic subset and suppresses fitting of high-frequency noise — a mechanism reflected in Informer's unique 6/6 clean-fit record at base configuration. Had this study reported only test-set Combined%, BiLSTM would appear nearly equivalent to Informer; the diagnostics reveal opposite ends of the generalisation spectrum. The uniformly negative effect of hyperparameter tuning on fit quality (good-fit rate falling from 26% to 14%) reinforces the same caution: added capacity purchased small, inconsistent test gains at a consistent generalisation cost. For deployment, fit diagnostics must be evaluated alongside accuracy — a practice that directly motivates the robustness analysis below.

### 5.2.3 The 14-Day Look-Back Window Is Decisively Optimal Under Normal Conditions

For a seven-day forecast horizon under normal operating conditions, a 14-day historical window is optimal, and longer context actively harms most architectures. Across Section 4.4's sweep, lb14 was optimal for ten of sixteen baseline variants, lb28 for six, and lb56 for none; HMT-TSF's profile was single-peaked at lb14 (85.78%), declining to 77.68% at lb84 — where the 84-day window also produced the model's only overfit verdict.

The most severe case was STFGNN, which collapsed from 73.94% at lb14 to 43.25% at lb56 with R² of −0.112 because longer temporal profiles used to construct its temporal adjacency matrix supply misleading structural priors (Chang et al., 2025). Conversely, MTGNN's near-perfect window stability (77.93 / 78.11 / 77.76) confirms Wu et al.'s (2025) account of multi-scale dilated inception preventing any single temporal scale from dominating. Ridership predictability concentrates in the two most recent weekly cycles: fourteen days span exactly two full commuting periods, while slower seasonal structure — dominant in SHAP attribution (Section 4.8) — is supplied explicitly through calendar features rather than recovered from extended history. This aligns with Ma and Zhang (2025), who report that decomposition-based forecasters perform stably only under sequence lengths matched to dominant signal periodicity. For graph-based baselines, MCO-regime operation still favours longer context on average; HMT-TSF, with explicit future-calendar conditioning and regime gating, retains its lb14 optimum in both regimes — which transitions naturally to structural-break robustness.

### 5.2.4 Structural Inductive Biases Determine Robustness Under Distributional Shift

Robustness to structural breaks is governed less by nominal accuracy than by the rigidity or adaptivity of a model's structural assumptions. Section 4.5 documented a degradation range from HMT-TSF's Δ −3.79 to STFGNN's Δ −32.94; CNN-LSTM sequential and STFGNN fell below naive mean-prediction (R² of −0.050 and −0.306) despite strong default-configuration rankings.

Models with strong structural constraints proved disproportionately robust: STGCN's fixed Chebyshev graph limited prediction drift (Deng, 2025), and Informer's sparse attention engaged the most predictive query-key pairs across the regime boundary (Song et al., 2024). The analysis reveals a consistent expressiveness-stability trade-off: flexible learned representations memorised pre-break regularities that became liabilities after the break, while constrained or explicitly regime-aware designs transferred. HMT-TSF resolved this trade-off through Reversible Instance Normalisation, which removed within-window amplitude drift, and Regime Gating Embedding, which adjusted prediction priors per operational period — retaining both the highest absolute MCO accuracy (81.99%) and clean good-fit records under the break. A complementary observation sharpens the mechanism: the feature-reduced variant, which drops near-constant administered fuel-price columns, is markedly less MCO-robust (−7.85 versus −3.79 at lb14), indicating that under structural breaks feature redundancy itself contributes to the robustness budget. For Malaysian transit — whose ridership history contains an 85–95% demand collapse — production forecasting systems must be evaluated against, and architecturally prepared for, regime discontinuity.

### 5.2.5 The Learned Feature Hierarchy Is Domain-Consistent and Enables Principled Feature Reduction

The model's learned feature importance is interpretable, theoretically coherent, and actionable. SHAP analysis (Section 4.8) shows seasonal calendar position carrying the largest attribution magnitudes, aggregate ridership autocorrelation as the most consistent driver across all ten configurations, and holiday anticipation as the leading discrete-event signal, with unfrozen fuel grades and monsoon-corridor rainfall as leading external covariates. These rankings quantify three established demand mechanisms: calendar-locked seasonal and holiday cycles, secondary mode-substitution through fuel prices, and weather-induced accessibility effects — consistent with Cui et al. (2025), who document multi-source demand drivers for urban rail passenger flow.

The practical consequence was HMT-TSF-FR: removing 26 zero- or near-zero-SHAP features improved accuracy at every nomco lookback ≥14, raising the headline to 86.59 Combined% while shrinking the graph adjacency from 79×79 to 53×53. Static spatial descriptors were redundant not because spatial structure is irrelevant, but because time-invariant capacity is implicitly encoded in historical ridership given sufficient look-back. The caveat — FR's consistent losses under MCO (−0.85 to −5.22 across look-backs) — reinforces that feature decisions validated under normal conditions must be re-validated under regime shift. Together, these five findings ground the implications discussed below.

---

## 5.3 Implications of the Study

### 5.3.1 Theoretical Implications

This study contributes three generalisable insights to spatio-temporal forecasting research. First, it provides systematic evidence — across sixteen baseline variants, three look-back windows, two distributional regimes, and tuned/untuned conditions — that the features-as-nodes graph formulation transfers graph-based traffic methods (Chang et al., 2025; Deng, 2025; Wu et al., 2025) from sensor-network settings to multivariate national ridership data, while exposing where transfer breaks down (temporal-adjacency construction at long windows). Second, it demonstrates that fit diagnostics constitute a necessary complement to test-set accuracy, materially reordering practical model rankings relative to accuracy-only evaluation. Third, it establishes regime gating combined with reversible instance normalisation as an effective mechanism for structural-break robustness, extending regime-shift observations from attention sparsity (Song et al., 2024) into an explicit, learnable regime representation.

### 5.3.2 Practical Implications for Malaysian Transit

For established networks such as the Klang Valley and Penang systems, the findings support phased integration: HMT-TSF can run in parallel with existing planning processes, with forecast-informed decisions adopted incrementally as walk-forward validation (Section 4.7.3) accumulates confidence. For developing corridors in Johor and East Malaysia, where new services can be planned around demand forecasts from the outset, the system serves as a proof of concept for demand-responsive network design.

From an operational perspective, accurate seven-day demand forecasts enable frequency setting, rolling-stock allocation, and staffing to match predicted demand. The 28.2% MAE reduction over the best baseline (33.3% for FR) translates into tighter capacity planning: operators can scale service by procuring additional cabins for existing lines rather than committing capital to new infrastructure until forecast demand justifies it. Hassan et al. (2025) demonstrate through a data-driven assessment of Johor Bahru and Penang networks that matching service provision to spatial demand patterns is central to transit efficiency in Malaysian cities; the present MAE reductions operationalise that principle at weekly forecasting horizons by giving planners a quantitative basis for capacity decisions before inefficiency materialises as overcrowding or empty-running. From a sustainability perspective, demand-matched service levels reduce empty-running and associated energy cost; SHAP-detected fuel-price signal provides an initial quantified link between subsidy policy and transit uptake, extending the fuel-elasticity patterns identified in Chapter 3's EDA into a model-attributed causal hierarchy. The features-as-nodes design scales by adding new service lines as feature nodes with recomputed correlation adjacency, and HMT-TSF-FR demonstrates that the feature set can remain lean as the network grows — a property increasingly important as Malaysian operators expand inter-city and regional services under the National Transit Policy framework.

---

## 5.4 Limitations of the Study

Seven limitations qualify the findings. First, the study forecasts at national daily granularity with service lines as features; it does not produce station-level or intra-day forecasts, limiting direct applicability to platform-level crowd management addressed by station-level studies (Wei et al., 2023; Cui et al., 2025). Second, evaluation relies on a single national dataset; walk-forward validation confirms temporal stability, but cross-network external validity remains untested. Third, the dataset is batch-historical: kiosk and vehicle records are ingested after the fact, and collection delays introduce integrity risks the pipeline imputes but cannot eliminate at source.

Fourth, the MCO period provides only one observed structural break; the Regime Gating Embedding's K=3 regimes are calibrated to that specific discontinuity, and behaviour under a future, differently shaped shock is extrapolative. Fifth, all reported results are single-run point estimates obtained with a fixed random seed; no repeated-seed variance, bootstrap confidence intervals, or forecast-comparison significance tests (e.g., Diebold–Mariano) were computed, so the statistical separability of closely ranked models is unestablished — sub-percentage-point ranking differences among the top-eight baselines in Table 4.2.1, which span only 1.5 Combined%, should be treated as indicative rather than definitive. This limitation is particularly salient for the Informer–BiLSTM comparison (79.13 versus 79.04), where fit diagnostics rather than test metrics provide the decisive deployment distinction. Sixth, the study addresses unreliability from the demand side only: it forecasts ridership to enable proactive capacity decisions but does not model supply-side reliability events (delays, breakdowns, cancellations), for which no public Malaysian dataset exists at daily granularity. Seventh, autoregressive lag features in the MCO-excluded dataset bridge the removed window: at the start of the post-MCO period, `ridership_lag_{7,14,28}` reference dates inside 2020-03-18 – 2021-12-31, so the nomco condition retains indirect MCO information through its lag channel — a deliberate design choice that should be borne in mind when interpreting no-MCO results.

---

## 5.5 Future Works

Seven directions follow from the findings and limitations.

**Statistical validation of model rankings.** Re-running headline configurations across multiple random seeds to attach variance estimates, complemented by Diebold–Mariano tests or bootstrap confidence intervals over test-window errors, would convert indicative rankings into statistically defensible claims — a prerequisite for journal publication of the comparative results.

**Supply-side reliability data integration.** Incorporating automatic vehicle location traces, on-time-performance records, and disruption logs as covariates and forecast targets would close the loop between demand forecasting and unreliability mitigation, extending the present analytics layer toward predicting delay-risk events directly.

**Real-time data integration.** Ingesting service alerts, trip updates, and vehicle positions would shift the system from offline planning support toward operational decision support, including passenger-facing waiting-time estimation. Informer-style sparse attention embedded in the architecture is computationally suited to longer, finer-grained sequences that real-time data implies (Song et al., 2024).

**Data integrity at the source.** Validation at ingestion, automated anomaly flagging, and reconciliation across fare-collection channels would reduce imputation burden and strengthen the autocorrelation signals SHAP identified as dominant predictive drivers. Multi-source fusion approaches described by Cui et al. (2025) offer a template for reconciling heterogeneous collection channels.

**Spatial and temporal disaggregation.** Extending the features-as-nodes formulation to station-level nodes with geographic adjacency would connect this work to station-level passenger flow literature (Wei et al., 2023) and enable intra-day forecasting, with MTGNN-style learned adjacency (Wu et al., 2025) and SHAP-based feature reduction keeping graph growth tractable.

**Generalisation to new service lines and networks.** The scalability claim should be validated by introducing a genuinely new line and measuring cold-start forecast quality as its history accumulates, including transfer-learning warm starts from established Klang Valley corridors.

**Regime modelling beyond the MCO.** Generalising the regime mechanism from fixed K=3 embeddings to an open-set regime detector capable of recognising novel structural breaks online — fuel subsidy reform, fare restructuring, or new-line openings — would convert the robustness demonstrated retrospectively in Section 4.5 into a prospective operational property. Amir et al. (2025) demonstrate that ridership prediction systems for regional Malaysian transit can benefit from adaptive model updating as new operational data arrive; integrating a similar online adaptation layer atop HMT-TSF's regime gating would align the architecture with that deployment paradigm.

---

## 5.6 Conclusion

This study set out to determine whether a purpose-built hybrid architecture could outperform established deep-learning paradigms for seven-day Malaysian transit ridership forecasting, and to characterise the conditions under which each paradigm succeeds or fails. The answer is affirmative and well-bounded. HMT-TSF leads all sixteen baseline variants on every headline metric (Combined% 85.78, R² 0.897, MAE 49,897 — 86.59 / 0.906 / 46,323 in its feature-reduced form), is the most robust model under the MCO structural break (Δ −3.79, retaining a study-best 81.99%), and posts a 19-of-20 good-fit record — generalisation quality no baseline family approaches.

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
