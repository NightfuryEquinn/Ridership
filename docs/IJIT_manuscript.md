# Hybrid Multi-Scale Temporal Spatio-Feature Forecasting of Daily Public Transit Ridership in Malaysia under Structural Break

*Author 1 1 & Author 2 2*

1 Affiliation 1 (i.e. institution, (department), city, (state), country)
2 Affiliation 2 (i.e. institution, (department), city, (state), country)

Corresponding author: [NAME]
Email: [EMAIL]

## Abstract

Predicting daily public transit ridership in Malaysia is challenged by complex spatiotemporal variables (e.g., population density, weather, and socio-economic activities) and extreme data anomalies driven by the Movement Control Order (MCO). This study identifies key ridership factors and introduces the Hybrid Multi-Scale Temporal Spatio-Feature (HMT-TSF) Forecaster. Utilizing a robust deep learning (DL) framework integrating General Transit Feed Specification (GTFS) schedules, historical weather, and Points of Interest (POIs), the proposed model is benchmarked against state-of-the-art architectures, including LSTM, BiLSTM, STGCN, and Informer. To evaluate pandemic-driven noise and predictive bias, the analysis is executed across two distinct dataset segmentations: "With MCO" (2019–2025) and "Without MCO" (2022–2025), utilizing MAE, RMSE, and R2 evaluation metrics. Results indicate that incorporating exogenous variables significantly enhances DL predictive power over classical statistical models like ARIMA. The proposed HMT-TSF consistently outperforms baseline models across all lookback windows and dataset segmentations. Critically, training architectures on the MCO-inclusive dataset introduces severe predictive bias. Conversely, strategic data segmentation isolates these anomalies, enabling the model to accurately capture peak demands and standard urban mobility patterns. This paper provides transport authorities with an effective, data-driven methodology to optimize scheduling and close the infrastructure-to-ridership gap.

## Keywords

Public Transit Reliability, Deep Learning Analytics, Ridership Forecasting, Urban Mobility Pattern Analysis

## 1 Introduction

Prioritising public transport over private vehicles improves national economic performance while advancing environmental sustainability [1], aligning with the United Nations Sustainable Development Goals. For Malaysia, a developing nation undergoing rapid urbanisation, an efficient public transit system is a prerequisite for developed-country status [2]. Yet unreliable service prediction, driven by weather, socio-economic activity, and geopolitical events, pushes citizens back toward private vehicles, accelerating congestion and air pollution [3]. Reliable transit therefore addresses environmental and social challenges simultaneously, improving quality of life and mobility across metropolitan and rural areas [2].

Despite substantial government investment in Mass Rapid Transit (MRT) Putrajaya, Shah Alam, and the planned MRT Circle Line, national public transport usage remains near 20%, far short of the 40% target set by the National Transport Policy 2019–2030 [3]. Prolonged travel times, untimely delays, and limited accessibility are the primary causes of passenger dissatisfaction in Kuala Lumpur [4]. These deficiencies compound the spatiotemporal complexity that conventional statistical models such as ARIMA cannot resolve, establishing a clear need for a predictive model that forecasts ridership accurately, consistently, and reliably.

Advances in big data analytics and deep learning (DL) offer a route to modelling the intricate spatiotemporal relationships within transit networks [5]. Unlike ARIMA, architectures such as LSTM and Graph Convolutional Networks (GCN) integrate multiple heterogeneous sources: General Transit Feed Specification (GTFS) schedules, historical weather, and Points of Interest (POIs), to capture multidimensional dependencies [5, 6]. Applying state-of-the-art (SOTA) DL architectures addresses Malaysia's present reliance on conventional models.

A critical barrier lies in the historical record itself. Malaysia's Movement Control Order (MCO), enforced from 18 March 2020 to 31 December 2021, produced an 85–90% passenger collapse, generating noisy patterns unrepresentative of normal operations [3]. Training on such data introduces severe predictive bias, because pandemic-era containment behaviour differs fundamentally from typical urban mobility [7]. Models fitted to contaminated series subsequently fail to recognise true ridership peaks once daily life normalises [8]. This study therefore compares two dataset segmentations, one including and one excluding the MCO period, to determine empirically whether this anomaly should be treated as noise before training.

Three research questions follow. First, which spatiotemporal factors most strongly affect Malaysian ridership? Second, how accurately and consistently do spatiotemporal, graph-based, and attention-based DL architectures perform relative to one another and to a purpose-built hybrid? Third, how does the inclusion of MCO-period anomalies distort predictive accuracy and bias? The corresponding objectives are to identify and categorise the dominant spatiotemporal factors through systematic analysis of historical performance data; to develop the Hybrid Multi-Scale Temporal Spatio-Feature (HMT-TSF) Forecaster; to benchmark it against spatiotemporal, graph-based, and attention-based baselines using a composite accuracy index and R²; and to quantify the effect of extreme outlier data by evaluating every model under both dataset segmentations.

The study is scoped to national daily granularity across twelve Malaysian service lines (LRT, MRT, Monorail, KTM, and RapidBus) over 2019–2025, with a seven-day forecast horizon. It contributes a comparative analysis of fifteen architectures across multiple look-back windows and both MCO conditions; a predictive model that outperforms SOTA baselines under every configuration, accompanied by a feature-reduced variant derived from SHAP importance ranking; and a command-line inference tool that predicts seven-day ridership for twelve transit networks from historical distribution percentages.

## 2 Literature Review



### 2.1 Public Transport Reliability and Demand Dynamics

Reliability is a multi-dimensional construct combining objective operational performance with subjective passenger experience. Travel time reliability is defined as the gap between expected and actual travel times, with route length, signalised intersections, and departure delays as the primary structural causes of unreliability [9]. Satisfaction within the Klang Valley Light Rail Transit (LRT) system, however, is mediated by "soft" factors: accurate real-time information and station amenity quality [10]. Agencies traditionally prioritise hard metrics such as headway adherence, yet perceived reliability is shaped by the immediate physical environment and by transparent communication. Chronic unreliability then generates a demand-erosion cycle: passengers allocate buffer time against systemic delay, and that temporal penalty outweighs the fare saving, producing stagnant modal share even as fuel costs rise [3, 4]. High private vehicle ownership in states such as Sarawak persists precisely because bus service tangibility and responsiveness are perceived as inadequate [11]. Physical network expansion alone therefore cannot shift commuter habits; infrastructure must be paired with localised, data-driven operational optimisation.

### 2.2 Transit Data Ecosystems and Anomalies

Modern transit data ecosystems combine endogenous operational streams with exogenous environmental inputs. Core endogenous sources are Automated Fare Collection systems for ridership, Automated Vehicle Location for GPS tracking, Automated Passenger Counting sensors, and static GTFS schedules [6, 12]; critical exogenous variables are real-time weather, road traffic, POIs derived from land-use profiles, and regional socio-economic demographics [3]. These raw streams are fragmented and noisy. Agencies frequently collect disjointed data that impedes system-wide interpretation [13], while sensor omissions, GPS telemetry errors, formatting inconsistencies across legacy systems, and extreme outliers are pervasive [14]. Beyond ordinary noise, spatiotemporal profiles are vulnerable to extreme structural anomalies: the MCO caused an abrupt 85–90% ridership drop across all networks [15]. Networks ingesting uncorrected outlier sequences overfit to anomalous troughs and fail to recognise normal peaks once operations resume [7]. Statistical-threshold anomaly detection performs poorly in high-dimensional spatiotemporal settings, motivating strategic dataset segmentation as adopted here.

### 2.3 Evolution from Traditional to Deep Learning Models

Classical frameworks — ARIMA, Historical Average, Vector Autoregression, Support Vector Regression, and Random Forest Regression — offer interpretability but demand strict stationarity and rigid distributional assumptions that conflict with highly fluctuating traffic data [16, 17]. DL architectures circumvent these restrictions by learning hierarchical non-linear spatiotemporal features directly. Convolutional Neural Networks capture spatial correlation over grid structures, recurrent networks such as LSTM track sequential dependency, GCNs model non-Euclidean topology, and attention mechanisms weight feature significance dynamically. A multi-modal geospatial-temporal LSTM achieved a 15% MAPE reduction and 20% RMSE reduction over traditional methods and standard DL baselines [14]. Verifying such claims requires standardised multi-metric benchmarking: comprehensive surveys adopt MAE, RMSE, and MAPE as primary metrics [18], and pairing relative with absolute indices is essential because a single metric can mask critical performance flaws [19]. A rigorous framework must also combine comparison against established baselines with sensitivity and robustness testing across anomalous contexts.

### 2.4 AI-Driven Operational Strategies and Sustainability

Moving from reactive to predictive management enables proactive scheduling and optimised resource allocation [20]. An LSTM encoder-decoder outperformed rule-based business systems by 18–23% in real-time delay forecasting [17], and AI-powered planning platforms maximise schedule efficiency and fleet deployment by anticipating demand shifts [21, 22]. Traditional dispatch adjusts frequency only after disruption has occurred, whereas DL models predict passenger surges in advance, permitting dynamic service intervals and coach allocation. Adoption is nonetheless obstructed by fragmented corporate data structures and organisational silos [13], high upfront implementation costs, data privacy compliance, and stakeholder resistance [23, 24]. Where these barriers are overcome, the environmental payoff is direct: transforming urban mobility is central to lowering transportation emissions [25], and AI-driven regularisation of headways minimises fuel waste, emergency maintenance, and operational inefficiency [26]. Reduced waiting times accelerate the modal shift from private vehicles, lowering urban emissions and improving air quality [22].

### 2.5 Research Gap

The literature lacks a hybrid ridership forecaster that integrates multi-scale spatiotemporal structure while explicitly isolating predictive bias arising from severe structural breaks. Recent studies apply DL to transit ridership [3, 27] but rely on single-paradigm architectures that do not jointly model multi-scale temporal dynamics and multi-dimensional spatial interaction. Pandemic-era anomalies further confound training [15], yet no Malaysian study systematically quantifies how such breaks distort graph-based and attention-based SOTA models. Excluding the MCO period disrupts series continuity; retaining it uncorrected introduces bias. Without a unified benchmark spanning sequential, graph, and attention paradigms under both MCO conditions, operators have no evidence base for selecting a robust forecaster. This study addresses that gap through HMT-TSF, enabling a shift from reactive to proactive, data-driven transit management.

## 3 Methodology



### 3.1 Data Collection and Understanding

Eight heterogeneous sources are integrated into a single spatiotemporal forecasting dataset: daily ridership, GTFS network supply, GADM administrative boundaries, retail fuel prices, public and school holidays, state-level rainfall, gridded population density, and OSM point-of-interest counts.

Daily ridership counts for twelve Malaysian service lines (LRT, MRT, Monorail, KTM, and RapidBus) and static GTFS data were obtained from data.gov.my; combining GTFS with spatial layers supports granular assessment of network efficiency and accessibility [28]. Real-time GTFS streams are excluded because the national portal supplies only vehicle positions, and deriving metrics from raw coordinates would degrade inference efficiency. GADM level-1 boundaries were standardised to WGS-84 to construct binary and border-length-weighted adjacency matrices, since graph-based spatial structures capture regional heterogeneity in demand [29]. Weekly retail fuel prices (RON95, RON97, diesel) were forward-filled to a daily index, and Malaysian public and school holidays for 2019–2026 were extracted from timeanddate.com; fuel price increases correlate positively with transit ridership in developing countries [30]. Gridded population density, seven-category POI counts within a 500 m stop radius, and sub-national daily rainfall were added as environmental covariates, because non-linear ridership relationships require joint semantic and spatial encoding [31]. The full series (1 January 2019 – 31 December 2025) was split into MCO-inclusive and MCO-exclusive configurations to measure how atypical operational periods affect generalisation.

### 3.2 Feature Engineering

Eight independent preprocessing scripts transform the raw formats into a standardised daily matrix. Percent-change fuel features are retained alongside level series to capture short-run economic response [32], and cyclic sine/cosine encoding avoids ordinal discontinuity at calendar boundaries [33]. All variables were aligned to consecutive daily indexes for both research periods under explicit merge and null-handling rules, with interpolation capped at seven days. Autoregressive lags at 7, 14, and 28 days were computed from the full historical ridership series before merging [34, 35], yielding a 79-feature matrix organised into five semantic groups: targets, temporal, external, lag, and static.

The aligned matrix was converted to sliding-window tensors with look-back Tin and a fixed Tout = 7-day horizon, under a 70%/15%/15% chronological train/validation/test split. MinMax scaling was fitted on training data only, preventing information leakage across the split boundary [33].

### 3.3 Exploratory Data Analysis

Four findings from the exploratory analysis directly determined design decisions in the proposed architecture.

Figure 1. Total daily transit volume, 2019 to mid-2026, with six detected MCO structural shifts.

The MCO enforced between 18 March 2020 and 31 December 2021 is the largest disturbance in the series, spanning roughly a quarter of the observations. Figure 1 shows volume falling by 85–90% at onset, with six structural shifts delimiting four behavioural phases. The collapse is near-instantaneous while the return is staged, indicating recovery governed by the sequential release of activities rather than one policy date. Recovery is also mode-unequal: rail returned to a lower fraction of its own baseline than bus, because rail serves fixed corridors feeding the Kuala Lumpur central business district whose office commuters retained the option of remote work, so part of that demand was removed rather than displaced, whereas bus carries more trips that cannot be performed from home [8]. The mapping from calendar and weather inputs to ridership is therefore not the same function before and after the break, with both mean level and variance shifting — motivating the dual MCO-inclusive / MCO-exclusive dataset construction and the regime-aware components of Section 3.5 [3].

Figure 2. Ridership response under heavy-rain versus normal conditions.

The rainfall-ridership relationship is not monotonic. Figure 2 shows that heavy-rain days do not carry the reduction a simple discouragement effect would predict; pooling every day returns an overall correlation of only r = 0.14. Two opposing mechanisms operate at different intensities. At low intensity, suppression prevails as discretionary trips are deferred and the disutility of the walking leg rises. Above roughly the 75th percentile, substitution takes over: surface roads flood, driving becomes hazardous, and car users transfer onto grade-separated rail, while buses gain far less because they share the flooded carriageway [38]. The weak pooled coefficient reflects cancellation between these effects, not an unimportant covariate. Three design decisions follow. Rainfall is kept continuous rather than a binary wet-day flag, since a flag can express only one sign of effect. All twelve services are forecast alongside the national total, so a response positive for rail and negative for bus need not collapse into one parameter [27]. Rainfall is kept at state level, because east-coast states record several times the extreme-day count of the Klang Valley while generating little of the national ridership, so a national mean would attenuate or reverse the relationship [36, 37].

Figure 3. Holiday-period demand showing pre-dip and post-surge patterns.

Holiday effects extend well beyond the holiday itself. Figure 3 isolates the holiday window, where ridership falls by approximately 18% one to two days before a public holiday and rises sharply on the first working day afterwards. Dip and spike are two halves of one displacement, and the asymmetry is informative: departure can be brought forward at the traveller's discretion whereas employment resumes on a fixed date, so the outflow spreads across two days while the return compresses into one. The driver is the *balik kampung* practice — major festivals trigger migration out of the Klang Valley, with commuters taking leave on bracketing days to extend the break. Short holidays produce the deepest dip, counter-intuitive under a proportional model but natural under displacement, since a short break forces the entire outflow and return into a narrow window. A binary flag would capture only the trough and force the model to treat flanking days as noise, so four directional counters — days until and days since each holiday type — supply a monotone ramp on either side, with public and academic holidays kept separate because merging a sharp one-day commuter drop with a multi-week academic break weakens both signals [34, 40]. Because these quantities are deterministic for any future date, this finding motivates the known-future conditioning branch of Section 3.5.

Figure 4. Correlation matrix of key demand-forecasting indicators and dominant predictor coefficients.

Figure 4 pairs a 6 × 6 correlation matrix of the key inputs with the standardised coefficients of the dominant predictors. Ridership lags at 7, 14, and 28 days carry the largest weights, and a principal component analysis of the same matrix attributes approximately 72% of total variance to a first component loading primarily on those lags. Lag dominance reflects the demand process rather than a modelling artefact: transit trips are generated by recurring work and school commitments that reproduce weekly, so the same weekday one week earlier is a near-sufficient statistic under normal conditions. The informative part is where that baseline fails — residuals from a naïve seven-day-lag predictor are largest around public holidays and at the regime boundary, precisely where calendar, weather, and fuel features earn their place. By contrast, static network and demographic features return near-zero linear correlation with the targets and with one another. Population counts, GTFS stop statistics, POIs, and administrative areas are constant across the window by construction, and a column with no temporal variance has essentially no linear correlation with anything that varies, regardless of causal importance; such features can act only as fixed scalings interacting multiplicatively with time-varying drivers, which linear screening cannot detect [31]. This is evidence of an inadequate measure, not of irrelevant features, and is why the reduction in Section 4.4 uses SHAP attribution rather than correlation ranking [41].

Two further findings are stated briefly. Rainfall peaks between November and January under the Northeast Monsoon, concentrated in east-coast states sheltered by the Titiwangsa range; this sets the seven-day interpolation cap, since a monsoon event lasts a few days and seven days bridges one event without fabricating a wet spell [37]. Fuel prices are revised weekly and held constant between announcements, making the level series a staircase whose apparent +0.35 post-MCO correlation with ridership is shared upward drift rather than elasticity; the genuine mechanism is threshold-and-lag, with the response to price *changes* peaking one to three days later, so signed weekly change series are added alongside the levels [32, 39]. The subsidised RON95 variants introduced from mid-2023 are administratively frozen and carry almost no variance, which is why they later fall out under SHAP-guided reduction.

### 3.4 Baseline and Comparative Models

Fourteen baseline DL models form a progressive comparison chain isolating each architectural contribution, from sequential encoding through graph propagation to series decomposition.

The spatiotemporal family comprises LSTM, the foundational sequential baseline whose gated memory captures daily and weekly commuter cycles without graph or attention bias [42]; BiLSTM, testing bidirectional encoding over the look-back window [43]; TPA-LSTM, testing pattern-level attention over LSTM hidden states for multi-scale periodicity [44]; CNN-LSTM, testing 1-D convolutional local feature extraction before recurrent encoding across sequential, parallel, and augmented fusion modes [45]; CNN-BiLSTM, combining local extraction with bidirectional context [46]; and ST-LSTM, a bridge model decoupling temporal (LSTM) and spatial (MLP) streams [41].

The graph family comprises STGCN, with fixed Pearson inter-feature adjacency and Chebyshev graph convolution [47]; PDR-STGCN, combining dynamic attention adjacency with periodicity-aware two-channel input [48]; STSGCN, fusing spatial and temporal graph operations in a single synchronous 3N × 3N adjacency [49]; STFGNN, using dual graphs for cross-feature co-variation and intra-window temporal-profile similarity fused by a learnable gate [50]; and MTGNN, learning an asymmetric adjacency end-to-end from node embeddings [51]. The attention family comprises ASTGCN, adding learnable spatial and temporal attention over the STGCN backbone [41]; Autoformer, applying series decomposition with FFT-based autocorrelation attention [52]; and Informer, a ProbSparse efficient transformer with a generative multi-step decoder [53]. At Tin = 14, ProbSparse degenerates to near-full attention, ensuring fair comparison with Autoformer.

### 3.5 Hybrid Multi-Scale Temporal Spatio-Feature Forecaster

*Hybrid* is used here in its established architectural sense: a model is hybrid when it integrates two or more distinct modelling paradigms, each carrying a different inductive bias, inside a single end-to-end trainable predictor. This is distinct from ensemble hybridity, which averages outputs of separately trained models, and from pre-processing hybridity, which chains a statistical decomposition to a neural predictor [33]. Every primitive used is prior art and attributed as such; what is claimed is the composition — the selection of paradigms, the routing between them, and the gating that combines their representations.

HMT-TSF addresses three limitations shared by all fourteen baselines. First, no model captures all three types of structure simultaneously: LSTM-family models handle temporal sequences but carry no graph structure and no regime awareness; graph models propagate inter-feature correlation but are rigid under distributional shift; attention models decompose periodic signals but cannot propagate feature-to-feature messages. Second, no model explicitly handles the MCO regime shift — every graph baseline uses a static Pearson graph computed on training data, which the structural break invalidates, while static normalisation applies fixed-scale transformations across fundamentally different regimes. Third, no model uses domain-aware feature grouping; all treat the 79 features as a flat vector, giving equal initial weight to structurally disparate sources.

Each limitation is answered by a named, separately ablatable module. The 79 inputs are organised into the five semantic groups of Section 3.2, each independently projected to a common embedding dimension so internal structure is learned without cross-group interference, then concatenated and passed through a learned softmax gate that weights each group per timestep — on holiday dates the temporal group receives higher weight, on normal weekdays the target-context and lag groups dominate. The gated output is projected with LayerNorm to a unified dmodel representation on which three parallel encoders operate.

Ahead of group embedding, Reversible Instance Normalisation (RevIN) normalises the raw input by each sample's per-feature mean and variance under learnable affine parameters, reducing the within-batch distribution shift caused by the regime change; without it, a minibatch spanning multiple regimes produces gradient conflicts that impair convergence [54]. RevIN denormalises predictions after the forecast heads so the loss is measured in the original scaled space.

The first encoder, a Multi-Scale Temporal Convolution Network, applies causal dilated convolutions at up to three scales — the full window with exponentially increasing dilation, the first half for medium-range structure, and the first quarter for long-range baselines — with WaveNet-style gated activations, DropPath, and learned attention pooling across active scales. The second, a Spatial Graph Encoder, compresses the time dimension by learned temporal attention rather than using the raw sequence as node features, then propagates signals through a two-layer GCN over the Pearson-correlation adjacency (|corr| ≥ 0.1, matching the STGCN and ASTGCN construction); decoupling relational encoding from temporal dynamics is what distinguishes the branch from graph baselines [29, 47]. The third, a Regime Gating Embedding, maintains three learnable vectors for pre-COVID, MCO, and post-MCO operation and projects the mean-pooled input to three soft logits, blending adjacent regime characteristics during transitional periods without hard chronological boundaries and requiring no date metadata at inference.

The three encoder outputs are concatenated and passed through a Squeeze-and-Excitation bottleneck gate, calibrating which stream contributes most per input context while avoiding an expensive 3d × 3d weight matrix [18]. Two forecast heads follow: a primary head with highway residual and LayerNorm, and a boosting head whose scalar gate is initialised so its contribution is suppressed early and activates as training converges. The custom loss penalises accuracy and smoothness jointly, weighting the one-day-ahead step at 1.0 and decaying to approximately 0.53 at seven days. Finally, the sixteen temporal features are deterministic for any future calendar date and are pre-computed for the forecast steps of each window as Xfuture; a per-step zero-initialised MLP learns an additive correction from this known-future context, letting the model anticipate weekend dips and holiday effects inside the horizon rather than extrapolating them — the direct architectural consequence of the displacement pattern in Figure 3.

Figure 5. HMT-TSF architecture organised by the three types of structural components.

The graph and regime streams read the instance-normalised raw feature tensor directly, deliberately bypassing feature-group fusion and the temporal encoder, because those layers project the feature channels into an abstract latent space and destroy the feature identity that a features-as-nodes graph and a regime detector both require. This asymmetric tri-stream routing, shown in Figure 5, is what allows one model to address all three limitations at once [48]. Known trade-offs remain: a higher parameter count and tuning complexity than any single baseline; static Pearson graph sensitivity to distributional shift, only partially offset by RevIN and regime gating; manually specified regime boundaries tied to Malaysian MCO chronology; and an optional post-hoc gradient-boosted residual correction that adds a less interpretable second stage.

### 3.6 Evaluation Metrics

Combined% is a composite index aggregating absolute, proportional, and squared error into a single comparison rank. All three constituent terms are mean-demand-normalised percentages, making them dimensionally homogeneous and directly addable; the index is clipped so it cannot be negative, and higher is better. Equal weighting reflects the operational reality of transit management, where percentage error relative to daily volume (MAPE%), average absolute passenger discrepancy (MAE%), and large individual forecast failures (RMSE%) are equally critical.

Raw MAE and RMSE in absolute passenger counts are reported separately as operational diagnostics. R² measures explained variance and discriminates between models with similar error magnitudes but different directional tracking, with values below zero indicating performance worse than predicting the mean. Together with Combined%, these five metrics evaluate HMT-TSF against the fourteen baselines across look-back horizons and both MCO conditions.

## 4 Findings and Results



### 4.1 Experimental Setup

The central question is whether a purpose-built hybrid architecture can outperform established DL paradigms for seven-day Malaysian ridership forecasting, and under what conditions each paradigm succeeds or fails. Default-configuration performance under normal operating conditions (MCO-excluded, 14-day look-back) establishes the baseline ranking; hyperparameter tuning tests whether capacity increases translate into genuine generalisation; look-back sensitivity across 14, 28, and 56 days examines how much historical context a seven-day horizon requires; and structural-break robustness under the MCO period stress-tests each model against the 85–90% collapse.

All models follow a shared training protocol (AdamW, Huber loss, 70/15/15 chronological split) and are ranked by Combined%, with R², MAE, and RMSE completing the profile. The headline configuration is `base_nomco_lb14`. Fit diagnostics flag validation drift exceeding 25% above the best checkpoint and a train/validation gap ratio above 3×.

### 4.2 Baseline Model Performance

Under the project default, baseline accuracy is tightly clustered: the top eight models occupy a 1.5 Combined% band, indicating a saturated regime in which architectural differences yield incremental rather than step-change gains.

**Table 1.** Default-configuration (base_nomco_lb14) performance of the eight highest-ranked baseline variants. Full 16-variant ranking available from the authors.


| Rank | Model      | Combined% | MAPE% | R2    | MAE    |
| ---- | ---------- | --------- | ----- | ----- | ------ |
| 1    | Informer   | 79.13     | 6.59  | 0.772 | 69,476 |
| 2    | BiLSTM     | 79.04     | 6.52  | 0.776 | 72,460 |
| 3    | TPA-LSTM   | 78.67     | 6.62  | 0.776 | 75,871 |
| 4    | CNN-BiLSTM | 78.18     | 6.81  | 0.764 | 76,884 |
| 5    | LSTM       | 78.13     | 6.71  | 0.760 | 77,719 |
| 6    | ST-LSTM    | 78.01     | 6.93  | 0.770 | 78,779 |
| 7    | MTGNN      | 77.93     | 6.91  | 0.775 | 81,292 |
| 8    | CNN-LSTM   | 77.64     | 6.92  | 0.752 | 79,375 |


Recurrent architectures are the strongest single-paradigm family under normal conditions, occupying five of the top eight positions, because gated sequential memory aligns with the daily and weekly periodicity identified in Section 3.3 [42]. Bidirectional encoding provides a measurable advantage, with BiLSTM gaining 0.91 Combined% over LSTM by contextualising mid-sequence events [43], and TPA-LSTM matching BiLSTM's R² of 0.776, validating temporal pattern attention over hidden states [44]. CNN-BiLSTM and sequential CNN-LSTM approach the same band through hierarchical local-then-global encoding [5], while ST-LSTM ranks sixth with decoupled spatial encoding [41]. Graph models show wider dispersion, reflecting heterogeneous strategies for encoding inter-feature correlation: MTGNN leads the family at 77.93% because asymmetric learned edges can represent directed relationships — fuel-price signals influencing ridership without the reverse — which symmetric Pearson graphs cannot [51].

Test-set accuracy alone is an unreliable deployment criterion, because several high-ranking baselines achieve their position partly through memorisation. Across all twelve configurations per model, all six LSTM variants and both simple-convolution graph models record zero good-fit verdicts despite strong headline scores. Gap ratios span 3.3× to 13.6× for the LSTM family, and PDR-STGCN is the most extreme memoriser at 38.61×. Informer alone achieves a perfect 6/6 good-fit rate at base configuration, with gap ratios of 2.16×–2.98× consistently below threshold; MTGNN and STSGCN are cleanest thereafter, consistent with intrinsic regularisation from learned adjacency [51]. Tuning uniformly harms diagnostics, with the good-fit rate falling from 26% to 14% across all fourteen baselines.

### 4.3 HMT-TSF Model Performance

HMT-TSF was evaluated across ten configurations spanning two MCO conditions and five look-back windows.

**Table 2.** HMT-TSF (79 features) results across all ten configurations.


| Configuration | Combined% | MAPE% | MAE% | RMSE% | R²    | MAE    | RMSE    |
| ------------- | --------- | ----- | ---- | ----- | ----- | ------ | ------- |
| nomco_lb7     | 84.88     | 4.74  | 4.25 | 6.13  | 0.888 | 53,325 | 76,951  |
| nomco_lb14    | 85.78     | 4.39  | 3.97 | 5.87  | 0.897 | 49,897 | 73,757  |
| nomco_lb28    | 84.23     | 4.94  | 4.55 | 6.28  | 0.883 | 57,268 | 78,910  |
| nomco_lb56    | 82.53     | 5.52  | 5.11 | 6.84  | 0.859 | 64,198 | 85,874  |
| nomco_lb84    | 77.68     | 6.99  | 6.59 | 8.73  | 0.773 | 82,174 | 108,800 |
| mco_lb7       | 78.61     | 6.89  | 6.12 | 8.38  | 0.807 | 75,401 | 103,170 |
| mco_lb14      | 81.99     | 5.68  | 5.18 | 7.15  | 0.859 | 63,853 | 88,228  |
| mco_lb28      | 79.89     | 6.26  | 5.83 | 8.02  | 0.822 | 72,032 | 99,059  |
| mco_lb56      | 79.47     | 6.68  | 5.91 | 7.95  | 0.825 | 73,334 | 98,598  |
| mco_lb84      | 78.45     | 6.73  | 6.17 | 8.64  | 0.791 | 76,593 | 107,293 |


Performance peaks at a 14-day look-back under both conditions, declining at longer windows once Xfuture supplies explicit future calendar structure. At `nomco_lb14` the full deployable system leads every baseline on every headline metric — 85.78 Combined% against Informer's 79.13%, a margin of 6.65 points, with MAE 28.2% lower (49,897 against 69,476). The best tuned baseline, Informer at 79.99%, remains 5.79 points behind. HMT-TSF is also the most robust model under structural break at lb14, with degradation of only Δ −3.79 against Informer's −5.88, and retains the highest absolute MCO accuracy at 81.99%. It posts 19/20 good-fit verdicts with Xfuture enabled (gap ratios 1.57×–2.64×); only `nomco_lb84` is overfit.

Three sequential non-overlapping out-of-sample blocks test whether the headline depends on one favourable temporal partition. No block falls below 71.99 Combined% or 0.707 R² across all ten configurations; at `nomco_lb14` block scores are 90.62, 82.14, and 85.12. The configuration `mco_lb84` is the most temporally stable (block spread 3.7 points) while `nomco_lb14` spreads 8.5 points, linking peak-accuracy deployment to short windows and cross-period consistency to longer MCO-trained ones.

Because HMT-TSF alone receives the known-future calendar tensor, all ten configurations were retrained with the conditioning disabled to separate input design from architecture. Without Xfuture, HMT-TSF still leads Informer at `nomco_lb14` by +1.76 Combined% and cuts MAE by roughly 10%, a modest but positive architecture-only margin within the saturated baseline band. Disabling it also worsens MCO degradation at lb14 from Δ −3.79 to Δ −5.40, showing that known-future holiday and weekend structure aids structural-break transfer rather than serving only as a deployment convenience. The 14-day optimum holds under both settings, though `nomco_lb56` becomes overfit in the ablated runs.

### 4.4 Feature-Reduced Variant: HMT-TSF-FR

SHAP values computed for the trained `nomco_lb14` model recover a domain-consistent hierarchy. Annual-cycle encodings dominate attribution magnitude, anchoring seven-day forecasts on seasonal position — school terms, festive seasons, and the Northeast Monsoon. Total ridership is the strongest non-seasonal driver and the only feature in the top fifteen of all ten configurations. Holiday structure forms the second calendar tier, with days-to-next-public-holiday in the headline top ten, quantifying the displacement pattern of Figure 3. Among external covariates, unfrozen fuel grades and monsoon-corridor rainfall complete the top ten [41].

Cross-configuration aggregation reveals 23 zero-SHAP features plus three near-zero RON95 variants, defining a 26-feature removal set — the entire static group plus nine near-constant administered fuel columns — and yielding HMT-TSF-FR on a 53-feature input. Their persistent inactivity independently validates the exploratory conclusion of Section 3.3: static descriptors are redundant because time-invariant structural capacity is implicitly encoded in historical ridership given sufficient look-back, and administered fuel columns carry near-zero day-to-day variance across the post-MCO window.

**Table 3.** HMT-TSF-FR versus HMT-TSF Combined% by look-back window.


| Look-back | HMT-TSF | HMT-TSF-FR | Δ     |
| --------- | ------- | ---------- | ----- |
| lb14      | 85.78   | 86.59      | +0.81 |
| lb28      | 84.23   | 86.07      | +1.84 |
| lb56      | 82.53   | 84.05      | +1.52 |
| lb84      | 77.68   | 81.94      | +4.26 |


HMT-TSF-FR reaches 86.59 Combined% at `nomco_lb14` (R² = 0.906, MAE = 46,323), the study-wide headline. The graph adjacency shrinks from 79 × 79 to 53 × 53 and explainability memory falls by roughly 33%. As Table 3 shows, the advantage grows with look-back and repairs the `nomco_lb84` overfit verdict. The trade-off appears under structural break: HMT-TSF-FR loses to the full model in every MCO configuration, from Δ −0.85 at lb7 to Δ −5.22 at lb84, indicating that near-constant fuel-price columns stabilise predictions under lockdown-spanning shift. The full model should therefore be retained for shock-prone conditions and FR deployed for normal operations. All FR configurations remain good-fit with gap ratios of 1.66×–2.72×, confirming that reduction introduces no overfitting.

## 5 Discussion and Conclusion

Principled fusion of complementary inductive biases outperforms any single architectural paradigm for seven-day Malaysian ridership forecasting. HMT-TSF achieved 85.78 Combined% and R² 0.897 at `nomco_lb14` (86.59 and 0.906 for the feature-reduced variant), leading Informer by 6.65 points and cutting daily MAE by 28.2%, with the best tuned baseline still 5.79 points behind. Because HMT-TSF alone receives known-future calendar conditioning — legitimately available at deployment but absent from baseline inputs — the ablation of Section 4.3 is what bounds the claim: a positive but modest architecture-only margin of +1.76 points remains without it. Within the 1.5 Combined% saturation band separating the top eight baselines, marginal gains therefore lie in multi-modal fusion combined with task-appropriate input design rather than in incremental paradigm refinement.

Nominal test-set accuracy is an unreliable indicator of deployment readiness. All six LSTM models and both simple-convolution graph models record zero good-fit verdicts across twelve configurations, with train/validation gaps of 3.3×–13.6× and PDR-STGCN at 38.61×, despite BiLSTM and TPA-LSTM ranking second and third. This dissociation reflects memorisation at the expense of adaptability [45] and contrasts with Informer's unique clean-fit record achieved through ProbSparse regularisation [53]. Had only Combined% been reported, BiLSTM would appear nearly equivalent to Informer, yet diagnostics place them at opposite ends of the generalisation spectrum. Hyperparameter tuning further eroded fit quality for small, inconsistent test gains, so fit diagnostics must be evaluated alongside accuracy.

Two structural results follow. A 14-day look-back is decisively optimal for a seven-day horizon: it was best for ten of sixteen baseline variants, and HMT-TSF is single-peaked there. Fragile architectures collapse at longer windows when structural priors mislead [50], whereas MTGNN's multi-scale dilated inception remains near-stable [51]; predictability concentrates in two weekly cycles, with slower seasonal structure supplied through calendar features rather than extended history [52]. Robustness under distributional shift, meanwhile, is governed by the rigidity or adaptivity of structural assumptions rather than by nominal accuracy — degradation ranges from HMT-TSF's Δ −3.79 to STFGNN's Δ −32.94, with two baselines falling below naïve mean-prediction despite strong default rankings. This reveals an expressiveness–stability trade-off in which constrained designs [47] and sparse attention [53] transfer more reliably than flexible learners that memorise pre-break regularities; RevIN and regime gating resolve it for HMT-TSF, which retains 81.99% MCO accuracy under a clean fit.

The study contributes three theoretical results: that the features-as-nodes graph formulation transfers graph-based traffic methods to multivariate national ridership data while exposing where transfer breaks down; that fit diagnostics are a necessary complement to test-set accuracy, materially reordering practical rankings; and that regime gating combined with RevIN is an effective mechanism for structural-break robustness. Practically, accurate seven-day forecasts enable frequency setting, rolling-stock allocation, and staffing matched to predicted demand, with HMT-TSF-FR recommended for normal operations and the full model for shock-prone regimes [28]. Established networks such as the Klang Valley and Penang systems support phased integration in parallel with existing planning, while developing corridors in Johor and East Malaysia can be designed around demand forecasts from the outset. Limitations bound these claims: the study operates at national daily granularity and does not address station-level or intra-day crowd management [41, 44]; it relies on one national dataset, leaving cross-network external validity untested; the MCO supplies only one observed structural break, so the three-regime gating embedding is calibrated to a single shock type; headline results are single-seed point estimates, so sub-percentage gaps within the saturation band should be treated as indicative; and only demand-side unreliability is modelled, since no public Malaysian daily dataset covers supply-side disruption.

Future work should re-run headline configurations across multiple seeds with bootstrap or Diebold-Mariano tests, and either equip top baselines with the same per-horizon future-calendar tensors or retrain all models under a shared no-conditioning protocol, separating input-design gains from architecture gains. Integrating supply-side reliability data — AVL traces, on-time-performance records, and disruption logs — would link demand forecasting to service reliability management, while real-time service alerts and trip updates would move the system from offline planning toward operational decision support. Extending the features-as-nodes design to station-level nodes with geographic adjacency would enable intra-day forecasting, and an open-set regime detector recognising fare reforms or new-line openings online would turn retrospective robustness into prospective deployment capability. This study set out to determine whether a purpose-built hybrid architecture could outperform established DL paradigms for seven-day Malaysian ridership forecasting, and to characterise when each paradigm succeeds or fails. The answer is affirmative and well-bounded: HMT-TSF leads all baselines on headline metrics and structural-break robustness, a modest architecture-only margin persists without calendar conditioning, and deployment should follow the 14-day window, fit-diagnostic, and feature-reduction rules established here — together, a credible foundation for demand-responsive transit planning in Malaysia.

## List of Abbreviations

AFC, Automated Fare Collection; APC, Automated Passenger Counting; ARIMA, AutoRegressive Integrated Moving Average; ASTGCN, Attention-based Spatial-Temporal Graph Convolutional Network; AVL, Automated Vehicle Location; BiLSTM, Bidirectional Long Short-Term Memory; CNN, Convolutional Neural Network; DL, Deep Learning; FR, Feature-Reduced; GADM, Global Administrative Areas; GCN, Graph Convolutional Network; GTFS, General Transit Feed Specification; HMT-TSF, Hybrid Multi-Scale Temporal Spatio-Feature Forecaster; KTM, Keretapi Tanah Melayu; LRT, Light Rail Transit; LSTM, Long Short-Term Memory; MAE, Mean Absolute Error; MAPE, Mean Absolute Percentage Error; MCO, Movement Control Order; MRT, Mass Rapid Transit; MTGNN, Multivariate Time series Graph Neural Network; OSM, OpenStreetMap; PDR-STGCN, Periodic Dynamic Relational STGCN; POI, Point of Interest; RevIN, Reversible Instance Normalisation; RMSE, Root Mean Square Error; SE, Squeeze-and-Excitation; SHAP, SHapley Additive exPlanations; SOTA, State of the Art; STFGNN, Spatial-Temporal Fusion Graph Neural Network; STGCN, Spatio-Temporal Graph Convolutional Network; ST-LSTM, Spatio-Temporal LSTM; STSGCN, Spatial-Temporal Synchronous Graph Convolutional Network; TCN, Temporal Convolutional Network; TPA-LSTM, Temporal Pattern Attention LSTM.

## Statements and Declarations

**Competing Interests:** [TO BE COMPLETED — disclose any financial or non-financial interests directly or indirectly related to the submitted work, or state that the authors have no relevant interests to declare.]

**Funding:** [TO BE COMPLETED — state funding sources with full organisation names, or state that no funding was received.]

**Ethics Approval:** [TO BE COMPLETED — not applicable if the study used only publicly available aggregate data.]

**Data Availability:** [TO BE COMPLETED — all primary sources are publicly available from data.gov.my, worldpop.org, OpenStreetMap, and timeanddate.com.]

**Acknowledgments:** [TO BE COMPLETED.]

## References

[1] Lu, K, Liu, J, Zhou, X, et al., "A Review of Big Data Applications in Urban Transit Systems", IEEE Transactions on Intelligent Transportation Systems, 22(5), 2020, pp. 2535-2552, 10.1109/TITS.2020.2973365

[2] Wang, J, Liu, C, Wu, Z, et al., "The Roadmap and Strategy for Prioritizing the Development of Public Transport in China", Multimodal Transportation, 4(1), 2025, 100184, 10.1016/j.multra.2024.100184

[3] Amir, N N, Anuar, N S, Ismail, B, et al., "Ridership Prediction System for Rapid Bus Kuantan and Penang using Multi-Feature Analysis", Journal of the Malaysian Institute of Planners, 23(6), 2025, pp. 332-347, 10.21837/pm.v23i39.1913

[4] Mee, C K, Subramaniam, G, Ating, R, et al., "Willingness to use Public Transport in Kuala Lumpur & Manila", Environment-Behaviour Proceedings Journal, 7(21), 2022, pp. 1-9, 10.21834/ebpj.v7i21.3751

[5] Chen, P, Fu, X and Wang, X, "A Graph Convolutional Stacked Bidirectional Unidirectional-LSTM Neural Network for Metro Ridership Prediction", IEEE Transactions on Intelligent Transportation Systems, 23(7), 2022, pp. 6950-6962, 10.1109/TITS.2021.3065404

[6] Ge, L, Sarhani, M, Voss, S, et al., "Review of Transit Data Sources: Potentials, Challenges and Complementarity", Sustainability, 13(20), 2021, 11450, 10.3390/su132011450

[7] Zhao, M and Lan, C-L, "Modeling Travel Time Reliability for Non-Interstate National Highway System Routes", Virginia Transportation Research Council, 2025, pp. 1-68, [https://vtrc.virginia.gov/media/vtrc/vtrc-pdf/vtrc-pdf/26-R23.pdf](https://vtrc.virginia.gov/media/vtrc/vtrc-pdf/vtrc-pdf/26-R23.pdf)

[8] Lv, Z, Li, J, Dong, C, et al., "Deep Learning in the COVID-19 epidemic: A deep model for urban traffic revitalization index", Data & Knowledge Engineering, 135, 2021, 101912, 10.1016/j.datak.2021.101912

[9] Mohamed, A H, Adwan, I A, Ahmeda, A G, et al., "Identification of Affecting Factors on the Travel Time Reliability for Bus Transportation", Knowledge-based Engineering and Science, 2(1), 2021, pp. 19-30, 10.51526/kbes.2021.2.1.19-30

[10] Ibrahim, A N, Borhan, M N, Osman, M H, et al., "The Influence of Service Quality on User's Perceived Satisfaction with Light Rail Transit Service in Klang Valley, Malaysia", Mathematics, 10, 2022, 2213, 10.3390/math10132213

[11] Ubaidillah, N Z, Sa'ad, N H, Ismail, F, et al., "The Impact of Public Bus Service Quality on the Users' Satisfaction: Evidence from a Developing Asian City", Review of Applied Socio-Economic Research, 23(1), 2022, pp. 83-96, 10.54609/reaser.v23i1.185

[12] Liu, C, Xiao, Z, Long, W, et al., "Vehicle Trajectory Data Processing, Analytics and Applications: A Survey", ACM Computing Surveys, 57(9), 2025, pp. 1-36, 10.1145/3715902

[13] Keller, C, Gluck, F, Gerlach, C F, et al., "Investigating the Potential of Data Science Methods for Sustainable Public Transport", Sustainability, 14, 2022, 4211, 10.3390/su14074211

[14] S.K.B, S, Mathivanan, S K, Rajadurai, H, et al., "A multi-modal geospatial-temporal LSTM based deep learning framework for predictive modeling of urban mobility patterns", Scientific Reports, 14, 2024, pp. 1-19, 10.1038/s41598-024-74237-3

[15] Maria-Arribas, D, Pantrigo, J J and Cuesta-Infante, A, "A massive, graph augmented, traffic dataset for machine learning and deep learning spatio-temporal traffic analysis", Research Square, 2026, pp. 1-17, 10.21203/rs.3.rs-8670080/v1

[16] Yin, X, Wu, G, Wei, J, et al., "Deep Learning on Traffic Prediction: Methods, Analysis, and Future Directions", IEEE Transactions on Intelligent Transportation Systems, 23(6), 2022, pp. 4927-4943, 10.1109/TITS.2021.3054840

[17] Sobrie, L, Verschelde, M, Hennebel, V, et al., "Capturing complexity over space and time via deep learning: An application to real-time delay prediction in railways", European Journal of Operational Research, 310(3), 2023, pp. 1201-1217, 10.1016/j.ejor.2023.03.040

[18] Jiang, R, Yin, D, Wang, Z, et al., "DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction", Proceedings of the 30th ACM International Conference on Information & Knowledge Management, 2021, pp. 4515-4525, 10.1145/3459637.3482000

[19] Mystakidis, A, Koukaras, P and Tjortjis, C, "Advances in Traffic Congestion Prediction: An Overview of Emerging Techniques and Methods", Smart Cities, 8(1), 2025, 25, 10.3390/smartcities8010025

[20] Jevinger, A, Zhao, C, Persson, J A, et al., "Artificial Intelligence for improving public transport: A mapping study", Public Transport, 16, 2024, pp. 99-158, 10.1007/s12469-023-00334-7

[21] Levner, A, "AI-Powered Public Transportation Gains Momentum as Optibus Platform hits One Million Optimizations", Optibus, 2025. Available at [https://blog.optibus.com/ai-powered-public-transportation-gains-momentum-as-optibus-platform-hits-one-million-optimizations](https://blog.optibus.com/ai-powered-public-transportation-gains-momentum-as-optibus-platform-hits-one-million-optimizations) (accessed 31 March 2025)

[22] CCS Global Tech, "From Reactive to Proactive Decisions: How AI Enablement is Solving Everyday Transit Challenges", 2026. Available at [https://ccsglobaltech.com/from-reactive-to-proactive-decisions-how-ai-enablement-is-solving-everyday-transit-challenges/](https://ccsglobaltech.com/from-reactive-to-proactive-decisions-how-ai-enablement-is-solving-everyday-transit-challenges/) (accessed 9 March 2026)

[23] Suwaidi, J A, Aydin, R and Rashid, H, "Investigating Barriers and Challenges to Artificial Intelligence (AI) Implementation in Logistics Operations: A Systematic Review of Literature", Proceedings of the 5th European International Conference on Industrial Engineering and Operations Management, 2022, pp. 1600-1616, 10.46254/EU05.20220308

[24] Hashimzai, I A and Mohammadi, M Q, "The Integration of Artificial Intelligence in Project Management: A Systematic Literature Review of Emerging Trends and Challenges", TIERS Information Technology Journal, 5(2), 2024, pp. 152-164, 10.38043/tiers.v5i2.5963

[25] Son, H, Jang, J, Park, J, et al., "Leveraging Advanced Technologies for (Smart) Transportation Planning: A Systematic Review", Sustainability, 17(5), 2025, 2245, 10.3390/su17052245

[26] Vujadinovic, V L, Damnjanovic, A, Cakic, A, et al., "AI-Driven Approach for Enhancing Sustainability in Urban Public Transportation", Sustainability, 16(17), 2024, 7763, 10.3390/su16177763

[27] Farahmand, Z H, Gkiotsalitis, K and Geurs, K T, "Predicting bus ridership based on the weather conditions using deep learning algorithms", Transportation Research Interdisciplinary Perspectives, 19, 2023, 100833, 10.1016/j.trip.2023.100833

[28] Hassan, M, Mahin, H D, Ahmed, F, et al., "Assessing Public Transit Network Efficiency and Accessibility in Johor Bahru and Penang, Malaysia: A Data-driven Approach", Results in Engineering, 27, 2025, 106126, 10.1016/j.rineng.2025.106126

[29] Jin, S, Jing, C, Wang, Y, et al., "Spatiotemporal Graph Convolutional Neural Networks for Metro Flow Prediction", The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLIII, 2022, pp. 403-409, 10.5194/isprs-archives-XLIII-B4-2022-403-2022

[30] Ahmad, M, Anwer, I, Yousuf, M I, et al., "Investigating the Key Factors Affecting Public Transport Ridership in Developing Countries through Structural Equation Modeling", Sustainability, 16(11), 2024, 4426, 10.3390/su16114426

[31] Shi, B, Wang, Z, Yan, J, et al., "A Novel Spatial-Temporal Deep Learning Method for Metro Flow Prediction Considering External Factors and Periodicity", Applied Sciences, 14(5), 2024, 1949, 10.3390/app14051949

[32] Pour, M A, Zargari, F, Samimi, A, et al., "Analyzing the influence of fuel price shock on urban public transit and interurban automobile travel demand: Evidence from a country with fixed fuel price regulation", Transportation Research Interdisciplinary Perspectives, 36, 2026, 101829, 10.1016/j.trip.2025.101829

[33] Casolaro, A, Capone, V, Iannuzzo, G, et al., "Deep Learning for Time Series Forecasting: Advances and Open Problems", Information, 14(11), 2023, 598, 10.3390/info14110598

[34] Wu, J-L, Lu, M and Wang, C-Y, "Forecasting metro rail transit passenger flow with multiple-attention deep neural networks and surrounding vehicle detection devices", Applied Intelligence, 53, 2023, pp. 18531-18546, 10.1007/s10489-023-04483-x

[35] Yusuf, O, Rasheed, A and Lindseth, F, "Data-driven predictive modelling of stop-level public transit patterns", Transportation, 2025, pp. 1-53, 10.1007/s11116-025-10689-4

[36] Ngo, N S and Bashar, S, "The impacts of extreme weather events on U.S. Public transit ridership", Transportation Research Part D: Transport and Environment, 137, 2024, 104504, 10.1016/j.trd.2024.104504

[37] Jiang, S and Cai, C, "The impacts of weather conditions on metro ridership: An empirical study from three mega cities in China", Travel Behaviour and Society, 31, 2023, pp. 166-177, 10.1016/j.tbs.2022.12.003

[38] Huang, S, Li, W, Wen, J, et al., "Spatiotemporal Variations in Shanghai Metro Commuting Flows during Rainfall Events", Weather, Climate and Society, 14(3), 2022, pp. 979-991, 10.1175/WCAS-D-21-0167.1

[39] Chevance, G, Andrieu, B, Koch, N, et al., "How gasoline prices influence the effectiveness of interventions targeting sustainable transport modes?", Sustainable Mobility and Transport, 1, 2024, pp. 1-8, 10.1038/s44333-024-00017-1

[40] Rahmani, B, Moghaddam, A M and Maghrebi, M, "Forecasting demand fluctuations of public bus transit during special events and adverse weather conditions through smart card data analysis", Travel Behaviour and Society, 40, 2025, 101033, 10.1016/j.tbs.2025.101033

[41] Cui, H, Si, B, Chi, D, et al., "Short-term passenger flow prediction for urban rail systems: A deep learning approach utilizing multi-source big data", PLoS One, 20(10), 2025, pp. 1-23, 10.1371/journal.pone.0333094

[42] Barath, Z, Veres, P and Banyai, A, "Beyond Traditional Forecasting Methods: Evaluating LSTM Performance on Diverse Time Series", Mathematics, 14(5), 2026, 838, 10.3390/math14050838

[43] Krishnasamy, L, C, S, Dhanaraj, R K, et al., "Intelligent traffic congestion forecasting using BiLSTM and adaptive secretary bird optimizer for sustainable urban transportation", Scientific Reports, 15, 2025, 18423, 10.1038/s41598-025-02933-9

[44] Wei, L, Guo, D, Chen, Z, et al., "Forecasting Short-Term Passenger Flow of Subway Stations Based on the Temporal Pattern Attention Mechanism and the Long Short-Term Memory Network", ISPRS International Journal of Geo-Information, 12(1), 2023, 25, 10.3390/ijgi12010025

[45] Topilin, I, Jiang, J, Feofilova, A, et al., "Traffic Flow Prediction via a Hybrid CPO-CNN-LSTM-Attention Architecture", Smart Cities, 8(5), 2025, 148, 10.3390/smartcities8050148

[46] Zhuang, W and Cao, Y, "Short-Term Traffic Flow Prediction Based on CNN-BILSTM with Multicomponent Information", Applied Sciences, 12(17), 2022, 8714, 10.3390/app12178714

[47] Deng, H, "Traffic-Forecasting Model with Spatio-Temporal Kernel", Electronics, 14(7), 2025, 1410, 10.3390/electronics14071410

[48] Hu, J, Tang, B, Zhu, L, et al., "PDR-STGCN: An Enhanced STGCN with Multi-Scale Periodic Fusion and a Dynamic Relational Graph for Traffic Forecasting", Systems, 14(1), 2026, 102, 10.3390/systems14010102

[49] Chen, L, Ren, Q, Zheng, J, et al., "CSFPre: Expressway key sections based on CEEMDAN-STSGCN-FCM during the holidays for traffic flow prediction", PLoS One, 18(4), 2023, pp. 1-22, 10.1371/journal.pone.0283898

[50] Chang, J, Yin, J, Hao, Y, et al., "STFDSGCN: Spatio-Temporal Fusion Graph Neural Network Based on Dynamic Sparse Graph Convolution GRU for Traffic Flow Forecast", Sensors, 25(11), 2025, 3446, 10.3390/s25113446

[51] Wu, Z, Liu, X and Zhang, X, "Multi dynamic temporal representation graph convolutional network for traffic flow prediction", Scientific Reports, 15, 2025, 16734, 10.1038/s41598-025-01157-1

[52] Ma, X and Zhang, H, "Time Series Forecasting Method Based on Multi-Scale Feature Fusion and Autoformer", Applied Sciences, 15(7), 2025, 3768, 10.3390/app15073768

[53] Song, Y, Luo, R, Zhou, T, et al., "Graph Attention Informer for Long-Term Traffic Flow Prediction under the Impact of Sports Events", Sensors, 24(15), 2024, 4796, 10.3390/s24154796

[54] Kim, T, Kim, J, Tae, Y, et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift", ICLR 2022 Conference, 2023, pp. 1-25, [https://openreview.net/forum?id=cGDAkQo1C0p](https://openreview.net/forum?id=cGDAkQo1C0p)

---



## Citation mapping (audit only)


| [n] | Original APA key            | First cited in |
| --- | --------------------------- | -------------- |
| 1   | Lu et al., 2020             | §1 ¶1          |
| 2   | Wang et al., 2025           | §1 ¶1          |
| 3   | Amir et al., 2025           | §1 ¶1          |
| 4   | Mee et al., 2022            | §1 ¶2          |
| 5   | Chen, P. et al., 2022       | §1 ¶3          |
| 6   | Ge et al., 2021             | §1 ¶3          |
| 7   | Zhao & Lan, 2025            | §1 ¶4          |
| 8   | Lv et al., 2021             | §1 ¶4          |
| 9   | Mohamed et al., 2021        | §2.1           |
| 10  | Ibrahim et al., 2022        | §2.1           |
| 11  | Ubaidillah et al., 2022     | §2.1           |
| 12  | Liu et al., 2025            | §2.2           |
| 13  | Keller et al., 2022         | §2.2           |
| 14  | S.K.B et al., 2024          | §2.2           |
| 15  | Maria-Arribas et al., 2026  | §2.2           |
| 16  | Yin et al., 2022            | §2.3           |
| 17  | Sobrie et al., 2023         | §2.3           |
| 18  | Jiang, R. et al., 2021      | §2.3           |
| 19  | Mystakidis et al., 2025     | §2.3           |
| 20  | Jevinger et al., 2024       | §2.4           |
| 21  | Levner, 2025                | §2.4           |
| 22  | CCS Global Tech, 2026       | §2.4           |
| 23  | Suwaidi et al., 2022        | §2.4           |
| 24  | Hashimzai & Mohammadi, 2024 | §2.4           |
| 25  | Son et al., 2025            | §2.4           |
| 26  | Vujadinovic et al., 2024    | §2.4           |
| 27  | Farahmand et al., 2023      | §2.5           |
| 28  | Hassan et al., 2025         | §3.1           |
| 29  | Jin et al., 2022            | §3.1           |
| 30  | Ahmad et al., 2024          | §3.1           |
| 31  | Shi et al., 2024            | §3.1           |
| 32  | Pour et al., 2026           | §3.2           |
| 33  | Casolaro et al., 2023       | §3.2           |
| 34  | Wu, J.-L. et al., 2023      | §3.2           |
| 35  | Yusuf et al., 2025          | §3.2           |
| 36  | Ngo & Bashar, 2024          | §3.3           |
| 37  | Jiang & Cai, 2023           | §3.3           |
| 38  | Huang et al., 2022          | §3.3           |
| 39  | Chevance et al., 2024       | §3.3           |
| 40  | Rahmani et al., 2025        | §3.3           |
| 41  | Cui et al., 2025            | §3.3           |
| 42  | Barath et al., 2026         | §3.4           |
| 43  | Krishnasamy et al., 2025    | §3.4           |
| 44  | Wei et al., 2023            | §3.4           |
| 45  | Topilin et al., 2025        | §3.4           |
| 46  | Zhuang & Cao, 2022          | §3.4           |
| 47  | Deng, 2025                  | §3.4           |
| 48  | Hu et al., 2026             | §3.4           |
| 49  | Chen, L. et al., 2023       | §3.4           |
| 50  | Chang et al., 2025          | §3.4           |
| 51  | Wu, Z. et al., 2025         | §3.4           |
| 52  | Ma & Zhang, 2025            | §3.4           |
| 53  | Song et al., 2024           | §3.4           |
| 54  | Kim et al., 2023            | §3.5           |


**Dropped from the thesis bibliography** (cited only in removed content): Khalil et al. (2021), Rezapour & Ferraro (2021), Sogbe et al. (2024). 57 → 54 references.

## Figure and table source map (audit only)


| Output   | Thesis source                                   | Source paragraph / table index |
| -------- | ----------------------------------------------- | ------------------------------ |
| Figure 1 | Figure 3.1                                      | para 447                       |
| Figure 2 | Figure 3.6                                      | para 472                       |
| Figure 3 | Figure 3.15                                     | para 508                       |
| Figure 4 | Figure 3.16                                     | para 516                       |
| Figure 5 | Figure 3.18                                     | para 619                       |
| Table 1  | Table 4.1, rows 1-8, 6 of 9 cols                | `doc.tables[11]`               |
| Table 2  | Table 4.5, all rows/cols                        | `doc.tables[15]`               |
| Table 3  | authored — derived from Table 4.5 + §4.9 deltas | none                           |


