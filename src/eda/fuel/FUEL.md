# Fuel Price Exploratory Data Analysis Report

This report presents the exploratory data analysis of various fuel price time series including diesel, RON95, RON97, and their regional variants. Each visualization helps understand different characteristics of the fuel price data such as trends, seasonality, volatility, and autocorrelation patterns.

## Autocorrelation Function (ACF) Analysis

### acf_diesel.png
**Explanation:** This autocorrelation function (ACF) plot shows the correlation of diesel fuel prices with their lagged values over time. The x-axis represents the lag (time delay) and the y-axis shows the correlation coefficient.

**Analysis:** The ACF helps identify patterns of dependence in the time series. For diesel prices, significant autocorrelation at low lags would indicate short-term price persistence, while significant correlations at higher lags might reveal seasonal or cyclical patterns. The rate at which autocorrelation decays provides insights into the memory of the process - slow decay suggests long-term dependencies, while rapid decay indicates short memory. This information is crucial for selecting appropriate time series models (AR, MA, ARIMA) for forecasting diesel prices.

### acf_diesel_eastmsia.png
**Explanation:** This ACF plot shows the autocorrelation structure of diesel fuel prices in the East Malaysia region.

**Analysis:** Comparing this with national diesel ACF can reveal regional differences in price behavior. Similar autocorrelation patterns would suggest common national influences on pricing, while different patterns might indicate regional supply chain factors, transportation logistics, or local market dynamics affecting price persistence. Understanding these regional variations helps in developing localized pricing strategies and forecasting models.

### acf_ron95.png
**Explanation:** This ACF plot shows the autocorrelation structure of RON95 fuel prices.

**Analysis:** RON95 being the most widely used fuel grade in Malaysia, its autocorrelation pattern provides insights into general fuel market behavior. The ACF can reveal how quickly RON95 prices revert to mean levels after shocks, which is important for understanding market efficiency and the speed of price adjustment to changes in crude oil prices or government policies.

### acf_ron95_budi95.png
**Explanation:** This ACF plot examines the autocorrelation of RON95 blended with biodiesel (Budi95) fuel prices.

**Analysis:** The inclusion of biodiesel component might affect price volatility and persistence. Comparing this ACF with pure RON95 can show how biodiesel blending influences price dynamics - whether it adds stability through diversification or introduces new sources of volatility related to palm oil prices (the main biodiesel feedstock in Malaysia).

### acf_ron95_skps.png
**Explanation:** This ACF plot shows the autocorrelation structure of RON95 blended with specifically formulated diesel (SKPS variant).

**Analysis:** Similar to the Budi95 analysis, this helps understand how different blending formulations affect price behavior. SKPS might have different stability characteristics compared to standard diesel, which would be reflected in the autocorrelation pattern - potentially showing different lag structures or decay rates.

### acf_ron97.png
**Explanation:** This ACF plot shows the autocorrelation structure of RON97 fuel prices, the premium gasoline grade.

**Analysis:** Premium fuels often show different price dynamics compared to regular grades. The ACF for RON97 might reveal different sensitivity to crude oil prices, different seasonal patterns due to different usage profiles (more recreational/performance vehicles), or different responsiveness to government subsidies that typically target regular grades more than premium.

## Boxplot Analysis

### boxplot_diesel.png
**Explanation:** This boxplot displays the distribution of diesel fuel prices, showing median, quartiles, outliers, and range.

**Analysis:** The boxplot provides a quick visual summary of price distribution characteristics. The median shows central tendency, the interquartile range (IQR) shows price variability, and any outliers indicate extreme price events. Symmetry or skewness in the boxplot reveals whether price distributions are balanced or biased toward higher/lower values. This helps understand typical price ranges and identify unusual price spikes that might correspond to supply disruptions, policy changes, or global oil market shocks.

### boxplot_diesel_eastmsia.png
**Explanation:** This boxplot shows the distribution of diesel prices in East Malaysia.

**Analysis:** Comparing this boxplot with the national diesel boxplot reveals regional price differences. Variations in median prices indicate different price levels, while differences in IQR show varying degrees of price volatility. Outlier patterns might reveal region-specific events affecting prices differently than the national average, such as local supply chain issues or regional economic activity differences.

### boxplot_ron95.png
**Explanation:** This boxplot displays the distribution of RON95 fuel prices.

**Analysis:** As the most commonly used fuel, understanding RON95 price distribution is critical for assessing consumer cost burdens. The boxplot helps identify the typical price range consumers experience and the frequency of extreme prices. This information is valuable for policymakers considering subsidy mechanisms or price stabilization funds.

### boxplot_ron95_budi95.png
**Explanation:** This boxplot shows the distribution of RON95 blended with biodiesel prices.

**Analysis:** Comparing this with pure RON95 boxplot reveals how biodiesel blending affects price distribution. Similar medians would suggest comparable average prices, while differences in spread or skewness might indicate that biodiesel blending either stabilizes or destabilizes prices relative to petroleum components. This helps evaluate the economic viability of biodiesel mandates from a price stability perspective.

### boxplot_ron95_skps.png
**Explanation:** This boxplot displays the distribution of RON95 blended with SKPS variant prices.

**Analysis:** Similar to the Budi95 comparison, this helps assess how different blending approaches affect price characteristics for consumers. Understanding these differences is important for optimizing fuel formulation policies that balance environmental goals with price stability concerns.

### boxplot_ron97.png
**Explanation:** This boxplot shows the distribution of RON97 premium fuel prices.

**Analysis:** Premium fuel price distribution analysis helps understand the cost premium consumers pay for higher octane fuels. Comparing the median price and variability of RON97 with RON95 quantifies the typical premium and its stability. This information is relevant for understanding consumer choice between fuel grades and the effectiveness of pricing strategies for different market segments.

## Decomposition Analysis

### decomposition_diesel.png
**Explanation:** This time series decomposition plot breaks down diesel prices into trend, seasonal, and residual components.

**Analysis:** Decomposition helps understand the underlying drivers of price movements. The trend component shows long-term price direction (influenced by crude oil prices, exchange rates, and structural factors). The seasonal component reveals regular repeating patterns (possibly related to driving seasons, holiday travel, or agricultural cycles). The residual component shows irregular fluctuations after removing trend and seasonality (related to supply shocks, speculative activity, or unexpected events). Understanding these components is essential for accurate forecasting and policy analysis.

### decomposition_diesel_eastmsia.png
**Explanation:** This decomposition plot shows the trend, seasonal, and residual components of diesel prices in East Malaysia.

**Analysis:** Comparing decomposition components between national and regional diesel prices helps identify whether price drivers are similar or different across regions. Similar trend components would suggest common national/international influences, while different seasonal patterns might indicate region-specific consumption patterns or supply chain characteristics. Different residual patterns could reveal region-specific volatility sources.

### decomposition_ron95.png
**Explanation:** This decomposition plot breaks down RON95 prices into trend, seasonal, and residual components.

**Analysis:** For the most widely consumed fuel, understanding these components is particularly important. The trend component reflects long-term cost pressures, seasonal components might show increased demand during holiday periods or monsoon seasons affecting transportation, and residuals capture unexpected events like refinery outages or sudden global oil price changes. This analysis helps separate structural trends from temporary fluctuations for better policy responses.

### decomposition_ron95_budi95.png
**Explanation:** This decomposition plot shows the components of RON95 blended with biodiesel prices.

**Analysis:** Comparing this decomposition with pure RON95 helps assess how biodiesel blending affects different price components. If biodiesel reduces trend volatility, it might indicate better insulation from global crude oil prices. Changes in seasonal patterns could reflect different demand characteristics for biodiesel blends. Different residual patterns might show how biodiesel affects sensitivity to specific types of market shocks.

### decomposition_ron95_skps.png
**Explanation:** This decomposition plot breaks down RON95 blended with SKPS variant prices into components.

**Analysis:** Similar to the Budi95 analysis, this helps evaluate how different blending formulations influence price behavior across trend, seasonal, and irregular components. Understanding these differences supports optimization of fuel policies for both environmental objectives and price stability goals.

### decomposition_ron97.png
**Explanation:** This decomposition plot shows the trend, seasonal, and residual components of RON97 premium fuel prices.

**Analysis:** Premium fuel decomposition can reveal different behavior patterns compared to regular grades. For example, RON97 might show weaker seasonal components if its consumption is less tied to regular commuting patterns. Different trend behaviors might indicate different linkages to crude oil prices or different responsiveness to macroeconomic factors affecting premium vehicle ownership and usage.

## Distribution Analysis

### distribution_diesel.png
**Explanation:** This histogram or density plot shows the probability distribution of diesel fuel prices.

**Analysis:** The shape of the distribution provides insights into price behavior characteristics. A normal distribution would suggest random fluctuations around a mean, while skewness indicates asymmetric price movements (more frequent large increases or decreases). Kurtosis reveals tail thickness - high kurtosis suggests extreme price events occur more frequently than expected in a normal distribution. Understanding the actual distribution helps in risk assessment, option pricing for fuel contracts, and setting appropriate confidence intervals for forecasts.

### distribution_diesel_eastmsia.png
**Explanation:** This plot shows the probability distribution of diesel prices in East Malaysia.

**Analysis:** Comparing distributions between regions helps assess whether price risks are similar or different geographically. Similar distribution shapes would suggest comparable risk profiles, while different skewness or kurtosis might indicate region-specific factors leading to more asymmetric price movements or different frequencies of extreme events. This information is valuable for regional risk management and strategic planning.

### distribution_ron95.png
**Explanation:** This histogram or density plot displays the distribution of RON95 fuel prices.

**Analysis:** As the benchmark fuel for most consumers, understanding RON95 price distribution is crucial for assessing household and business cost burdens. The distribution shape helps predict the likelihood of different price levels occurring, which is important for budgeting at both individual and macroeconomic levels. Policymakers can use this information to design effective subsidy mechanisms that target appropriate price levels.

### distribution_ron95_budi95.png
**Explanation:** This plot shows the distribution of RON95 blended with biodiesel prices.

**Analysis:** Comparing this distribution with pure RON95 helps evaluate how biodiesel blending affects price risk profiles. If the biodiesel blend shows reduced skewness or kurtosis, it might indicate improved price stability. Changes in distribution shape could reflect how the renewable fuel component interacts with petroleum pricing dynamics under different market conditions.

### distribution_ron95_skps.png
**Explanation:** This histogram or density plot shows the distribution of RON95 blended with SKPS variant prices.

**Analysis:** Similar to the Budi95 comparison, this helps assess how different blending approaches affect price distribution characteristics. Understanding these differences is important for evaluating which blending strategies might offer better price stability characteristics while meeting environmental objectives.

### distribution_ron97.png
**Explanation:** This plot displays the probability distribution of RON97 premium fuel prices.

**Analysis:** Understanding the distribution of premium fuel prices helps analyze consumer choice behavior between fuel grades. If RON97 shows different distribution characteristics than RON95 (different volatility, skewness, etc.), it might explain observed patterns in fuel grade selection. This information is also relevant for understanding how premium fuel prices respond to economic cycles compared to regular grades.

## Partial Autocorrelation Function (PACF) Analysis

### pacf_diesel.png
**Explanation:** This partial autocorrelation function (PACF) plot shows the correlation between diesel prices and their lagged values after removing the effects of intermediate lags.

**Analysis:** While ACF shows total correlation at each lag, PACF shows the unique correlation after accounting for shorter lags. This is particularly useful for identifying the order of autoregressive (AR) terms in ARIMA modeling. Significant PACF values at specific lags suggest those lags should be included as autoregressive predictors. The PACF helps distinguish between direct and indirect relationships in time series data, leading to more parsimonious and accurate forecasting models.

### pacf_diesel_eastmsia.png
**Explanation:** This PACF plot shows the partial autocorrelation structure of diesel prices in East Malaysia.

**Analysis:** Comparing PACF structures between regions helps determine whether similar or different autoregressive models are needed for regional forecasting. Similar significant PACF lags would suggest comparable price memory structures, while different patterns might indicate region-specific factors influencing how past prices affect current prices. This supports development of appropriately tailored forecasting models for different regions.

### pacf_ron95.png
**Explanation:** This PACF plot shows the partial autocorrelation structure of RON95 fuel prices.

**Analysis:** For the most widely used fuel, understanding the PACF helps identify the appropriate complexity for forecasting models. The lags at which PACF shows significant values indicate how many past prices directly influence current prices after accounting for intermediate effects. This information is critical for building accurate short-to-medium term forecasts that inform trading decisions, budget planning, and policy analysis.

### pacf_ron95_budi95.png
**Explanation:** This PACF plot shows the partial autocorrelation structure of RON95 blended with biodiesel prices.

**Analysis:** Comparing PACF between blended and pure fuels helps assess how biodiesel affects the direct temporal relationships in pricing. Different significant PACF lags might indicate that biodiesel blending changes how quickly or strongly past prices influence current prices. This has implications for forecasting blended fuel prices and understanding how renewable fuel components interact with petroleum pricing dynamics.

### pacf_ron95_skps.png
**Explanation:** This PACF plot shows the partial autocorrelation structure of RON95 blended with SKPS variant prices.

**Analysis:** Similar to the Budi95 analysis, this helps evaluate how different blending formulations affect the direct autoregressive structure of fuel prices. Understanding these differences supports selection of appropriate forecasting models for different fuel formulations and helps assess whether blending introduces new temporal dependencies or modifies existing ones.

### pacf_ron97.png
**Explanation:** This PACF plot shows the partial autocorrelation structure of RON97 premium fuel prices.

**Analysis:** Comparing PACF between premium and regular grades helps understand whether they follow similar or different price formation processes. Different PACF structures might indicate that premium fuel prices respond differently to past prices, potentially due to different consumer bases, usage patterns, or linkages to different segments of the crude oil market. This information is relevant for developing grade-specific forecasting approaches and understanding market segmentation in fuel pricing.

## Trend Analysis

### trend_diesel.png
**Explanation:** This plot shows the trend component of diesel fuel prices extracted through decomposition or filtering techniques.

**Analysis:** The trend reveals the long-term direction of diesel prices after removing short-term fluctuations and seasonal patterns. An upward trend indicates sustained price increases over time, potentially reflecting rising crude oil costs, inflation, or structural changes in supply/demand. A downward trend would suggest improving supply conditions or decreasing demand. Understanding the trend is essential for long-term planning, investment decisions, and assessing the effectiveness of price stabilization policies over extended periods.

### trend_diesel_eastmsia.png
**Explanation:** This plot shows the trend component of diesel prices in East Malaysia.

**Analysis:** Comparing regional trends with national trends helps identify whether different regions experience similar long-term price pressures or divergent trends due to local factors. Similar trends would suggest common national/international drivers, while divergent trends might indicate region-specific economic development patterns, infrastructure investments, or local supply chain developments affecting long-term price trajectories.

### trend_ron95.png
**Explanation:** This plot shows the trend component of RON95 fuel prices.

**Analysis:** As the primary transportation fuel, understanding RON95's long-term trend is critical for assessing cumulative cost impacts on consumers and businesses over time. The trend helps separate persistent price movements from temporary fluctuations, enabling better evaluation of whether observed price changes represent temporary market conditions or more permanent shifts requiring policy responses. This analysis supports long-term energy planning and affordability assessments.

### trend_ron95_budi95.png
**Explanation:** This plot shows the trend component of RON95 blended with biodiesel prices.

**Analysis:** Comparing trends between blended and pure fuels helps evaluate how biodiesel affects long-term price trajectories. If the blended fuel shows a different trend slope, it might indicate that biodiesel blending provides some insulation from or amplification of long-term crude oil price movements. This information is relevant for assessing the long-term economic implications of biofuel mandates and blending policies.

### trend_ron95_skps.png
**Explanation:** This plot shows the trend component of RON95 blended with SKPS variant prices.

**Analysis:** Similar to the Budi95 comparison, this helps assess how different blending approaches influence long-term price behavior. Understanding trend differences between formulations supports optimization of fuel policies that balance multiple objectives including environmental goals, energy security, and price stability over extended time horizons.

### trend_ron97.png
**Explanation:** This plot shows the trend component of RON97 premium fuel prices.

**Analysis:** Comparing premium fuel trends with regular grade trends helps understand whether different market segments experience similar or divergent long-term price pressures. Different trends might indicate varying exposure to crude oil prices, different demand elasticity patterns, or different responsiveness to macroeconomic factors affecting vehicle fleet composition. This information is relevant for understanding segmentation in fuel markets and developing targeted policies for different consumer groups.

## Volatility Analysis

### volatility_diesel.png
**Explanation:** This plot shows the volatility (typically measured as rolling standard deviation or similar metric) of diesel fuel prices over time.

**Analysis:** Volatility measures indicate how much prices fluctuate around their trend. High volatility periods suggest greater uncertainty and risk for consumers, businesses, and traders. Analyzing volatility helps identify periods of market stress, such as during geopolitical events, natural disasters affecting supply chains, or major economic shifts. Understanding volatility patterns is crucial for risk management, setting appropriate buffer stocks, and designing effective price stabilization mechanisms.

### volatility_diesel_eastmsia.png
**Explanation:** This plot shows the volatility of diesel prices in East Malaysia over time.

**Analysis:** Comparing regional volatility with national volatility helps assess whether price risks are similar or different across geographical areas. Similar volatility patterns would suggest common national/international risk factors, while different patterns might indicate region-specific factors influencing price stability. This information is valuable for regional risk management strategies and understanding geographical differences in market resilience.

### volatility_ron95.png
**Explanation:** This plot shows the volatility of RON95 fuel prices over time.

**Analysis:** As the most widely consumed fuel, understanding RON95 volatility is critical for assessing price risk faced by the majority of consumers and businesses. Periods of high volatility can significantly impact household budgets and business operating costs. Analyzing what drives volatility spikes (global events, local supply issues, policy changes) helps in developing appropriate mitigation strategies and understanding the sources of price uncertainty in the transportation fuel market.

### volatility_ron95_budi95.png
**Explanation:** This plot shows the volatility of RON95 blended with biodiesel prices over time.

**Analysis:** Comparing volatility between blended and pure fuels helps assess how biodiesel blending affects price stability. If biodiesel blends show lower volatility, it might indicate that diversification into renewable fuels provides some insulation from petroleum market shocks. Conversely, higher volatility might suggest that biodiesel introduces new sources of price variability related to agricultural commodity markets. This information is relevant for evaluating the risk implications of biofuel policies.

### volatility_ron95_skps.png
**Explanation:** This plot shows the volatility of RON95 blended with SKPS variant prices over time.

**Analysis:** Similar to the Budi95 comparison, this helps evaluate how different blending formulations affect price volatility characteristics. Understanding these differences supports assessment of which blending strategies might offer better risk profiles while meeting environmental and performance objectives.

### volatility_ron97.png
**Explanation:** This plot shows the volatility of RON97 premium fuel prices over time.

**Analysis:** Analyzing premium fuel volatility helps understand whether different market segments experience similar or different levels of price uncertainty. Different volatility patterns might indicate varying exposure to market shocks, different consumer responsiveness to price changes, or different linkages to specific crude oil fractions. This information is relevant for understanding how price risk is distributed across different consumer groups and for developing targeted risk management approaches.

## Conclusion

This exploratory analysis of fuel price data using various visualization techniques has provided comprehensive insights into the characteristics of different fuel price series in Malaysia. Key findings from the analysis include:

1. **Temporal Dependencies:** ACF and PACF analyses reveal the memory structures in fuel prices, helping identify appropriate forecasting models and understanding how past prices influence current values.

2. **Price Distribution:** Distribution analyses show the shape and characteristics of price movements, revealing whether prices fluctuate symmetrically or with bias toward increases/decreases, and how frequently extreme events occur.

3. **Trend Components:** Trend analyses isolate long-term price movements from short-term fluctuations, enabling assessment of sustained price pressures over time.

4. **Seasonal Patterns:** Decomposition reveals regular repeating patterns in prices that may relate to driving seasons, holidays, or other cyclical factors.

5. **Volatility Characteristics:** Volatility analyses identify periods of high and low price uncertainty, helping understand risk dynamics in fuel markets.

6. **Regional Differences:** Comparisons between peninsula Malaysia and East Malaysia (where available) reveal geographical variations in price behavior.

7. **Blending Effects:** Analyses of biodiesel and special blend fuels show how renewable fuel components affect price dynamics compared to conventional fuels.

8. **Grade Differentiation:** Comparisons between fuel grades (RON95, RON97, diesel) reveal how different market segments experience fuel price changes.

These insights collectively support better understanding of fuel market dynamics, improved forecasting accuracy, more effective policy design, and enhanced risk management strategies for stakeholders across the fuel supply chain from producers to consumers.