# Bivariate Analysis Report

This report presents the findings from the bivariate analysis conducted on various transportation and demographic datasets. Each visualization explores relationships between two variables to uncover patterns, correlations, and potential causal relationships.

## Fuel Price and Ridership Analysis

### fuel_ridership_correlation.png
**Explanation:** This visualization shows the correlation between fuel prices and public transportation ridership over time. It likely includes a scatter plot with a trend line or a time series comparison of both variables.

**Analysis:** The relationship between fuel prices and ridership is typically inverse - as fuel prices increase, public transportation becomes more attractive relative to private vehicle use, leading to higher ridership. This analysis helps quantify the elasticity of demand for public transit with respect to fuel costs. Strong correlation would suggest fuel pricing is a significant lever for influencing mode choice, while weak correlation might indicate other factors dominate ridership decisions.

### fuel_ridership_lagged_effects.png
**Explanation:** This plot examines the time-lagged relationship between fuel price changes and subsequent ridership responses. It likely shows correlation coefficients at different time lags (e.g., 0, 1, 2, 3 months) to capture delayed behavioral responses.

**Analysis:** Transportation mode choices may not change immediately with fuel price fluctuations due to factors like vehicle ownership commitments, travel planning horizons, and habitual behavior. Identifying the optimal lag period helps understand how quickly riders respond to economic signals. Short lags suggest immediate discretionary travel changes, while longer lags might reflect more substantive changes like vehicle purchases or residential relocation decisions.

### fuel_ridership_mode_substitution.png
**Explanation:** This visualization likely breaks down ridership changes by transportation mode (bus, rail, etc.) in response to fuel price variations, showing which modes benefit most from fuel price increases.

**Analysis:** Different transit modes have varying levels of substitutability for automobile travel. Rail systems with fixed guideways and higher capacity might capture more discretionary trips during high fuel price periods compared to bus services. Understanding these differential responses helps transit agencies allocate resources effectively during economic fluctuations and target service improvements to modes with highest potential ridership gains.

### fuel_ridership_scatter_regression.png
**Explanation:** This scatter plot displays individual data points of fuel prices versus ridership with a superimposed regression line, quantifying the statistical relationship between the two variables.

**Analysis:** The regression analysis provides a measure of sensitivity - how much ridership changes per unit change in fuel price. The slope indicates elasticity, while R-squared shows how much of ridership variance is explained by fuel prices alone. Outliers in this plot might reveal special circumstances (economic downturns, major service changes, or extreme weather events) that disrupt the typical fuel price-ridership relationship.

### fuel_ridership_threshold.png
**Explanation:** This visualization likely identifies threshold effects in the fuel price-ridership relationship, showing whether ridership responds differently below or above certain fuel price levels.

**Analysis:** Behavioral responses to fuel prices may be non-linear, with minimal ridership impact at low price levels but pronounced effects once prices exceed psychological or economic thresholds (e.g., $4/gallon). Identifying such thresholds helps policymakers understand when fuel taxation policies might significantly influence travel behavior and when price changes are unlikely to alter mode choice substantially.

## GTFS (General Transit Feed Specification) Analysis

### gtfs_admin_coverage.png
**Explanation:** This image likely shows the relationship between administrative boundaries (city limits, county jurisdictions) and GTFS transit service coverage, possibly displaying what percentage of each administrative area is served by public transit.

**Analysis:** Transit equity and accessibility are often measured by how well service covers populated areas within administrative boundaries. Gaps in coverage might indicate underserved communities or jurisdictional fragmentation challenges. This analysis helps identify areas where service expansion could improve access to transportation for residents, particularly in low-income or transit-dependent populations.

### gtfs_osm_connectivity.png
**Explanation:** This visualization probably compares GTFS transit network connectivity with OpenStreetMap (OSM) road network connectivity, showing how well the transit network integrates with the overall transportation network.

**Analysis:** Effective public transportation requires good first/last mile connectivity. Comparing GTFS routes with OSM pedestrian and bicycle networks reveals how easily riders can access transit stops/stations. Poor connectivity might explain low ridership despite service presence, highlighting needs for infrastructure improvements like sidewalks, crosswalks, or bike lanes near transit stops.

### gtfs_population_coverage.png
**Explanation:** This plot likely shows what proportion of the population lives within walking distance (typically ¼ to ½ mile) of transit stops or stations, based on GTFS data and population demographics.

**Analysis:** Population coverage is a key metric for transit accessibility and equity. High coverage percentages indicate good service reach, while low coverage suggests many residents lack convenient transit access. This analysis helps prioritize areas for service expansion or identify populations that remain transit-dependent despite limited service availability.

### gtfs_ridership_frequency.png
**Explanation:** This visualization likely examines the relationship between transit service frequency (headways) and ridership levels across different routes or time periods.

**Analysis:** Service frequency is a critical determinant of transit attractiveness and ridership. Higher frequency reduces wait times and increases spontaneity of use. This analysis typically shows a positive correlation between frequency and ridership, though with diminishing returns at very high frequencies. Understanding this relationship helps optimize resource allocation between frequency improvements and service coverage expansion.

### gtfs_ridership_route_length.png
**Explanation:** This plot probably shows how ridership varies with route length, possibly examining whether longer routes attract more riders due to serving more destinations or shorter routes perform better due to focused service.

**Analysis:** Route length influences both operating costs and potential ridership. Very short routes may not provide sufficient destination access, while extremely long routes might suffer from low productivity if they serve sparse populations. The optimal route length balances coverage needs with operational efficiency, often favoring routes that connect major activity centers without excessive deviation.

### gtfs_ridership_stops.png
**Explanation:** This visualization likely explores the relationship between the number of stops along a route and its ridership, examining whether more stops (better access) or fewer stops (faster travel) correlate with higher ridership.

**Analysis:** Stop spacing involves a trade-off between accessibility and travel time. Too many stops slow service and discourage longer trips, while too few stops limit access for potential riders. This analysis helps identify optimal stop spacing patterns that maximize ridership by balancing access needs with service efficiency, often showing a concave relationship where moderate stop density yields highest ridership.

### gtfs_ridership_summary.png
**Explanation:** This summary visualization likely presents multiple bivariate relationships from the GTFS data in a compact format, possibly showing correlations between various service characteristics (frequency, speed, coverage) and ridership metrics.

**Analysis:** By synthesizing multiple relationships, this view helps identify which service characteristics most strongly influence ridership. Transit agencies can use such summaries to prioritize improvements that offer the greatest ridership gains per investment dollar, focusing on high-impact factors like frequency in key corridors or coverage in underserved areas.

### gtfs_walking_analysis.png
**Explanation:** This image likely analyzes walking accessibility to transit stops, showing relationships between walkability metrics (sidewalk coverage, intersection density, etc.) and transit usage or potential catchment areas.

**Analysis:** Walking is the predominant access mode for most transit users. This analysis evaluates how pedestrian infrastructure quality affects transit ridership. Areas with good walking conditions typically show higher transit utilization, while poor pedestrian environments create barriers to transit access even when service is nearby. Findings can guide investments in sidewalk networks, crossings, and path connectivity to improve transit accessibility.

## Holiday Analysis

### holiday_gtfs_alignment.png
**Explanation:** This visualization likely compares GTFS transit schedules on holidays versus regular days, showing service level adjustments made for holiday periods.

**Analysis:** Transit agencies typically reduce service on holidays due to lower demand, but the extent and pattern of reductions vary. This analysis helps evaluate whether service adjustments match actual holiday travel patterns. Over-reduction can strand essential workers or holiday travelers, while insufficient reduction wastes resources. Understanding holiday travel behavior helps optimize holiday service levels.

### holiday_ridership_duration_effect.png
**Explanation:** This plot likely examines how ridership patterns change throughout holiday periods, possibly showing how ridership evolves before, during, and after holiday events.

**Analysis:** Holiday travel behavior often shows complex patterns with pre-holiday travel increases (shopping, preparation), variable patterns during holidays (depending on holiday type), and post-holiday returns. Understanding these temporal dynamics helps agencies allocate resources effectively throughout holiday periods rather than applying uniform service levels.

### holiday_ridership_overall_impact.png
**Explanation:** This visualization probably shows the net impact of holidays on total ridership compared to regular days, possibly comparing different holiday types or measuring percentage changes.

**Analysis:** Different holidays affect ridership differently based on their nature (religious, national, observance) and typical activities associated with them. Some holidays (like New Year's Day) might see substantial ridership drops, while others (like Independence Day with fireworks events) might see increases in specific corridors. This analysis helps agencies anticipate and plan for holiday-specific travel demands.

### holiday_ridership_pre_post_profile.png
**Explanation:** This image likely displays ridership profiles comparing periods before and after holidays, showing how travel patterns transition into and out of holiday periods.

**Analysis:** The transition periods around holidays often show anticipatory or compensatory travel patterns. Pre-holiday increases might reflect preparation travel, while post-holiday patterns could show return travel or extended leisure activities. Understanding these profiles helps fine-tune service levels during shoulder periods of holidays.

### holiday_ridership_type_analysis.png
**Explanation:** This visualization likely breaks down holiday ridership impacts by holiday type (federal holidays, religious observances, etc.) or by purpose of travel (work, shopping, recreation).

**Analysis:** Not all holidays impact transit equally. Federal holidays affecting work schedules typically show different patterns than religious holidays centered around home-based observances. This disaggregated analysis helps agencies understand the specific drivers of holiday travel changes and tailor service adjustments accordingly.

### holiday_ridership_weekday_vs_weekend.png
**Explanation:** This plot likely compares how holidays affect weekday versus weekend ridership patterns, showing whether holidays make weekdays more weekend-like or create unique patterns.

**Analysis:** Holidays can blur the traditional weekday/weekend distinction in travel patterns. Some holidays might suppress weekday commuting while maintaining or increasing weekend-style discretionary travel. This analysis helps agencies understand whether to apply weekend-level, weekday-level, or specialized holiday service patterns.

## OpenStreetMap (OSM) and Administrative Analysis

### osm_admin_diversity.png
**Explanation:** This visualization likely examines the diversity of point of interest (POI) types or land use categories from OpenStreetMap within different administrative areas, possibly showing entropy or variety measures.

**Analysis:** Land use diversity is often associated with higher walking and transit ridership because it creates more trip origins and destinations within walkable distances. Areas with mixed-use development (residential combined with retail, offices, services) typically support more sustainable transportation than homogeneous residential or industrial zones. This analysis helps identify areas where zoning policies might be adjusted to encourage more mixed-use, transit-supportive development.

## Population Analysis

### population_admin_distribution.png
**Explanation:** This image likely shows how population is distributed across different administrative jurisdictions (cities, counties, districts) and possibly compares this distribution with transit service allocation.

**Analysis:** Understanding population distribution relative to administrative boundaries is crucial for equitable transit planning and funding allocation. Discrepancies between where people live and where transit resources are concentrated can highlight equity issues. This analysis helps identify whether service provision aligns with population needs or whether certain administrative areas are over- or under-served relative to their populations.

### population_density_overview.png
**Explanation:** This visualization probably displays population density patterns across the study area, potentially showing density gradients or clusters.

**Analysis:** Population density is a fundamental predictor of transit viability and ridership potential. Higher densities generally support more frequent and cost-effective transit service due to larger concentrations of potential riders within service areas. This analysis helps identify corridors and areas with sufficient density to support various transit service levels, guiding investment decisions toward locations with highest ridership potential per dollar invested.

### population_osm_accessibility.png
**Explanation:** This image likely examines the relationship between population locations and accessibility to amenities or services mapped in OpenStreetMap, showing what proportion of population has walkable access to various POIs.

**Analysis:** Beyond transit-specific accessibility, overall access to daily needs (groceries, healthcare, education, etc.) influences travel behavior and quality of life. Populations with good walkable access to essential services may generate fewer auto trips even without excellent transit. This analysis helps identify "complete neighborhoods" where residents can meet many needs without driving, highlighting areas of intrinsic sustainability.

### population_ridership_correlation.png
**Explanation:** This visualization probably shows the correlation between population characteristics (density, demographics, etc.) and transit ridership levels across different geographic areas.

**Analysis:** Understanding which population factors most strongly predict ridership helps target service and marketing efforts effectively. While density is often a strong predictor, specific demographic factors (age, income, auto ownership) can modify this relationship. This analysis supports data-driven decisions about where to invest in transit improvements for maximum ridership impact.

### population_spatial_heatmap.png
**Explanation:** This image likely presents a heatmap showing population density or concentration patterns across the study area, highlighting population clusters and sparse areas.

**Analysis:** Spatial population patterns reveal where potential transit markets exist. Identifying dense population clusters helps locate optimal areas for transit investment, while understanding sparse areas sets realistic expectations for service levels needed. This analysis is foundational for aligning transit infrastructure with actual population distribution to maximize accessibility and ridership potential.

### population_transit_coverage.png
**Explanation:** This visualization likely shows what percentage of the population in different areas has access to transit service within a reasonable distance (typically walking distance).

**Analysis:** Transit coverage metrics are essential for evaluating equity and accessibility. High coverage indicates good service reach, while low coverage highlights transportation disadvantaged populations. This analysis helps prioritize areas for service expansion to improve access for underserved communities and supports Title VI and environmental justice analyses in transportation planning.

### population_urban_rural.png
**Explanation:** This image likely compares population characteristics, densities, or transit usage patterns between urban and rural areas within the study region.

**Analysis:** Urban and rural areas present fundamentally different challenges and opportunities for transportation planning. Urban areas typically support higher transit ridership due to density and mixed-use development, while rural areas face challenges of long distances and low density. Understanding these differences helps tailor appropriate transportation solutions—more transit-oriented approaches in urban areas versus demand-responsive or rural transit models in sparsely populated regions.

### population_walking_exposure.png
**Explanation:** This visualization probably examines the relationship between population locations and their exposure to walking infrastructure or walkable environments, possibly showing what proportion of population lives in areas with good sidewalk coverage or intersection density.

**Analysis:** Walking exposure affects not just transit access but overall active transportation potential and public health. Populations with good walking infrastructure tend to utilize active transportation more for short trips and transit access. This analysis helps identify neighborhoods where walking improvements would yield significant benefits in terms of transportation options, health outcomes, and equity.

## Rainfall Analysis

### rainfall_ridership_category.png
**Explanation:** This visualization likely categorizes days by rainfall intensity (none, light, moderate, heavy) and shows average ridership for each category.

**Analysis:** Weather significantly impacts travel behavior, with precipitation often reducing discretionary travel and affecting mode choice. This analysis quantifies how different rainfall intensities influence ridership, helping agencies understand weather-related demand variability. Findings can inform service planning (e.g., whether to maintain full service during heavy rain when ridership drops) and communications strategies for adverse weather conditions.

### rainfall_ridership_correlation.png
**Explanation:** This scatter plot or time series shows the relationship between rainfall measurements and ridership levels, likely demonstrating the correlation between wet weather and transit usage.

**Analysis:** The rainfall-ridership relationship is typically negative but complex. Light rain might have minimal impact, while heavy rain or storms can substantially reduce ridership as people avoid travel or shift modes. Understanding this relationship helps agencies predict weather-related demand fluctuations and adjust resources accordingly. In some cases, extreme weather might increase ridership if people avoid driving due to hazardous conditions.

### rainfall_ridership_extreme_events.png
**Explanation:** This image likely focuses on ridership patterns during extreme rainfall events (heavy storms, flooding) compared to normal conditions or other weather events.

**Analysis:** Extreme weather events can disrupt normal travel patterns significantly, sometimes causing mode shifts or travel suppression. Analyzing ridership during these events helps evaluate transportation system resilience and identify vulnerabilities. Findings can inform emergency service planning, real-time passenger information during disasters, and infrastructure investments to improve weather resilience.

### rainfall_ridership_seasonal.png
**Explanation:** This visualization probably examines how the rainfall-ridership relationship varies by season, showing whether weather impacts differ between summer, winter, spring, and fall.

**Analysis:** The impact of rainfall on travel behavior often varies seasonally due to factors like temperature, daylight hours, and typical activity patterns. For example, rain might have a different effect in summer (when outdoor activities are common) versus winter (when people may already be limiting discretionary travel). This seasonal analysis helps refine weather-adjusted service planning throughout the year.

### rainfall_ridership_weekday_weekend.png
**Explanation:** This plot likely compares how rainfall affects ridership on weekdays versus weekends, recognizing that travel purposes and flexibility differ between these periods.

**Analysis:** Weekday travel often involves more fixed commitments (work, school) that are less weather-flexible, while weekend travel includes more discretionary activities that may be postponed or canceled due to poor weather. This analysis helps understand whether weather impacts are concentrated on discretionary weekend travel or affect essential weekday travel as well, informing different service strategies for weekdays versus weekends during inclement weather.

### rainfall_walking_mobility.png
**Explanation:** This visualization likely shows how rainfall affects walking mobility or pedestrian activity, potentially examining sidewalk usage or pedestrian counts in relation to precipitation.

**Analysis:** Walking is particularly sensitive to weather conditions, with rain often substantially reducing pedestrian activity. Understanding this relationship is crucial for transit planning since most transit trips begin and end with walking segments. Poor walking conditions during rain can effectively isolate populations from transit access even when service is available, highlighting the importance of covered walkways, good drainage, and other pedestrian improvements to maintain year-round accessibility.

## Walking Analysis

### walking_admin_disparity.png
**Explanation:** This image likely examines disparities in walking infrastructure, safety, or accessibility across different administrative jurisdictions, showing inequities in pedestrian environments.

**Analysis:** Walking infrastructure quality often varies significantly across jurisdictional boundaries due to differences in funding priorities, development patterns, or maintenance capabilities. Identifying these disparities helps target investments to underserved areas and promotes equity in active transportation access. Such analysis supports broader goals of creating walkable communities regardless of administrative boundaries.

### walking_osm_accessibility.png
**Explanation:** This visualization probably shows the accessibility of destinations or services via walking networks derived from OpenStreetMap, measuring what proportion of population can walk to various amenities within reasonable distances.

**Analysis:** Walking accessibility to daily needs (groceries, schools, parks, healthcare) is a key component of sustainable, healthy communities. This analysis helps identify "walkable neighborhoods" where residents can accomplish routine trips without automobiles, reducing vehicle emissions and promoting public health. Findings can guide land use policies, infrastructure investments, and development regulations to enhance walkability.

## Conclusion

This bivariate analysis has explored numerous relationships between transportation system characteristics, demographic factors, weather patterns, and service attributes. The insights gained from these analyses can inform:

1. **Service Planning:** Understanding which service characteristics most strongly influence ridership helps prioritize investments
2. **Equity Analysis:** Identifying underserved populations and areas supports equitable resource allocation
3. **Resilience Planning:** Evaluating weather and holiday impacts prepares the system for variable conditions
4. **Land Use Coordination:** Analyzing population patterns and walkability supports transit-oriented development
5. **Performance Measurement:** Establishing baseline relationships helps evaluate the effectiveness of future interventions

Each relationship revealed in these visualizations contributes to a more nuanced understanding of the factors shaping transportation behavior and system performance in the study region.