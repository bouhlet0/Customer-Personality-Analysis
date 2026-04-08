# Customer Personality Analysis

End-to-end customer analytics pipeline integrating PCA-based feature compression, KMeans clustering, statistical campaign effectiveness testing, and supervised ML models for CLV and response prediction. Built on a 2,206-customer retail dataset with 29 behavioral and demographic features, producing customer segmentation labels, conversion propensity scores, and revenue estimates for targeted marketing optimization.
---

## Business Questions

- Did the marketing campaigns work, and on which customers?
- Can we segment the customer base into meaningful, actionable groups?
- Which customers are at risk of churning, and how do we target them?
- Can we predict how much a customer will spend, what they will buy, and through which channel?

---

## Dataset

**Source:** [Customer Personality Analysis - Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

2,240 customers, 29 features covering demographics, purchase history across 6 product categories, 3 purchase channels, and 5 historical marketing campaigns.

| Feature Group | Variables |
|---|---|
| Demographics | Age, Income, Education, Marital Status, Children |
| Spending | Wines, Fruits, Meat, Fish, Sweets, Gold |
| Channels | Web, Store, Catalog purchases |
| Campaigns | AcceptedCmp1–5, Response (last campaign) |
| Engagement | Recency, Web visits, Deals purchased |

After cleaning (NaN removal, age outliers >100, income outlier at 666,666): **2,206 customers**.

---

## Project Structure

```
├── Customer_Personality_Analysis.ipynb
├── README.md
├── requirements.txt
├── dataset/
│   └── marketing_campaign.csv
└── models/
    ├── scaler.pkl
    ├── pca.pkl
    ├── kmeans_k3.pkl
    ├── response_classifier_gbc.pkl
    ├── response_classifier_gbc_smote.pkl
    ├── response_classifier_xgb.pkl
    ├── high_spender_classifier.pkl
    ├── clv_regressor.pkl
    ├── churn_risk_classifier.pkl
    ├── channel_preference_classifier.pkl
    ├── sweets_buyer_classifier.pkl
    ├── propensity_models.pkl
    ├── channel_label_encoder.pkl
    └── category_label_encoder.pkl
```

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of income (right-skewed, mean ~52k), age (mean 44), spending (mean 607, high variance)
- Outlier detection and removal via boxplots
- Correlation heatmaps before and after ordinal encoding of categorical features
- Scatter grid analysis of key variable pairs: income vs deals, children vs spending, web visits vs web purchases

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `Age_at_enrollment` | Year of enrollment − Year of birth |
| `Children` | Kidhome + Teenhome |
| `Total_Spending` | Sum of all 6 product categories |
| `Total_Purchases` | Sum of web + store + catalog purchases |
| `Channel_ratio` | Per-channel share of total purchases |
| `ChannelsUsed` | Count of channels used (1–3) |
| `TotalAcceptedCmp` | Sum of campaigns 1–5 accepted |
| `Education_Ordinal` | Ordinal encoding: Basic=0 → PhD=4 |
| `Is_Partnered` | Binary: Married/Together or Single |
| `Deal_Sensitivity` | NumDealsPurchases / Total_Purchases |
| `Web_Conversion_Rate` | NumWebPurchases / NumWebVisitsMonth |
| `Tenure_Days` | Days since enrollment |
| `RFM_Score` | Recency + Frequency + Monetary quintile scores |

### 3. Campaign Effectiveness Analysis
Mann-Whitney U tests (non-parametric) with effect size (rank-biserial correlation) per campaign, comparing acceptors vs rejectors on income, total spending, and age. All campaigns show significant income and spending differences (p < 0.05) except Campaign 3 (p_income = 0.48).

| Campaign | Accept Rate | p_Income | p_Spending |
|---|---|---|---|
| Cmp1 | 6.4% | < 0.001 | < 0.001 |
| Cmp2 | 1.4% | < 0.001 | < 0.001 |
| Cmp3 | 7.4% | 0.484 | 0.090 |
| Cmp4 | 7.4% | < 0.001 | < 0.001 |
| Cmp5 | 7.3% | < 0.001 | < 0.001 |
| Response | 15.1% | < 0.001 | < 0.001 |

### 4. Customer Segmentation
- StandardScaler → PCA (9 components, 90% variance retained, PC1 explains 40.3%)
- KMeans with elbow + silhouette selection → **k=3** chosen (silhouette peaks at k=2 at 0.293 but produces insufficient granularity; k=3 silhouette=0.191 selected for business interpretability)
- Cluster stability validated via 50-run bootstrap ARI: **KMEANS: 0.687 +- 0.239, GMM: 0.761 +- 0.270**
- KMeans vs GMM comparison: GMM ARI=0.800 ± 0.248, BIC monotonically decreasing (no convergence), KMeans retained for interpretability

| Cluster | Size | Avg Income | Avg Spend | Profile |
|---|---|---|---|---|
| 0 | 891 | 71,520 | 1,233 | Affluent, childless, high campaign response (59%), multichannel |
| 1 | 739 | 43,625 | 290 | Mid-income, moderate engagement, more children |
| 2 | 576 | 32,369 | 53 | Low income, family-oriented, deal-sensitive, low campaign response (4%) |

### 5. Predictive Models

| Model | Target | Algorithm | Key Metric |
|---|---|---|---|
| Campaign Response | Binary (last campaign) | GBM + SMOTE + XGBoost | AUC = 0.877 |
| High Spender | Top 25% by spending (≥1,049) | GBM | Precision/Recall: 0.85/0.86 |
| CLV Regression | Total spending () | Gradient Boosting Regressor | R²=0.887, MAE=110 |
| Churn Risk | Recency proxy (≥74 days) | XGBoost + GridSearch | AUC = 0.616 |
| Channel Preference | Web / Store / Catalog | XGBoost multiclass | AUC = 0.780 |
| Product Propensity | 6 binary models (above-median buyer) | XGBoost | AUC range: 0.90–0.98 |
| Sweets Buyer | Binary (any sweet purchase) | GBM | AUC = 0.770 |

**Top response model features (SHAP + feature importance):** Income, Cluster, Total_Purchases, TotalAcceptedCmp, NumCatalogPurchases.

### 6. Spend Share Prediction
Regression on per-category share of basket:

| Category | R² | Interpretation |
|---|---|---|
| Wines | 0.531 | Strong demographic signal → targetable |
| Gold | 0.509 | Strong demographic signal → targetable |
| Meat | 0.314 | Moderate - household composition driven |
| Fish | 0.265 | Moderate |
| Fruits | 0.182 | Weak |
| Sweets | −0.499 | Unpredictable from demographics (impulse/occasion driven) |

Sweet product spend share remains unpredictable even after log-transformation (R²=−0.004), confirmed by separate investigation. Binary classifier (buys/doesn't buy sweets) achieves AUC=0.770 - use this instead of spend regression for sweets targeting.

### 7. Additional Analyses
- **RFM Segmentation:** 330 Best Customers (avg spend 1,336, TotalAcceptedCmp=0.67), 171 Lost customers (avg spend 38, churn proba=0.61)
- **Pareto Analysis:** Top 37.9% of customers generate 80% of revenue
- **Deal Sensitivity:** High-sensitivity customers earn 37k avg vs 72k for low-sensitivity - discounting erodes margins without loyalty benefit
- **Product Co-purchase Lift Matrix:** Wines × Meat, Wines × Gold show above-chance co-purchase lift → bundle targeting opportunities
- **Cumulative Gains / Lift Curve:** Model captures ~60% of responders by contacting top 30% of customers (vs 30% under random targeting)
- **A/B Test Power Analysis:** Detecting a 5pp campaign lift requires n=907 per group; current dataset (n≈2,200) is sufficient for lifts ≥8%
- **Cluster Stability (Bootstrap ARI):** 50-run bootstrap confirms moderate-to-good stability; high standard deviation (±0.22) reflects genuine customer continuum rather than hard-boundary clusters
- **Error Analysis:** False negatives (missed responders) average 899 spending vs 603 for correct predictions - the model disproportionately misses high-value customers

---

## Key Findings

1. **Income is the dominant driver** across all models. High-income customers respond to campaigns, spend more, and are reliably targetable.

2. **Cluster 0 (affluent, ~40% of base) generates the majority of campaign responses and revenue.** Campaign 2 is the exception - it uniquely over-indexes on high-income customers (median accepted income 896k) suggesting a premium/niche positioning.

3. **CLV is highly predictable (R²=0.887)** - customer lifetime value can be reliably estimated from demographic and behavioral features at acquisition.

4. **Sweet products are the anomaly** - the only category where spend intensity shows no demographic signal. Binary targeting is the only viable ML approach; spend amount requires transaction-level or promotional data.

5. **150 high-value customers are at churn risk** (Cluster 0 members with Recency ≥74 days). Preferred channel for most is Store or Web - win-back campaigns should be routed accordingly.

6. **The 80/20 rule does not hold strictly** - closer to a 38/80 split, meaning revenue is less concentrated than typical retail benchmarks.

---

## Limitations

- **No true churn label** - churn is proxied by recency quartile (≥74 days). AUC of 0.616 after grid search reflects the difficulty of predicting a proxy rather than an observed event.
- **Cross-sectional data** - no transaction-level timestamps. True CLV, basket evolution, and seasonal effects cannot be modeled.
- **Catalog channel underrepresented** (158 customers, 7%) - channel preference model performs poorly on Catalog class (recall=0.03). Insufficient data to learn Catalog buyer patterns.
- **Cluster boundaries are soft** - silhouette scores of 0.19–0.29 indicate overlapping segments. Clusters are operationally useful groupings, not ground-truth customer types.
- **Sweet spend unpredictability** - requires promotional history or visit-level data to resolve.

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
missingno
scipy
scikit-learn
statsmodels
shap
xgboost
imbalanced-learn
joblib
```

Install with:
```bash
pip install -r requirements.txt
```

