# Machine Learning Project: Device Life Expectancy Prediction

**Author:** Saksham Jain 
**Course:** Machine Learning (Assignment 1)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#eda)
4. [Data Preprocessing](#preprocessing)

   * Outlier Handling
   * Train/Test Split
   * Feature Scaling and Transformation
5. [Model Development](#model-development)

   * Candidate Models and Baseline Selection
   * Cross‑Validation Strategy
6. [Baseline Model: Random Forest Regression](#baseline-model)
7. [Regularization and Hyperparameter Tuning](#hyperparameter-tuning)
8. [Performance Analysis](#performance-analysis)
9. [Conclusions and Limitations](#conclusions)
10. [References](#references)

---

## <a name="introduction"></a>1. Introduction

This project aims to predict the life expectancy (in months) of a device based on a variety of manufacturing and quality‑control specifications. Starting from raw data ingestion, the workflow includes exploratory analysis, preprocessing, model selection, evaluation, and tuning. The final deliverable is a predictive model whose performance and generalisation ability are critically assessed.

---

## <a name="dataset"></a>2. Dataset

The dataset (`Data_set.csv`) comprises 2,071 records and 24 initial columns. Key variables include:

* **TARGET\_LifeExpectancy**: Device life expectancy (months) — target variable.
* **Company**, **Year**, **Company\_Status** (developed=1/developing=0)
* Probabilistic measures per 1,000 samples:

  * `Company_Confidence`, `Company_device_confidence`, `Device_confidence`
  * `Device_returen`, `Test_Fail`, `Engine_Cooling`, `Obsolescence`, `Engine_failure`
* Quality and financial metrics:

  * `PercentageExpenditure`, `TotalExpenditure`, `STRD_DTP`
* Technical readings:

  * `Gas_Pressure`
* Country‑level indicators:

  * `GDP`, `Product_Quantity`, `IncomeCompositionOfResources`, `RD`
* Failure prevalence trends:

  * `Engine_failure_Prevalence`, `Leakage_Prevalence`

The column **ID** was dropped as a non‑informative index.

---

## <a name="eda"></a>3. Exploratory Data Analysis (EDA)

* **Summary statistics** (`.info()`, `.describe()`): confirmed 2,071 non‑null entries across 23 features.
* **Distribution and skewness**: identified heavy skew in multiple probabilistic and financial columns.
* **Histograms & box plots**: visualised non‑normal distributions, outliers (notably in per‑1,000 metrics), and differences in life expectancy by company status.
* **Scatter plots**: detected linear vs. non‑linear relationships between features and life expectancy.
* **Correlation heatmap**: highlighted strong positive correlations among confidence measures and with the target, though non‑linear dependencies required models beyond simple linear approaches.

---

## <a name="preprocessing"></a>4. Data Preprocessing

### 4.1 Outlier Handling

* Detected values >1,000 in `Device_returen`, `Engine_Cooling`, `Obsolescence`.
* Applied **RobustScaler** to these columns to mitigate outlier influence.

### 4.2 Train/Test Split

* Separated features (`X`) and target (`y`).
* Used `train_test_split` (80% train, 20% test, shuffled) to create reproducible splits.

### 4.3 Feature Scaling and Transformation

* **MinMaxScaler** for features without extreme skew.
* **PowerTransformer** (Yeo–Johnson) on heavily skewed features, followed by MinMax scaling to align ranges.
* Verified consistent distributions between train and test sets via overlaid histograms.

---

## <a name="model-development"></a>5. Model Development

### Candidate Models

1. **Linear Regression**
2. **Polynomial (degree 2) Regression**
3. **Decision Tree Regression**
4. **Random Forest Regression**

### Cross‑Validation Strategy

* **K‑Fold CV** with *k*=5, shuffled, fixed random seed.
* Metrics:

  * **RMSE** (root mean squared error)
  * **R² score**
  * **Generalisation gap** (test RMSE minus train RMSE)

---

## <a name="baseline-model"></a>6. Baseline Model: Random Forest Regression

* **Choice rationale:** best trade‑off between RMSE and R² on the validation/test splits, capturing observed non‑linear relationships.
* **Performance**:

  * Test RMSE ≈ 2.83 months
  * Test R² ≈ 0.90
  * Train RMSE ≈ 1.14, Train R² ≈ 0.99
  * Generalisation gap \~1.70

---

## <a name="hyperparameter-tuning"></a>7. Regularization and Hyperparameter Tuning

To address overfitting and reduce the generalisation gap, we tuned:

* **n\_estimators**, **max\_depth**, **min\_samples\_split**, **min\_samples\_leaf**, **max\_features**, **bootstrap**, **min\_impurity\_decrease**
* Grid search (5‑fold CV) yielded a regularised forest with:

  * Reduced depth, controlled leaf size, optimized feature sampling, and bootstrap disabled.
* **Post‑tuning performance**:

  * Test RMSE ≈ 2.90
  * Train RMSE ≈ 2.70
  * Generalisation gap ≈ 0.59

---

## <a name="performance-analysis"></a>8. Performance Analysis

* **10‑fold CV** on the tuned model:

  * Mean RMSE ≈ 3.18 ± 0.26
  * Mean R² ≈ 0.89 ± 0.02
* **Final Test Set**:

  * RMSE ≈ 3.27
  * R² ≈ 0.88
* **Evaluation metrics**:

  * RMSE for magnitude of typical error
  * R² for proportion of variance explained
  * Generalisation gap for overfitting assessment

---

## <a name="conclusions"></a>9. Conclusions and Limitations

* **Conclusion:** A regularised Random Forest Regression model provides accurate, robust device life expectancy predictions, effectively handling non‑linear patterns and outliers.
* **Real‑world applicability:**

  * Suitable for manufacturing quality control, warranty forecasting, and reliability assessment.
* **Limitations:**

  * Model interpretability is limited (“black box”).
  * Computational cost increases with data size and hyperparameter complexity.
  * Assumes feature independence; performance may degrade with multicollinearity.
  * Reliant on data quality; noisy or missing inputs can impact accuracy.

---

## <a name="references"></a>10. References

1. **Scikit‑Learn Documentation**: RandomForestRegressor (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
2. Breiman, L. (2001). “Random Forests.” *Machine Learning* (https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
3. Brownlee, J. “Random Forest Ensemble in Python.” Machine Learning Mastery ((https://machinelearningmastery.com/random-forest-ensemble-in-python/))
4. Alavi, A. “RMIT COSC2673/2793 Machine Learning” (https://bitbucket.org/alavi_a/rmit_cosc_2673_2793-2310/src/master/)
