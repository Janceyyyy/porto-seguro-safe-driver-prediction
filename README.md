# Porto-Seguro Safe Driver Prediction

## Overview



This project entailed a comprehensive analysis and predictive modeling of drivers’ data, over 4GB in size, from multiple sources. The main objective was to forecast the likelihood of claim initiation among drivers, a key insight for insurance companies. Leveraging the power of Python for data extraction and transformation, we were able to collate and preprocess the data effectively.

The cornerstone of this predictive analytics was the implementation of the XGBoost model, which resulted in a commendable AUC of 0.89, signifying a high level of predictive accuracy. To refine our model further, we employed 5-fold cross-validation alongside GridSearchCV, which led to an enhancement in prediction performance by 4%, as measured by the AUC metric.

I took a deep dive into the impact of drivers' previous histories on claim initiation. Through meticulous analysis, I explored the interactive effects of various factors from the angle of policy renewals. Utilizing SHAP (SHapley Additive exPlanations) values, I uncovered significant insights into the contributory elements of claim likelihood, which were instrumental for understanding the interplay between historical data and future claim predictions.


**Packages**: `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `seaborn`


## Project Structure

### 1. Metadata Construction (`01-Metadata.ipynb`)
- **Focus**: Construction of metadata for data analysis.
- **Technologies/Methods**: Definition of roles, categories, levels (binary, nominal, ordinal, interval, ratio), data types (int, float, object, etc.), uniqueness, cardinality, missing values analysis, imputation strategies.
- **Key Highlights**: Application of specific rules for labeling based on column names and data types; addressing data imbalance.

### 2. Imbalanced Data Handling (`02_imbalanced.ipynb`)
- **Focus**: Techniques for managing imbalanced datasets.
- **Technologies/Methods**: Logistic regression for model evaluation, random undersampling and oversampling, discussion on information loss and overfitting.
- **Key Highlights**: Demonstrates the limitations of simple resampling methods in the context of imbalanced data.

### 3. Data Preprocessing(`03_data_preprocessing`)

#### 3.1. Handling Missing Values (`01_HandleMissingValues.ipynb`)
- **Focus**: Strategies and techniques for dealing with missing data.

#### 3.2. Anomaly Detection (`02_AnomalyDetection.ipynb`)
- **Focus**: Anomaly detection methodologies.
- **Technologies/Methods**: Z-score, IQR, Hampel, EllipticEnvelope, OneClassSVM, IsolationForests, LocalOutlierFactor.
- **Key Highlights**: Application to various data distributions; comparative analysis of methods.

#### 3.3. Exploratory Data Analysis (EDA) (`03_EDA.ipynb`)
- **Focus**: In-depth EDA to uncover insights from data.
- **Technologies/Methods**: Examination of nominal, binary, and ordinal features; correlation analysis.
- **Key Highlights**: Insights into data structure and feature importance.

### 4. Feature Engineering (`04_-Feature_Engineering.ipynb`)
- **Focus**: Identifying and handling missing values and outliers.
- **Technologies/Methods**: Z-score, IQR, Hampel, EllipticEnvelope, OneClassSVM, IsolationForests, LocalOutlierFactor.
- **Key Highlights**: Comparison of outlier detection methods; practical demonstrations on data.

### 5. Data Modeling (`05_Data_Modelling-.ipynb`)
- **Focus**: Data mining with an emphasis on feature analysis.
- **Technologies/Methods**: Analysis of nominal, binary, and ordinal data types; exploration of feature correlations.
- **Key Highlights**: Identification of less informative features and potential multicollinearity issues.



## Python Scripts

### 1. Data Management (`data_management.py`)
- Utility script for data loading, cleaning, and preprocessing tasks.

### 2. Metadata Utilities (`meta.py`)
- Provides functionality for constructing and managing metadata as described in `01-Metadata.ipynb`.



