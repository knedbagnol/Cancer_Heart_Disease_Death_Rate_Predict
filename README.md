## Problem Statement
How accurately can cancer and heart disease death rates by state be predicted? What are the top drivers given these features: median household income, access to stores/food, access to fitness centers, government assistance, food insecurity, taxes, obesity, diabetes, number/quantity of local farmers, poverty rates, air quality, water quality (lead/copper rates), smoking, industrial energy consumption, happiness, sunshine and temperature)?  The death rates for cancer and heart disease in the United States are shown below.

## Data Sets
Data was collected for Cancer and Heart Disease.  This included education level, air quality, lead, general health data such as obesity and diabetes, lifestyle data such as physical activity, smoking and religious beliefs.  As we modeled we continually searched for and added more data (health insurance rates and spending, socioeconomic data and energy consumption).  The data sets used are listed below. Approximately 600 variables were used in the evaluation.  The target variables are cancer death rate and heart disease death rate. The data_master.csv file contains all of the data we eventually used in our final notebooks. The datasets folder contain each individual .csv file used.

| **Data** | **Source** |
| --- | --- |
| Food Environment Atlas | [USDA ERS - Data Access and Documentation Downloads](https://www.ers.usda.gov/data-products/food-environment-atlas/data-access-and-documentation-downloads/) |
| Food Access Research Atlas | [USDA ERS - Download the Data](https://www.ers.usda.gov/data-products/food-access-research-atlas/download-the-data/) |
| Obesity | [https://www.cdc.gov/obesity/data/prevalence-maps.html](https://www.cdc.gov/obesity/data/prevalence-maps.html) |
| Cancer | [Cancer Rates By State 2021 (worldpopulationreview.com)](https://worldpopulationreview.com/state-rankings/cancer-rates-by-state) |
| Cancer | [Stats of the States - Cancer Mortality (cdc.gov)](https://www.cdc.gov/nchs/pressroom/sosmap/cancer_mortality/cancer.htm) |
| Heart Disease | [Stats of the States - Heart Disease Mortality (cdc.gov)](https://www.cdc.gov/nchs/pressroom/sosmap/heart_disease_mortality/heart_disease.htm#:~:text=Heart%20Disease%20Mortality%20by%20State%20%20%20,%20%207%2C575%20%2046%20more%20rows%20) |
| Smoking | [(https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm)](https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm) |
| Happiness | [https://worldpopulationreview.com/state-rankings/happiest-states](https://worldpopulationreview.com/state-rankings/happiest-states) |
| Air Quality | [https://www.epa.gov/outdoor-air-quality-data](https://www.epa.gov/outdoor-air-quality-data) |
| Water Quality | [https://www.waterqualitydata.us/portal/](https://www.waterqualitydata.us/portal/) |
| Industrial Energy Consumption | [https://data.nrel.gov/submissions/50](https://data.nrel.gov/submissions/50) |
| Health Care Coverage | [(https://www.kff.org/other/state-indicator/health-insurance-coverage-of-the-total-population-cps/?currentTimeframe=0&amp;sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D)](https://www.kff.org/other/state-indicator/health-insurance-coverage-of-the-total-population-cps/?currentTimeframe=0&amp;sortModel=%7B%22colId%22:%22Location%22,%22sort%22:%22asc%22%7D) |
| State Health Data | [(https://www.americashealthrankings.org/explore/annual/measure/fvcombo/state/ALL)](https://www.americashealthrankings.org/explore/annual/measure/fvcombo/state/ALL) |
| Socioecological Data | [https://data.world/kevinnayar/us-states-sociological-metrics](https://data.world/kevinnayar/us-states-sociological-metrics) |
| Healthcare Spending | [https://data.world/johnsnowlabs/us-healthcare-spending-by-state](https://data.world/johnsnowlabs/us-healthcare-spending-by-state) |
| Fast Food Data | [(https://www.nicerx.com/fast-food-capitals/)](https://www.nicerx.com/fast-food-capitals/) |

## Packages Used
The following packages were used in this analysis.

- import pandas as pd
- import seaborn as sns
- import matplotlib.pyplot as plt
- import math
- import numpy as np
- import plotly.express as px
- import plotly.graph_objects as go
- import plotly.offline as pyo
- import joblib
- import pickle
- import glob
- import statsmodels.api as sm

- from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
- from sklearn.pipeline import Pipeline
- from sklearn.neighbors import KNeighborsRegressor
- from sklearn.tree import DecisionTreeRegressor
- from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
- from sklearn.linear_model import LinearRegression, RidgeCV
- from sklearn import metrics
- from sklearn.svm import SVR
- from sklearn.preprocessing import StandardScaler, PolynomialFeatures
- from sklearn.metrics import r2_score, mean_squared_error
- from sklearn.decomposition import PCA

## Data Dictionary
#### These are the final features in our models
| **variable name** | **type** | **units** | **description** |
| --- | --- | --- | --- |
| 2005\_90th Percentile AQI | float64 | micrograms per cubic meter | The value from air quality index for which 90 percent of the rest of the measured values for the year are equal to or less than, 2005 |
| 2016\_Days Ozone | float64 | days | Days with higher ozone levels, 2016 |
| 2018\_Days Ozone | float64 | days | Days with higher ozone levels, 2018 |
| 2019\_Days Ozone | float64 | days | Days with higher ozone levels, 2019 |
| 2020\_Days PM2.5 | float64 | days | Days where fine inhalable particles with diameters that are generally 2.5 micrometer and smaller, 2020 |
| 2020\_drive\_alone\_to\_work\_score | float64 | weighted Z-score | weighted z score of adults that drive alone to work |
| 2020\_nutrition\_physical\_activity\_score | float64 | weighted Z-score | sum of weighted z-score of all ranked nutrition and physical activity scores |
| FSRPTH16 | object | # per 1,000 population | Fast-food restaurants/1,000 pop, 2016 |
| PCT\_DIABETES\_ADULTS08 | float64 | percent | Percent of adults with diabetes in the US, 2008 |
| PCT\_DIABETES\_ADULTS13 | float64 | percent | Percent of adults with diabetes in the US, 2013 |
| PCT\_OBESE\_ADULTS12 | float64 | percent | Percent obese adults in the US, 2012 |
| PCT\_OBESE\_ADULTS14 | float64 | percent | Percent obese adults in the US, 2014 |
| PCT\_OBESE\_ADULTS17 | float64 | percent | Percent obese adults in the US, 2017 |
| PCT\_OBESE\_ADULTS20 | float64 | percent | Percent obese adults in the US, 2020 |
| PCT\_WICWOMEN14 | float64 | percent | WIC infant and children participants (% infant &amp; children), 2014 |
| PCT\_WICWOMEN16 | float64 | percent | WIC infant and children participants (% infant &amp; children), 2016 |
| Percent Above Poverty Rate | float64 | percent | Percent population above poverty rate, 2020 |
| Percent Educational Attainment | float64 | percent | Percent population with college degree 2020 |
| Percent Non-religious | float64 | percent | Percent population that are not religious, 2020 |
| totalScore | float64 | weighted Z-score | sum of weighted z-score of all happiness scores, 2020 |
| WIC\_PART\_2015 | float64 | percent | WIC participants in the US, 2015 |
| WIC\_PART\_2016 | float64 | percent | WIC participants in the US, 2016 |
| 2020\_high\_cholesterol\_score | float64 | weighted Z-score | weighted z score of adults with high cholesterol |
| 2020\_frequent\_mental\_distress\_score | float64 | weighted Z-score | weighted z score of adults with frequent mental distress |
| 2020\_exercise\_score | float64 | weighted Z-score | weighted z score of adults that exercise |
| 2020\_health\_status\_score | float64 | weighted Z-score | weighted z score of adults health status |
| 2020\_high\_blood\_pressure\_score | float64 | weighted Z-score | weighted z score of adults blood pressure |
| 2015\_2010\_PC\_LACCESS\_SENIORS | float64 | percent | Seniors, low access to store (%), 2010 to 2015 |

## Exploratory Data Analysis
Extensive exploratory data analysis was performed.  Our first step was trying to understand the data we downloaded. Most of it was fairly straightforward, however some datasets required looking through an extensive data dictionary to fully make sense of what features we were getting. 

Fortunately, there was not a large amount of data cleaning that we had to implement. We had very few null values, and almost no categorical features since we were looking at rates and  percentages mostly. We did modify some datasets to be a percent change instead of a rate as we saw that our model sometimes would have better performance once integrating a percent change feature.
 
Once we got our modeling pipeline set up, it was just a matter of finding datasets we thought might be useful, and plugging it into the machine. 
The figures below show the relationships of the highest correlating variables with cancer mortality rate and heart disease mortality rate.

## Procedure Methodology for Refinement of Model
The mortality rates for 2015 through 2018 were not used to predict the 2019 mortality rates.  When the rates were used, the accuracy was 99% and none of the other variables influenced the rate.  Therefore, the final models did not use the historical mortality rates.

The highest correlating features were used to engineer features using polynomial features and/or principal component analysis.  Several different models were instantiated to find the best scores including Linear Regression, LassCV, Ridge CV, Random Forest Regressor, Bagging Regressor and Gradient Boosting Regressor.  The best performing models for cancer and heart disease were used to make predictions.

## Conclusions/Findings
Predicting cancer was easier than predicting heart disease.  The best performing cancer model was a LassoCV with a training score of 0.72 and a testing score of 0.75.  The highest correlating features were percent obesity and diabetes as well as the happiness score and nutrition and physical activity score.

The best performing model for heart disease was Linear Regression with principal component analysis (PCA) with a training score of 0.78 and testing score of 0.46. The highest correlating features were percent cholesterol, obesity, diabetes, frequent mental distress score, nutrition and physical activity score and WIC participation rates. 

It was surprising that the air quality index or lead was not more of a predictor. It was also somewhat surprising that happiness and mental health distress had fairly high correlations with cancer and heart disease, and both of those also correlate fairly highly with poverty rates and negatively correlate with education attainment.

## Recommendations
Obesity, diabetes and nutrition/physical activity score were highly correlated for both cancer and heart disease.  Maintaining a healthy diet and avoiding excessive sugar is recommended.  In addition exercise, on a regular basis to help maintain a healthy weight is recommended.  People should also know their family history for heart disease.

To improve upon our models in the future, more data needs to be collected, cleaned, and added. Some data to potentially look at would be genetics, blood samples with levels of carcinogens, and top employment industries for each state.


