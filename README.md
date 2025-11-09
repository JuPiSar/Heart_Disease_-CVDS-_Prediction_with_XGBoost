# Heart Disease (CVDS) Prediction with XGBoost

This notebook demonstrates a machine learning workflow for predicting heart disease using the XGBoost classifier. The project covers data loading, preprocessing (one-hot encoding), model training, and making predictions.

## Dataset

The dataset used for this prediction task is the "Heart Failure Prediction" dataset, sourced from Kaggle. It contains various health metrics and patient attributes which are used to predict the likelihood of heart disease.

Source: [https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## Project Steps

1.  **Data Loading**: The `heart.csv` dataset is loaded into a pandas DataFrame.
2.  **One-hot Encoding**: Categorical features such as 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', and 'ST_Slope' are converted into numerical format using one-hot encoding with `pd.get_dummies`.
3.  **Feature Selection**: The 'HeartDisease' column is identified as the target variable, and all other columns are selected as features for the model.
4.  **Data Splitting**: The dataset is split into training and validation sets to evaluate the model's performance. An 80/20 split is used for training and validation, respectively, with an additional split of the training data into `fit` and `eval` sets for XGBoost's early stopping mechanism.
5.  **Model Training**: An XGBoost Classifier (`XGBClassifier`) is trained on the preprocessed data. Early stopping is implemented to prevent overfitting, monitoring the `logloss` on the evaluation set.
6.  **Model Evaluation**: The accuracy of the trained model is evaluated on both the training and validation datasets.
7.  **Prediction**: The notebook demonstrates how to use the trained model to make predictions for individual or multiple patients.

## Key Libraries Used

*   `numpy`
*   `pandas`
*   `sklearn` (for `train_test_split`, `accuracy_score`)
*   `xgboost` (for `XGBClassifier`, `EarlyStopping`)
*   `matplotlib.pyplot`
