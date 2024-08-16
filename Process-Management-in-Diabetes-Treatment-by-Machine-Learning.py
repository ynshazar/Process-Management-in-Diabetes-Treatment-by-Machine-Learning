import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFE, SelectFromModel, mutual_info_classif, chi2, f_classif
from sklearn.calibration import calibration_curve
from scipy.stats import kruskal
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Plot settings (resolution and font sizes)
plt.rcParams.update({
    'figure.dpi': 300,  # Increase DPI for higher resolution
    'font.size': 14,  # Increase general font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 16,  # X and Y axis label font size
    'xtick.labelsize': 15,  # X-axis tick label font size
    'ytick.labelsize': 15,  # Y-axis tick label font size
    'legend.fontsize': 15,  # Legend font size
    'figure.figsize': (12, 8),  # Increase figure size
    'lines.linewidth': 2  # Increase line width
})

# Load and preprocess data
data = pd.read_csv('data_son.csv')
print(f'Number of rows: {data.shape[0]}, Number of columns: {data.shape[1]}')

# Create target variable
data['target'] = data['Diabetes_Control_Status'].apply(lambda x: 1 if x == 3 else 0)

# Separate features and target variable
features = data.drop(columns=['idhash', 'Diabetes_Control_Status', 'target'])
target = data['target']

# Remove constant and low-variance features
features = features.loc[:, (features != features.iloc[0]).any()]  # Remove constant features
print(f'Constant features: {list(features.columns[(features == features.iloc[0]).all()])}')
features = features.loc[:, features.var() > 0.01]  # Remove low-variance features
print(f'Low-variance features: {list(features.columns[features.var() < 0.01])}')

# Standardization (only using MinMaxScaler for chi2)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

minmax_scaler = MinMaxScaler()
features_minmax_scaled = minmax_scaler.fit_transform(features)

# Feature selection methods
num_features = 20

# 1. SelectFromModel (using LGBMClassifier)
lgbm = LGBMClassifier(random_state=42, verbose=-1)
sfm = SelectFromModel(lgbm, max_features=num_features, threshold="mean")
sfm.fit(features_scaled, target)
sfm_features = sfm.get_support(indices=True)
sfm_feature_names = features.columns[sfm_features]

# 2. Mutual Information
mi = mutual_info_classif(features_scaled, target)
mi_features = pd.Series(mi, index=features.columns).nlargest(num_features).index

# 3. Recursive Feature Elimination (RFE, using LGBMClassifier)
rfe = RFE(estimator=lgbm, n_features_to_select=num_features)
rfe.fit(features_scaled, target)
rfe_features = rfe.get_support(indices=True)
rfe_feature_names = features.columns[rfe_features]

# 4. Chi-squared (using MinMaxScaler)
chi2_scores, _ = chi2(features_minmax_scaled, target)
chi2_features = pd.Series(chi2_scores, index=features.columns).nlargest(num_features).index

# 5. Analysis of Variance (ANOVA)
anova_scores, _ = f_classif(features_scaled, target)
anova_features = pd.Series(anova_scores, index=features.columns).nlargest(num_features).index

# 6. Kruskal-Wallis
kw_features = []
for col in features.columns:
    try:
        score, _ = kruskal(features[col], target)
        kw_features.append((col, score))
    except:
        pass
kw_features = sorted(kw_features, key=lambda x: x[1], reverse=True)[:num_features]
kw_feature_names = [f[0] for f in kw_features]

# 7. Model-based Feature Importance
# Model initialization
# noinspection PyTypeChecker
model1 = CatBoostClassifier(verbose=False)
model2 = XGBClassifier(verbose=0)
model3 = LGBMClassifier(verbose=-1)

# Model fitting
model1.fit(features_scaled, target)
model2.fit(features_scaled, target)
model3.fit(features_scaled, target)

# Feature importance
top_features_model1 = pd.Series(model1.feature_importances_, index=features.columns).nlargest(num_features).index
top_features_model2 = pd.Series(model2.feature_importances_, index=features.columns).nlargest(num_features).index
top_features_model3 = pd.Series(model3.feature_importances_, index=features.columns).nlargest(num_features).index

# Combine model-based selected features
model_selected_features = set(top_features_model1).union(top_features_model2).union(top_features_model3)

# Combine all selected features and keep unique ones
selected_features = set(sfm_feature_names).union(mi_features).union(rfe_feature_names).union(chi2_features).union(anova_features).union(kw_feature_names).union(model_selected_features)
selected_features = list(selected_features)

# Print selected features
print("Selected features:")
for i, feature in enumerate(selected_features, start=1):
    print(f"{i}. {feature}")

# Create new dataset with selected features
features_selected = features[selected_features]

# Balance the imbalanced dataset
sm = SMOTE(random_state=42)
features_resampled, target_resampled = sm.fit_resample(features_selected, target)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features_resampled, target_resampled, test_size=0.2, random_state=42)

# Standardize the selected features for model training
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define the first-level models
xgb = XGBClassifier(
    n_estimators=1000,
    min_child_weight=2,
    reg_lambda=0.7,
    reg_alpha=0.6,
    subsample=1,
    max_depth=6,
    learning_rate=0.1,
    gamma=0.15,
    colsample_bytree=0.8,
    random_state=42
)
lgbm = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    num_leaves=75,
    max_depth=15,
    min_child_samples=70,
    colsample_bytree=0.3,
    min_child_weight=0.1,
    subsample=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    verbose=-1
)
catboost = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.2,
    depth=4,
    l2_leaf_reg=9,
    loss_function='CrossEntropy',
    border_count=600,
    bagging_temperature=1.0,
    random_strength=3,
    one_hot_max_size=1,
    colsample_bylevel=1.0,
    subsample=0.7,
    verbose=False
)

# Train the first-level models
xgb.fit(X_train_scaled, y_train)
lgbm.fit(X_train_scaled, y_train)
catboost.fit(X_train_scaled, y_train)

# Get predictions from the first-level models on the validation set
xgb_preds = xgb.predict_proba(X_val_scaled)[:, 1]
lgbm_preds = lgbm.predict_proba(X_val_scaled)[:, 1]
catboost_preds = catboost.predict_proba(X_val_scaled)[:, 1]

# Combine predictions
stacked_predictions = pd.DataFrame({
    'xgb': xgb_preds,
    'lgbm': lgbm_preds,
    'catboost': catboost_preds
})

# Cross-validation on the meta-model (ExtraTreesClassifier)
meta_model = ExtraTreesClassifier(n_estimators=1000, max_depth=10)
meta_model.fit(stacked_predictions, y_val)

# Predictions on the test set
X_test = scaler.transform(features_resampled)

final_preds = meta_model.predict(pd.DataFrame({
    'xgb': xgb.predict_proba(X_test)[:, 1],
    'lgbm': lgbm.predict_proba(X_test)[:, 1],
    'catboost': catboost.predict_proba(X_test)[:, 1]
}))

# Results
print("Accuracy:", accuracy_score(target_resampled, final_preds))
conf_matrix = confusion_matrix(target_resampled, final_preds)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(target_resampled, final_preds, digits=4))

# Plot Confusion Matrix as a graph
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(xgb_preds, label='XGBoost', shade=True)
sns.kdeplot(lgbm_preds, label='LGBM', shade=True)
sns.kdeplot(catboost_preds, label='CatBoost', shade=True)
plt.title("Distribution of Model Predictions")
plt.legend()
plt.show()

# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, xgb_preds)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# LGBM ROC
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_val, lgbm_preds)
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

# CatBoost ROC
fpr_catboost, tpr_catboost, _ = roc_curve(y_val, catboost_preds)
roc_auc_catboost = auc(fpr_catboost, tpr_catboost)

# Get final predictions for the meta-model
meta_preds_proba = meta_model.predict_proba(stacked_predictions)[:, 1]
# Meta Model ROC
fpr_meta, tpr_meta, _ = roc_curve(y_val, meta_preds_proba)
roc_auc_meta = auc(fpr_meta, tpr_meta)

# ROC Curves Comparison
plt.figure(figsize=(10, 6))
plt.plot(fpr_xgb, tpr_xgb, color='blue', lw=2, label=f'XGBoost (area = {roc_auc_xgb:.2f})')
plt.plot(fpr_lgbm, tpr_lgbm, color='green', lw=2, label=f'LGBM (area = {roc_auc_lgbm:.2f})')
plt.plot(fpr_catboost, tpr_catboost, color='red', lw=2, label=f'CatBoost (area = {roc_auc_catboost:.2f})')
plt.plot(fpr_meta, tpr_meta, color='darkorange', lw=2, label=f'Meta Model (area = {roc_auc_meta:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall curve for the meta model
precision, recall, _ = precision_recall_curve(y_val, meta_preds_proba)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='purple', lw=2, label='Meta Model (ExtraTrees)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

feature_importances = pd.Series(meta_model.feature_importances_, index=stacked_predictions.columns)
feature_importances = feature_importances.sort_values(ascending=False)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar', color='teal')
plt.title('Feature Importances of Meta Model (ExtraTrees)')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.show()

train_sizes, train_scores, test_scores = learning_curve(meta_model, stacked_predictions, y_val, cv=5, scoring='accuracy')
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', label='Training Accuracy')
plt.plot(train_sizes, test_mean, color='green', marker='o', label='Validation Accuracy')
plt.title('Learning Curve for Meta Model (ExtraTrees)')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Calibration curve for the meta model
prob_true, prob_pred = calibration_curve(y_val, meta_preds_proba, n_bins=10)

plt.figure(figsize=(10, 6))
plt.plot(prob_pred, prob_true, marker='o', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Calibration Curve for Meta Model')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.show()

# Use ROC curve for Gain chart
fpr, tpr, thresholds = roc_curve(y_val, meta_preds_proba)
gains = tpr - fpr
plt.figure(figsize=(10, 6))
plt.plot(thresholds, gains, marker='o', color='green')
plt.title('Cumulative Gain Chart for Meta Model')
plt.xlabel('Threshold')
plt.ylabel('Gain')
plt.show()

lift = tpr / fpr
plt.figure(figsize=(10, 6))
plt.plot(thresholds, lift, marker='o', color='orange')
plt.title('Lift Chart for Meta Model')
plt.xlabel('Threshold')
plt.ylabel('Lift')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_val, meta_preds_proba, color='blue', edgecolor='k', s=50, alpha=0.7)
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red', linestyle='--')
plt.title('Prediction Error Plot')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.show()
