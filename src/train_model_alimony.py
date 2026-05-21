import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

OUTPUT_FOLDER = Path('verdictlens_output')
MODELS_FOLDER = OUTPUT_FOLDER / 'models'
MODELS_FOLDER.mkdir(exist_ok=True)

# ==================== LOAD DATA ====================

print("\n" + "="*70)
print("📂 LOADING DATA")
print("="*70 + "\n")

df = pd.read_csv(OUTPUT_FOLDER / 'ALIMONY_DATASET_1500_FINAL.csv')

print(f"✓ Loaded {len(df):,} cases")

# Check features
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# ==================== DATA PREPARATION ====================

print("\n" + "="*70)
print("🔧 DATA PREPARATION")
print("="*70 + "\n")

# Select features
feature_cols = [
    'husband_income_monthly',
    'wife_income_monthly', 
    'marriage_duration_years',
    'number_of_children'
]

# Check feature availability
print("Checking features:")
for col in feature_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  ✓ {col}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)")
    else:
        print(f"  ✗ {col}: NOT FOUND")

# Add wife employment as binary
if 'wife_employment' in df.columns:
    df['wife_employed'] = (df['wife_employment'] == 'Employed').astype(int)
    feature_cols.append('wife_employed')
    print(f"  ✓ wife_employed: Created (binary)")

# Target variable
target_col = 'alimony_amount_monthly'

# Remove rows with missing target
df_clean = df[df[target_col].notna()].copy()
print(f"\n✓ Clean dataset: {len(df_clean):,} cases (removed {len(df) - len(df_clean)} with missing target)")

# Remove rows with missing features
for col in feature_cols:
    before = len(df_clean)
    df_clean = df_clean[df_clean[col].notna()]
    removed = before - len(df_clean)
    if removed > 0:
        print(f"  Removed {removed} cases with missing {col}")

print(f"\n✓ Final dataset: {len(df_clean):,} cases with complete features")

# ==================== FEATURE MATRIX ====================

print("\n" + "="*70)
print("📊 BUILDING FEATURE MATRIX")
print("="*70 + "\n")

X = df_clean[feature_cols].values
y = df_clean[target_col].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

print(f"\nFeatures used:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

print(f"\nTarget statistics:")
print(f"  Mean: ₹{y.mean():,.0f}/month")
print(f"  Median: ₹{np.median(y):,.0f}/month")
print(f"  Std Dev: ₹{y.std():,.0f}")
print(f"  Min: ₹{y.min():,.0f}")
print(f"  Max: ₹{y.max():,.0f}")

# ==================== TRAIN/TEST SPLIT ====================

print("\n" + "="*70)
print("✂️  TRAIN/TEST SPLIT")
print("="*70 + "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train):,} cases ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test):,} cases ({len(X_test)/len(X)*100:.1f}%)")

# ==================== MODEL TRAINING ====================

print("Model hyperparameters:")
print("  • n_estimators: 300  ← INCREASED")  # Change this line
print("  • max_depth: 6")
print("  • learning_rate: 0.05")
print("  • min_child_weight: 3")
print("  • subsample: 0.8")
print("  • colsample_bytree: 0.8")
print("  • early_stopping_rounds: 50  ← NEW: Stops if no improvement")

# Create model with 300 epochs
model = xgb.XGBRegressor(
    n_estimators=300,  
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror',
    early_stopping_rounds=50,
    eval_metric='rmse'
)

print("\n🔨 Training model with 300 epochs...")  
print("   (Early stopping enabled - may stop before 300 if converged)")  


# Train with validation set for early stopping
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

model.fit(
    X_train_split, y_train_split,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"✓ Training complete! Stopped at epoch {model.best_iteration + 1}")

# ==================== MODEL EVALUATION ====================

print("\n" + "="*70)
print("📊 MODEL EVALUATION")
print("="*70 + "\n")

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Training metrics
print("📈 TRAINING SET PERFORMANCE:")
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)
mape_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100

print(f"  MAE:  ₹{mae_train:,.0f}/month")
print(f"  RMSE: ₹{rmse_train:,.0f}/month")
print(f"  R² Score: {r2_train:.4f}")
print(f"  MAPE: {mape_train:.2f}%")

# Test metrics
print("\n📈 TEST SET PERFORMANCE:")
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

print(f"  MAE:  ₹{mae_test:,.0f}/month")
print(f"  RMSE: ₹{rmse_test:,.0f}/month")
print(f"  R² Score: {r2_test:.4f}")
print(f"  MAPE: {mape_test:.2f}%")

# Percentage accuracy
within_10pct = (np.abs(y_test - y_pred_test) / y_test <= 0.10).sum() / len(y_test) * 100
within_20pct = (np.abs(y_test - y_pred_test) / y_test <= 0.20).sum() / len(y_test) * 100
within_30pct = (np.abs(y_test - y_pred_test) / y_test <= 0.30).sum() / len(y_test) * 100

print(f"\n📊 PREDICTION ACCURACY:")
print(f"  Within ±10%: {within_10pct:.1f}%")
print(f"  Within ±20%: {within_20pct:.1f}%")
print(f"  Within ±30%: {within_30pct:.1f}%")

# ==================== CROSS-VALIDATION ====================

# ==================== CROSS-VALIDATION ====================

print("\n" + "="*70)
print("🔄 CROSS-VALIDATION (5-Fold)")
print("="*70 + "\n")

print("Running 5-fold cross-validation...")

# Create a new model WITHOUT early stopping for CV
cv_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
    # NO early_stopping_rounds here
)

cv_scores = cross_val_score(
    cv_model, X, y,  # Use cv_model instead of model
    cv=5, 
    scoring='r2',
    n_jobs=-1
)

print(f"\nCross-validation R² scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\n  Mean CV R²: {cv_scores.mean():.4f}")
print(f"  Std CV R²:  {cv_scores.std():.4f}")


# ==================== FEATURE IMPORTANCE ====================

print("\n" + "="*70)
print("📈 FEATURE IMPORTANCE")
print("="*70 + "\n")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature importance ranking:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f} ({row['importance']*100:.1f}%)")

# ==================== SAMPLE PREDICTIONS ====================

print("\n" + "="*70)
print("🔮 SAMPLE PREDICTIONS")
print("="*70 + "\n")

# Show 5 random test cases
sample_indices = np.random.choice(len(X_test), size=min(5, len(X_test)), replace=False)

print("Sample predictions from test set:\n")
for i, idx in enumerate(sample_indices, 1):
    actual = y_test[idx]
    predicted = y_pred_test[idx]
    error = actual - predicted
    error_pct = abs(error) / actual * 100
    
    print(f"Case {i}:")
    print(f"  Actual:    ₹{actual:,.0f}/month")
    print(f"  Predicted: ₹{predicted:,.0f}/month")
    print(f"  Error:     ₹{error:,.0f} ({error_pct:.1f}%)")
    
    # Show features
    print(f"  Features:")
    for j, feat_name in enumerate(feature_cols):
        print(f"    {feat_name}: {X_test[idx][j]}")
    print()

# ==================== VISUALIZATIONS ====================

print("="*70)
print("📊 CREATING VISUALIZATIONS")
print("="*70 + "\n")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predicted vs Actual
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred_test, alpha=0.5, s=30)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Alimony (₹/month)', fontsize=12)
ax1.set_ylabel('Predicted Alimony (₹/month)', fontsize=12)
ax1.set_title(f'Predicted vs Actual\nR² = {r2_test:.4f}, MAE = ₹{mae_test:,.0f}', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Residuals
ax2 = axes[0, 1]
residuals = y_test - y_pred_test
ax2.scatter(y_pred_test, residuals, alpha=0.5, s=30)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Alimony (₹/month)', fontsize=12)
ax2.set_ylabel('Residuals (₹)', fontsize=12)
ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Feature Importance
ax3 = axes[1, 0]
feature_importance_sorted = feature_importance.sort_values('importance')
ax3.barh(feature_importance_sorted['feature'], feature_importance_sorted['importance'], color='steelblue')
ax3.set_xlabel('Importance', fontsize=12)
ax3.set_title('Feature Importance', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 4. Error Distribution
ax4 = axes[1, 1]
error_pct = np.abs(y_test - y_pred_test) / y_test * 100
ax4.hist(error_pct, bins=30, color='coral', alpha=0.7, edgecolor='black')
ax4.axvline(error_pct.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {error_pct.mean():.1f}%')
ax4.axvline(np.median(error_pct), color='blue', linestyle='--', lw=2, label=f'Median: {np.median(error_pct):.1f}%')
ax4.set_xlabel('Absolute Error (%)', fontsize=12)
ax4.set_ylabel('Frequency', fontsize=12)
ax4.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(MODELS_FOLDER / 'alimony_model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: alimony_model_evaluation.png")

plt.close()

# ==================== SAVE MODEL ====================

print("\n" + "="*70)
print("💾 SAVING MODEL")
print("="*70 + "\n")

model_file = MODELS_FOLDER / 'alimony_calculator_model.json'
model.save_model(model_file)
print(f"✓ Saved model: {model_file.name}")

# Save feature names
feature_info = {
    'features': feature_cols,
    'feature_importance': feature_importance.to_dict('records'),
    'metrics': {
        'mae_test': float(mae_test),
        'rmse_test': float(rmse_test),
        'r2_test': float(r2_test),
        'mape_test': float(mape_test),
        'within_20pct': float(within_20pct)
    },
    'epochs_trained': int(model.best_iteration + 1)
}

import json
with open(MODELS_FOLDER / 'model_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f"✓ Saved feature info: model_info.json")

print("\n" + "="*70)
print("🎯 MODEL ACCURACY SUMMARY")
print("="*70 + "\n")

# Calculate accuracy at different thresholds
thresholds = [5, 10, 15, 20, 25, 30, 40, 50]
print("Prediction Accuracy at Different Error Thresholds:\n")

for threshold in thresholds:
    accuracy = (np.abs(y_test - y_pred_test) / y_test <= threshold/100).sum() / len(y_test) * 100
    stars = "★" * int(accuracy / 10)
    print(f"  Within ±{threshold:2d}%: {accuracy:5.1f}% {stars}")

print(f"\n{'='*70}")
print(f"📈 OVERALL MODEL ACCURACY")
print(f"{'='*70}\n")

# Primary accuracy metric (within ±20%)
primary_accuracy = within_20pct
print(f"  🎯 PRIMARY ACCURACY: {primary_accuracy:.1f}%")
print(f"     (Predictions within ±20% of actual amount)\n")

# Secondary accuracy metric (within ±30%)
secondary_accuracy = within_30pct
print(f"  ✅ SECONDARY ACCURACY: {secondary_accuracy:.1f}%")
print(f"     (Predictions within ±30% of actual amount)\n")

# R² as percentage
r2_percentage = r2_test * 100
print(f"  📊 R² SCORE: {r2_test:.4f} ({r2_percentage:.1f}%)")
print(f"     (Variance explained by the model)\n")

# Overall grade
if primary_accuracy >= 70:
    grade = "A (Excellent)"
    emoji = "🌟"
elif primary_accuracy >= 60:
    grade = "B (Good)"
    emoji = "✅"
elif primary_accuracy >= 50:
    grade = "C (Acceptable)"
    emoji = "👍"
else:
    grade = "D (Needs Improvement)"
    emoji = "⚠️"

print(f"  {emoji} OVERALL GRADE: {grade}\n")

print(f"{'='*70}\n")

# ==================== FINAL SUMMARY ====================

print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETE!")
print("="*70 + "\n")

print(f"📊 FINAL MODEL PERFORMANCE:\n")
print(f"  Training Info:")
print(f"    • Epochs trained:  {model.best_iteration + 1}/500")
print(f"    • Early stopped:   {'Yes' if model.best_iteration < 499 else 'No'}")
print(f"\n  Test Set Metrics:")
print(f"    • R² Score:        {r2_test:.4f}")
print(f"    • MAE:             ₹{mae_test:,.0f}/month")
print(f"    • RMSE:            ₹{rmse_test:,.0f}/month")
print(f"    • MAPE:            {mape_test:.1f}%")
print(f"    • Within ±20%:     {within_20pct:.1f}%")
print(f"    • Within ±30%:     {within_30pct:.1f}%")

print(f"\n  Cross-Validation:")
print(f"    • Mean CV R²:      {cv_scores.mean():.4f}")
print(f"    • Std CV R²:       {cv_scores.std():.4f}")

print(f"\n📁 Saved Files:")
print(f"    • {model_file.name}")
print(f"    • model_info.json")
print(f"    • alimony_model_evaluation.png")

print(f"\n🎯 Model Quality Assessment:")
if r2_test >= 0.60:
    quality = "EXCELLENT"
    emoji = "🌟"
elif r2_test >= 0.50:
    quality = "GOOD"
    emoji = "✅"
elif r2_test >= 0.40:
    quality = "ACCEPTABLE"
    emoji = "👍"
else:
    quality = "NEEDS IMPROVEMENT"
    emoji = "⚠️"

print(f"    {emoji} {quality} (R² = {r2_test:.4f})")

print(f"\n💡 Interpretation:")
print(f"    • Model explains {r2_test*100:.1f}% of variance in alimony amounts")
print(f"    • Average prediction error: ₹{mae_test:,.0f}/month")
print(f"    • {within_20pct:.1f}% of predictions within ±20% of actual")

print(f"\n{'='*70}")
print(f"🎉 Ready to build the alimony calculator!")
print(f"{'='*70}\n")
