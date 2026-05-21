# Save as `train_model_final_fixed.py`
import pandas as pd
from xgboost import XGBClassifier
import joblib
from pathlib import Path
import numpy as np

Path('models').mkdir(exist_ok=True)

def prepare_features(df):
    """Clean + mark categoricals for XGBoost"""
    print("Columns:", df.columns.tolist())
    
    # Clean numerics
    numeric_cols = ['female_petitioner', 'female_defendant', 'judge_position', 
                   'state_code', 'dist_code', 'court_no', 'year']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Convert strings to CATEGORICAL type (XGBoost native support)
    categorical_cols = ['state_name', 'dist_name', 'act_readable']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col].fillna('Unknown'))
            print(f"✅ Categorical: {col}")
    
    return df

# FEATURES - XGBoost handles categoricals natively
feature_sets = {
    'divorce': ['female_petitioner', 'female_defendant', 'judge_position', 'year', 
                'state_name', 'dist_name', 'court_no'],
    
    'custody': ['female_petitioner', 'judge_position', 'state_name', 'dist_name', 'court_no'],
    
    'maintenance': ['female_petitioner', 'female_defendant', 'dist_name', 'judge_position', 'state_name'],
    
    'other_family': ['female_petitioner', 'judge_position', 'state_name', 'dist_name']
}

categories = ['divorce', 'custody', 'maintenance', 'other_family']

for category in categories:
    print(f"\n🤖 TRAINING {category.upper()}")
    
    df = pd.read_csv(f'Verdictlens_output/dataset_decoded/{category}_cases_decoded.csv')
    df = prepare_features(df)
    
    # Target
    y = (df['disp_simple'] == 'granted').astype(int)
    
    # Available features
    features = feature_sets[category]
    available_features = [f for f in features if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    X = df[available_features]
    
    # XGBoost with CATEGORICAL SUPPORT
    model = XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        random_state=42,
        enable_categorical=True  # 🔥 FIX: Native categorical support
    )
    model.fit(X, y)
    
    # Feature importance (with readable names!)
    importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 TOP FEATURES ({category}):")
    print(importance.head())
    print(f"✅ Accuracy: {model.score(X, y):.3f}")
    
    # SAVE
    joblib.dump(model, f'models/{category}_categorical_model.pkl')
    joblib.dump(available_features, f'models/{category}_categorical_features.pkl')
    importance.to_csv(f'models/{category}_categorical_importance.csv', index=False)
    
    print(f"💾 models/{category}_categorical_model.pkl")

print("\n🎉 VERDICTLENS CATEGORICAL MODELS READY!")
print("✅ state_name='11 Maharashtra', dist_name='19: Balod'")
