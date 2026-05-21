import pandas as pd
import numpy as np
from pathlib import Path
import os

input_folder = 'Verdictlens_output/dataset'
output_folder = 'Verdictlens_output/dataset_decoded'
Path(output_folder).mkdir(exist_ok=True)

categories = ['divorce', 'custody', 'maintenance', 'other_family']
print("🔗 LOADING STATE/DISTRICT KEYS + DECODING...")

# Load ALL lookup keys
act_key = pd.read_csv('justice_data/csv/keys/keys/act_key.csv')
state_key = pd.read_csv('justice_data/csv/keys/keys/cases_state_key.csv')
dist_key = pd.read_csv('justice_data/csv/keys/keys/cases_district_key.csv')

# Convert to strings
act_key['act_s'] = act_key['act_s'].astype(str).fillna('Unknown')
state_key['state_name'] = state_key['state_code'].astype(str) + ': ' + state_key.get('state_name', 'Unknown').astype(str)
dist_key['dist_name'] = dist_key['dist_code'].astype(str) + ': ' + dist_key.get('district_name', 'Unknown').astype(str)

print("✅ Keys loaded:")
print(f"States: {len(state_key)} | Districts: {len(dist_key)}")

for cat in categories:
    input_file = f'{input_folder}/{cat}_cases.csv'
    if os.path.exists(input_file):
        print(f"\n📊 Decoding {cat}_cases.csv...")
        df = pd.read_csv(input_file)
        
        # 1. STATE CODE → STATE NAME
        df['state_name'] = 'Unknown State'
        for idx, row in df.iterrows():
            state_code = str(int(row.get('state_code', 0)))
            state_match = state_key[state_key['state_code'].astype(str) == state_code]
            if not state_match.empty:
                df.at[idx, 'state_name'] = state_match['state_name'].iloc[0]
        
        # 2. DISTRICT CODE → DISTRICT NAME  
        df['dist_name'] = 'Unknown District'
        for idx, row in df.iterrows():
            dist_code = str(int(row.get('dist_code', 0)))
            dist_match = dist_key[dist_key['dist_code'].astype(str) == dist_code]
            if not dist_match.empty:
                df.at[idx, 'dist_name'] = dist_match['dist_name'].iloc[0]
        
        # 3. EXISTING DECODING
        df['category'] = cat
        df['act_readable'] = 'Unknown Act'
        df['disp_readable'] = df['disp_simple'].map({
            'granted': 'Decree Granted', 'not_granted': 'Dismissed/Rejected',
            'dismissed': 'Dismissed', 'withdrawn': 'Withdrawn'
        }).fillna('Disposed')
        
        # ML + UI columns
        keep_cols = ['ddl_case_id', 'year', 'state_code', 'state_name', 
                    'dist_code', 'dist_name', 'court_no', 'category', 
                    'disp_simple', 'disp_readable', 'act_readable',
                    'female_petitioner', 'female_defendant', 'judge_position']
        
        df_final = df[[col for col in keep_cols if col in df.columns]].fillna('Unknown')
        
        # SAVE
        output_file = f'{output_folder}/{cat}_cases_decoded.csv'
        df_final.to_csv(output_file, index=False)
        
        print(f"💾 {output_file}: {len(df_final)} rows ✓")
        print("Sample with STATE/DISTRICT:")
        print(df_final[['category', 'state_name', 'dist_name', 'disp_readable']].head())
        
    else:
        print(f"❌ {input_file} not found")

print(f"\n🎉 STATE/DISTRICT NAMES ADDED!")
print("📁 New columns: state_name, dist_name")
