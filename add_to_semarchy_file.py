import pandas as pd
import os

# Path to the serialized asset views folder
base_path = 'files/serialized_asset_views'

# List of xlsx files to concatenate
xlsx_files = [
    'uva_tim_todd.xlsx',
    'marshfield_tim_todd.xlsx',
    'methodist_tim_todd.xlsx',
    'utsw_tim_todd.xlsx',
    'tx_childrens_jason_behm.xlsx',
    'metro_health_jason_behm.xlsx',
    'rochester_tim_todd.xlsx',
    'kettering_tim_todd.xlsx'
]

# Read the main CSV file
print("Reading SERIALIZED_ASSET_VIEW_ALL_CUSTOMERS.csv...")
main_csv_path = os.path.join(base_path, 'SERIALIZED_ASSET_VIEW_ALL_CUSTOMERS.csv')
df_main = pd.read_csv(main_csv_path)
original_row_count = len(df_main)
print(f"Loaded {original_row_count:,} rows from SERIALIZED_ASSET_VIEW_ALL_CUSTOMERS.csv\n")

# List to store all dataframes
all_dfs = [df_main]
total_files = len(xlsx_files)

# Read each xlsx file and append to list
for i, filename in enumerate(xlsx_files, start=1):
    file_path = os.path.join(base_path, filename)
    print(f"Processing {filename}... ({i} of {total_files})")
    
    if os.path.exists(file_path):
        df_temp = pd.read_excel(file_path)
        print(f"  -> Loaded {len(df_temp):,} rows")
        all_dfs.append(df_temp)
    else:
        print(f"  -> WARNING: File not found, skipping")

# Concatenate all dataframes
print("\nConcatenating all dataframes...")
df_combined = pd.concat(all_dfs, ignore_index=True)
final_row_count = len(df_combined)

# Save the combined dataframe
output_path = os.path.join(base_path, 'SERIALIZED_ASSET_VIEW_ALL_CUSTOMERS_new.csv')
print(f"Saving to {output_path}...")
df_combined.to_csv(output_path, index=False)

# Print summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Original rows in SERIALIZED_ASSET_VIEW_ALL_CUSTOMERS.csv: {original_row_count:,}")
print(f"Final rows after concatenation: {final_row_count:,}")
print(f"Rows added: {final_row_count - original_row_count:,}")
print(f"\nSaved to: {output_path}")
