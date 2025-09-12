import os
import pandas as pd
import config


def get_serialized_filepaths():
    x = os.listdir('./files/serialized_asset_views')
    x = [f for f in x if f.endswith('.csv')]
    print("The following files were found:\n")
    for i,f in enumerate(x):
        print(f"{i+1}. {f}")
    i = input("Enter the number of the GLASSBEAM file: ")
    try:
        i = int(i)
        if i > 0 and i <= len(x):
            glassbeam_file = x[i-1]
            non_glassbeam_files = [f for i2,f in enumerate(x) if i2 != i-1]
            return glassbeam_file, non_glassbeam_files
        else:
            print("Invalid input")
    except:
        print("Invalid input")


def read_and_join_customers(non_glassbeam_files):
    df_list = []
    for f in non_glassbeam_files:
        try:
            df = pd.read_csv(f"./files/serialized_asset_views/{f}", dtype=str, na_filter=False)
        except:
            print(f"Error reading {f}")
        df_list.append(df)
    df = pd.concat(df_list)
    return df

def get_glassbeam_data(glassbeam_filepath):
    try:
        df = pd.read_csv(f"./files/serialized_asset_views/{glassbeam_filepath}", dtype=str, na_filter=False)    
    except:
        print(f"Error reading {glassbeam_filepath}")
        exit()
    return df

def get_modality_mel_id(non_glassbeam_df, glassbeam_df):
    full = pd.concat([non_glassbeam_df, glassbeam_df])
    modality_mel_id = config.remove_duplicates(full, ['make_target','model_name_target'], ['mel_id', 'modality_target'])[['make_target','model_name_target','modality_target','mel_id']]
    return modality_mel_id

def aggregate_df(df):
    df_gb = df.groupby(['company_name','make_target','model_name_target']).agg({'asset_sys_id': 'nunique'}).reset_index()
    df_gb = df_gb.rename(columns={'asset_sys_id': 'count'})
    df_gb = df_gb.pivot(index=['make_target','model_name_target'], columns='company_name', values='count').reset_index()
    return df_gb

def merge_gb_and_non_gb_dfs(non_glassbeam_df, glassbeam_df, customer_list):
    df = pd.merge(non_glassbeam_df, glassbeam_df, on=['make_target','model_name_target'], how='outer')
    for x in df.columns:
        if x in customer_list:   
            df[x] = df[x].fillna(0)
    return df




def final_sort(df, customer_list):
    df['count_temp'] = df[customer_list].sum(axis=1)
    customer_list = sorted(customer_list)
    df = df.sort_values(by=['make_target','verified_model_name','count_temp'], ascending=[True, False, False])
    df = df[['make_target','modality_target','model_name_target','verified_model_name','mel_id','GLASSBEAM_COVERAGE'] + customer_list]
    df = df.rename(columns={x: x.upper()  for x in customer_list})
    return df

if __name__ == "__main__":
    # get inputs from customers
    glassbeam_file, non_glassbeam_files = get_serialized_filepaths()
    non_glassbeam_df = read_and_join_customers(non_glassbeam_files)
    non_glassbeam_df.to_csv(config.get_customer_joined_filepath(), index=False)
    glassbeam_df = get_glassbeam_data(glassbeam_file)
    # Get modality and mel id (for later)
    modality_mel_id = get_modality_mel_id(non_glassbeam_df, glassbeam_df)
    # Mask model names
    glassbeam_df['model_name_target'] = glassbeam_df.apply(lambda row: row['model_name_target'] if row['model_name_target'].strip() != '' else row['model_name_source'], axis=1)
    non_glassbeam_df['model_name_target'] = non_glassbeam_df.apply(lambda row: row['model_name_target'] if row['model_name_target'].strip() != '' else row['model_name_source'], axis=1)
    # Glassbeam - rename cols
    glassbeam_df = glassbeam_df.rename(columns={'product_mgmt_stage': 'GLASSBEAM_COVERAGE'})
    # Glassbeam = dedup make / models
    glassbeam_df = config.remove_duplicates(glassbeam_df, ['make_target','model_name_target'], ['mel_id', 'modality_target'])[['make_target','model_name_target','GLASSBEAM_COVERAGE']]
    # Non-glassbeam - replace model_name_target with model_name_source if model_name_target is empty
    customer_list = list(non_glassbeam_df['company_name'].unique())
    # Non-glassbeam - aggregate
    non_glassbeam_df_agg = aggregate_df(non_glassbeam_df)
    #Merge GB and non-GB dfs
    df = merge_gb_and_non_gb_dfs(non_glassbeam_df_agg, glassbeam_df, customer_list)
    df = pd.merge(df, modality_mel_id, on=['make_target','model_name_target'], how='left')
    df['mel_id'] = df['mel_id'].fillna('')
    # Add verified model name
    df['verified_model_name'] = df['mel_id'].apply(lambda x: 0 if x.strip() == '' else 1)
    # Postprocess
    df = final_sort(df, customer_list)
    df.to_csv(config.get_customer_summary_filepath(), index=False)




"""
def dedup_dfs(glassbeam_df, non_glassbeam_df):
    non_glassbeam_df_deduped = config.remove_duplicates(non_glassbeam_df, ['company_name','make_target','model_name_target'], ['mel_id', 'modality_target'])
    non_glassbeam_df_deduped = non_glassbeam_df_deduped[['asset_sys_id','company_name','make_target','model_name_target']]
    glassbeam_df_deduped = config.remove_duplicates(glassbeam_df, ['make_target','model_name_target'], ['mel_id', 'modality_target'])
    glassbeam_df_deduped = glassbeam_df_deduped[['make_target','model_name_target', 'product_mgmt_stage']]
    # get modality and mel id
    full = pd.concat([non_glassbeam_df, glassbeam_df])
    modality_mel_id = config.remove_duplicates(full, ['make_target','model_name_target'], ['mel_id', 'modality_target'])[['make_target','model_name_target','modality_target','mel_id']]
    return glassbeam_df_deduped, non_glassbeam_df_deduped, modality_mel_id
"""


"""
def split_glassbeam_row(row):
    for x in ['full', 'partial', 'codev']:
        if row[x].strip().upper() == 'Y':
            return f'GLASSBEAM_{x.strip()}'.upper()            
    for x in ['full', 'partial', 'codev']:
        y = row[x].strip().upper() #e.g., "Assess" for co-dev or "Not in public scope" for full
        if row[x].strip().upper() != '':
            return f'GLASSBEAM_{x}_{y}'.upper()"""

