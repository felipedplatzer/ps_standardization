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
            non_glassbeam_files = [f for i,f in enumerate(x) if i != i-1]
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


def split_glassbeam_row(row):
    for x in ['full', 'partial', 'codev']:
        if row[x].strip().upper() == 'Y':
            return f'GLASSBEAM_{x.strip()}'.upper()            
    for x in ['full', 'partial', 'codev']:
        y = row[x].strip().upper() #e.g., "Assess" for co-dev or "Not in public scope" for full
        if row[x].strip().upper() != '':
            return f'GLASSBEAM_{x}_{y}'.upper()




def aggregate_df(df, is_glassbeam):
    df['model_name_target'] = df.apply(lambda row: row['model_name_target'] if row['model_name_target'].strip() != '' else row['model_name_source'], axis=1)
    df['verified_model_name'] = df['mel_id'].apply(lambda x: 0 if x.strip() == '' else 1)
    if is_glassbeam:
        df = df.groupby(['company_name','make_target','modality_target','model_name_target','mel_id']).agg(count=('verified_model_name', 'size'), verified_model_name=('verified_model_name', 'first')).reset_index()
        df['count'] = df['count'].apply(lambda x: 1 if x > 0 else 0)
    else:
        df = df.groupby(['company_name','make_target','modality_target','model_name_target','mel_id']).agg({'asset_sys_id': 'nunique','verified_model_name': 'first'}).reset_index()
        df = df.rename(columns={'asset_sys_id': 'count'})
    df = df.pivot(index=['make_target','modality_target','model_name_target', 'verified_model_name','mel_id'], columns='company_name', values='count').reset_index()
    return df

def merge_gb_and_non_gb_dfs(non_glassbeam_df, glassbeam_df):
    df = pd.merge(non_glassbeam_df, glassbeam_df, on=['make_target','modality_target','model_name_target', 'verified_model_name','mel_id'], how='outer')
    for x in df.columns:
        if x not in ['make_target','modality_target','model_name_target', 'verified_model_name','mel_id']:   
            df[x] = df[x].fillna(0)
    return df


if __name__ == "__main__":
    # get inputs from customers
    glassbeam_file, non_glassbeam_files = get_serialized_filepaths()
    non_glassbeam_df = read_and_join_customers(non_glassbeam_files)
    glassbeam_df = get_glassbeam_data(glassbeam_file)
    # Split 'glassbeam' into 'full', 'partial', 'codev'
    glassbeam_df['company_name'] = glassbeam_df.apply(lambda row: split_glassbeam_row(row), axis=1)
    # Aggregate glassbeam df by company
    glassbeam_df = aggregate_df(glassbeam_df, is_glassbeam=True)
    non_glassbeam_df = aggregate_df(non_glassbeam_df, is_glassbeam=False)
    #Merge GB and non-GB dfs
    df = merge_gb_and_non_gb_dfs(non_glassbeam_df, glassbeam_df)
    # Postprocess
    df.to_csv(config.get_aggregate_customers_filepath(), index=False)

