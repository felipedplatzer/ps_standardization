import os
import pandas as pd
import config


def get_serialized_filepaths():
    x = os.listdir('./files/serialized_asset_views')
    x = [f for f in x if f.endswith('.csv')]
    print("The following files were found:\n")
    for i,f in enumerate(x):
        print(f"{i+1}. {f}")
    continue_input = input("Do you want to continue? (y/n)")
    if continue_input.lower().strip() == "y":
        while True:
            i = input("Enter the number of the GLASSBEAM file: ")
            try:
                i = int(i)
                if i > 0 and i <= len(x):
                    continue_input = input(f"You selected file: {x[i-1]} as the GLASSBEAM file. Continue (y/n)? ")
                    if continue_input.lower().strip() == "y":
                        glassbeam_file = x[i-1]
                        non_glassbeam_files = [f for i,f in enumerate(x) if i != i-1]
                        return glassbeam_file, non_glassbeam_files
                    else:
                        exit()
                else:
                    print("Invalid input")
            except:
                print("Invalid input")
    else:
        exit()


def read_and_join_customers(non_glassbeam_files):
    df_list = []
    for f in non_glassbeam_files:
        try:
            df = pd.read_csv(f"./files/serialized_asset_views/{f}")
            df = df.fillna('').astype(str)
        except:
            print(f"Error reading {f}")
        df_list.append(df)
    df = pd.concat(df_list)
    df = df.fillna('').astype(str)
    return df

def get_glassbeam_data(glassbeam_filepath):
    try:
        df = pd.read_csv(f"./files/serialized_asset_views/{glassbeam_filepath}")
        df = df.fillna('').astype(str)
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

def mask_unmapped_models(row):
    if row['ps_model_name'].strip() != '':
        return pd.Series([row['ps_model_name'], 1])
    else:
        return pd.Series([row['model_name_source'], 0])


def correct_glassbeam_count(row): #need corrections because of issues in the glassbeam data
    if 'GLASSBEAM' in row['company_name'].strip().upper():
        return min(row['count'], 1) #remove duplicates
    else:
        return row['count']
    


def aggregate_glassbeam_df(df):
    df[['ps_model_name','verified_model_name']] = df.apply(lambda row: mask_unmapped_models(row), axis=1)
    df = df.groupby(['company_name','ps_make','ps_modality','ps_model_name']).agg(count=('verified_model_name', 'size'), verified_model_name=('verified_model_name', 'first')).reset_index()
    df['exists_bool'] = df['count'].apply(lambda x: 1 if x > 0 else 0)
    df = df.pivot(index=['ps_make','ps_modality','ps_model_name', 'verified_model_name'], columns='company_name', values='exists_bool').reset_index()
    
    return df


def aggregate_non_glassbeam_df(df):
    df[['ps_model_name','verified_model_name']] = df.apply(lambda row: mask_unmapped_models(row), axis=1)
    df = df.groupby(['company_name','ps_make','ps_modality','ps_model_name']).agg({'asset_sys_id': 'size','verified_model_name': 'first'}).reset_index()
    df = df.rename(columns={'asset_sys_id': 'count'})
    df = df.pivot(index=['ps_make','ps_modality','ps_model_name', 'verified_model_name'], columns='company_name', values='count').reset_index()
    return df

def merge_gb_and_non_gb_dfs(non_glassbeam_df, glassbeam_df):
    df = pd.merge(non_glassbeam_df, glassbeam_df, on=['ps_make','ps_modality','ps_model_name', 'verified_model_name'], how='outer')
    for x in df.columns:
        if x not in ['ps_make','ps_modality','ps_model_name', 'verified_model_name']:   
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
    glassbeam_df = aggregate_glassbeam_df(glassbeam_df)
    non_glassbeam_df = aggregate_non_glassbeam_df(non_glassbeam_df)
    #Merge GB and non-GB dfs
    df = merge_gb_and_non_gb_dfs(non_glassbeam_df, glassbeam_df)
    # Postprocess
    df.to_csv(config.get_aggregate_customers_filepath(), index=False)

