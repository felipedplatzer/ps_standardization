import os
import pandas as pd
import config

_subfolders_to_exclude = ['__pycache__', 'batch_files']


def get_customer_folder_names():
    customer_folder_names = [f for f in os.listdir() if os.path.isdir(f) and f not in _subfolders_to_exclude]
    #print names of all folders, ask "do you want to continue?"
    print("The following folders were found:")
    print(customer_folder_names)
    continue_input = input("Do you want to continue? (y/n)")
    if continue_input.lower().strip() == "y":
        return customer_folder_names
    else:
        exit()


""" def get_customer_folder_names():
    #get all folders in the current directory
    customer_folder_names = [f for f in os.listdir() if os.path.isdir(f)]
    #print names of all folders, ask "do you want to continue?"
    print(customer_folder_names)
    continue_input = input("Do you want to continue? (y/n)")
    if continue_input.lower().strip() == "Y":
        #ask we found this folder with "glassbeam" in the name. Is this the glassbeam folder? 
        glassbeam_folder_name = input("We found this folder with 'glassbeam' in the name. Is this the correct glassbeam folder? (y/n)")
        if glassbeam_folder_name.lower().strip() == "Y":
            return customer_folder_names, None
        if glassbeam_folder_name.lower().strip() == "N":
            glassbeam_folder_name = input("What is the name of the glassbeam folder?")
            return customer_folder_names, glassbeam_folder_name
    else:
        exit()
 """

def get_customer_data(customer_folder_name):
    try:
        df = pd.read_csv(f"{customer_folder_name}/serialized_asset_view.csv", dtype=str)
    except:
        print(f"No serialized asset view found for {customer_folder_name}")
        exit()
    return df

def get_glassbeam_data(glassbeam_folder_name):
    try:
        df = pd.read_csv(f"{glassbeam_folder_name}/model_mapping.csv", dtype=str)
    except:
        print(f"No serialized asset view found for {glassbeam_folder_name}")
        exit()
    return df

def split_glassbeam(row):
    if row['company_name'].strip().upper() != 'GLASSBEAM':
        return row['company_name']
    else:
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
    



def postprocess_df(df):
    df['full'] = df['full'].fillna('')
    df['codev'] = df['codev'].fillna('')
    df['partial'] = df['partial'].fillna('')
    df['company_name'] = df.apply(lambda row: split_glassbeam(row), axis=1)
    # mask unmapped models
    df['ps_model_name'] = df['ps_model_name'].fillna('')
    df['model_name_source'] = df['model_name_source'].fillna('')
    df[['ps_model_name','verified_model_name']] = df.apply(lambda row: mask_unmapped_models(row), axis=1)
    # aggregate
    df = df.groupby(['company_name','ps_make','ps_modality','ps_model_name']).agg({'asset_sys_id': 'size','verified_model_name': 'first'}).reset_index()
    df.rename(columns={'asset_sys_id': 'count'}, inplace=True)
    #df['count'] = df.apply(lambda row: correct_glassbeam_count(row), axis=1)
    # pivot by customer
    df = df.pivot(index=['ps_make','ps_modality','ps_model_name', 'verified_model_name'], columns='company_name', values='count')
    df.fillna(0, inplace=True)
    df.reset_index(inplace=True)
    return df

if __name__ == "__main__":
    # get inputs from customers
    customer_folder_names = get_customer_folder_names()
    # Aggregate customer data
    df_list = []
    for customer_folder_name in customer_folder_names:
        df = get_customer_data(customer_folder_name)
        df_list.append(df)
    df = pd.concat(df_list)
    # Postprocess
    output_df = postprocess_df(df)
    output_df.to_csv(config.get_aggregate_customers_filepath(), index=False)


"""


import pandas as pd
import os
import glob
from pathlib import Path
import config
from datetime import datetime

CODEV_VALUES = ['Y'] # ADD ASSESS IF WE SHOULD CONSIDER 'ASSESS' AS CODEV

def get_all_subfolders():
    current_dir = Path('.')
    subfolders = [f for f in current_dir.iterdir() if f.is_dir()]
    return subfolders

def get_serialized_data():
    subfolders = get_all_subfolders()
    df_list = []
    for subfolder in subfolders:
        if config.get_serialized_asset_view_filepath(subfolder).exists():
            df = pd.read_csv(config.get_serialized_asset_view_filepath(subfolder))
            df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    return df


def get_glassbeam_data():
    glassbeam_subfolder = input('Enter the glassbeam subfolder name: ')
    df = pd.read_csv(config.get_model_mapping_filepath(glassbeam_subfolder))
    return df


def get_overview(serialized_data, glassbeam_data):
    # process serialized data
    # group by ps_make, ps_model_name, ps_modality, company_name
    # pivot by company_name
    serialized_data = serialized_data.groupby(['ps_make', 'ps_model_name', 'ps_modality', 'company_name']).size().reset_index(name='count')
    serialized_data = serialized_data.pivot(index=['ps_make', 'ps_model_name', 'ps_modality'], columns='company_name', values='count')
    serialized_data.fillna(0, inplace=True)
    serialized_data.reset_index(inplace=True)
    serialized_data.columns.name = None
    serialized_data.index.name = None
    # get glassbeam data
    # create column gb_coverage = 'full' if 'full' is Y, partial if full is N and partial is Y, codev if full is N, partial is N, and codev is one of the values listed above
    glassbeam_data['gb_coverage'] = glassbeam_data.apply(lambda x: 'full' if x['full'].strip().upper() == 'Y' else 'partial' if x['full'].strip().upper() == 'N' and x['partial'].strip().upper() == 'Y' else 'codev' if x['full'].strip().upper() == 'N' and x['partial'].strip().upper() == 'N' and x['codev'].strip().upper() in CODEV_VALUES else 'no coverage', axis=1)
    overview = pd.concat([serialized_data, glassbeam_data], ignore_index=True)
    return overview

if __name__ == "__main__":
    serialized_data = get_serialized_data()
    glassbeam_data = get_glassbeam_data()
    overview_df = get_overview(serialized_data, glassbeam_data)
    overview_df.to_csv('./overview_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))"""