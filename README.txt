To do before running the scripts:

MEL UPDATES
1. Get the latest MEL from Karen Holloway. Ensure it includes the fields "New ModelId", "New Model", "New Manufacturer", "New Lvl 2 Category"
2. Rename MEL fields if needed (e.g., "New_Lvl_2_Category_Name" -> "New Lvl 2 Category")
3. Replace all , by ; in MEL (otherwise, CSV format gets messed up)
4. Save MEL as mel.csv (with format UTF-8). Save it in a subfolder inside the same folder as this README file. You can name the subfolder however you want, but it's recommended to name it as {company name}_{date}. Outputs will be saved in the same subfolder

SOURCE DATA PREPROCESSING
5. Get the source data 
6. Check that source data has all the required columns: asset id, make, model name, modality, customer name
7. Replace all , by ; in the source data (otherwise, CSV format gets messed up)
8. In source data, Add "company_name" column. Usually, we get files from different companies (company = customer). If not (i.e. the assets come from different customers), don't add this column or add it and leave it blank
9. In source data, rename model column to "model_name_source"
10. In source data, rename manufacturer column to "make_source"
11. In source data, rename modality column to "modality_source"
12. In source data, rename model number column to "model_number" (without the _source suffix). If model number not found, no need to add it. This is not a required column
13. In source data, rename serial number or asset id to "asset_sys_id"
14. Save source data as source.csv in the same subfolder as the MEL(with UTF-8 encoding) 

RUN FILES
15. Run make_mapping.py
16. When prompted, enter the name of the source data file
17. Run model_mapping.py
18. When prompted, enter the name of the subfolder containing the MEL, the source data, and make_mapping.csv

Treat glassbeam files as customer files, with the following differences:
- don't add asset_sys_id
- add column product_mgmt_stage (if not added already)

