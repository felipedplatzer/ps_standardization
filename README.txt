To do before running the scripts:

API KEY (only needed for new users)
1. Create a file openai_api_key.txt in the same directory as this README file
2. Get an OpenAI key from openai.com
3. Copy the key and paste it into openai_api_key.txt (without quotes around it)

MEL UPDATES
4. Get the latest MEL from Karen Holloway. Ensure it includes the fields "New ModelId", "New Model", "New Manufacturer", "New Lvl 2 Category"
5. Rename MEL fields if needed (e.g., "New_Lvl_2_Category_Name" -> "New Lvl 2 Category")
6. Save MEL as mel.xlsx inside the files folder

SOURCE DATA PREPROCESSING
7. Get the source data 
8. If the source data is an Excel file, make sure it only has the required tab. Delete all other tabs 
9. Make sure the tab only contains the required dataset (e.g., no blank rows above it, no blank columns to the left of it) 
10. Make sure the data is unfiltered
11. Check that source data has all the required columns: asset id, make, model name, modality, customer name. For asset id, pick a column like "tag number" or "asset code" - the key is that this column should have as few duplicates as possible
12. In source data, Add "company_name" column and fill it with the name of the company (the company is the hospital / health system from where we get the CMMS data, e.g., UVA, INTERMOUNTAIN, etc). Put this name in all caps
13. In source data, rename model column to "model_name_source"
14. In source data, rename manufacturer column to "make_source"
15. In source data, rename modality column to "modality_source"
16. In source data, rename model number column to "model_number" (without the _source suffix). If model number not found, no need to add it. This is not a required column
17. In source data, rename serial number or asset id to "asset_sys_id"
18. Save source data in files/source_data as an xlsx file. Give it an appropriate name (e.g., name of customer + date)

RUN FILES
19. Run run_all.py
20. When prompted, enter the name of the source data file (from step 16)

Treat glassbeam files as customer files, with the following differences:
- don't add asset_sys_id
- add column product_mgmt_stage (if not added already)

