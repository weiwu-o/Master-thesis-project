import zipfile
import os
import itertools

import pandas as pd

def return_filepaths(directory):
    json_data=[]
    # walk through the folders 
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            filepath=os.path.join(root, name)
            json_data.append(ETL(filepath))
    return json_data

####
def ETL(foldername):
    
    json_data=[]
    
    foldername_long=foldername
    
    # here we split the foldername to obtain the year/month, day, project name and artifact
    foldername=foldername.split('\\')
    foldername=foldername[-1]
    foldername=foldername.split('.')[1:-1]
    
    # open the folder
    with zipfile.ZipFile(foldername_long,"r") as zfile:
        # read the files that have "ETL" in the name
        for name in zfile.namelist():
            if "ETL" in name:
                temp=zfile.read(name)
                json_data.append({'Year_month': foldername[0], 
                                  'Count': foldername[1], 
                                  'Project':foldername[2]+' '+foldername[3], 
                                  'Instance':foldername[4], 
                                  'Filename':name, 
                                  'Address': foldername_long,
                                  'Log': temp.decode('utf-8')})
    return json_data

####

def create_dataframe(folder_path):
    jsons=return_filepaths(folder_path)
    flat_jsons=[item for sublist in jsons for item in sublist]
    
    df=pd.DataFrame.from_records(flat_jsons)
    return df