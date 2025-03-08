import pandas as pd
import numpy as np
import glob
import os

def get_data(climate_data_path, monthly_data_dir):
    # Read climate data
    climate_data = pd.read_csv(climate_data_path)
    
    # Read all monthly prediction files
    monthly_files = glob.glob(os.path.join(monthly_data_dir, 'Energy_Prediction_*.csv'))
    monthly_data_list = []
    
    for file in monthly_files:
        df = pd.read_csv(file)
        monthly_data_list.append(df)
    
    # Combine all monthly data
    monthly_data = pd.concat(monthly_data_list, ignore_index=True)
    
    # Convert date columns to datetime
    climate_data['Date'] = pd.to_datetime(climate_data['Date'])
    monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])
    
    return climate_data, monthly_data
    