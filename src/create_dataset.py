import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from math import isnan

path_data_dir = 'data/'
list_assets = ["PETR3.SA","PRIO3.SA", "VALE3.SA", "GGBR3.SA", "ABCB4.SA", "ITUB3.SA", "FLRY3.SA", "RADL3.SA"]
relevant_cols = ['Date', 'Close', 'Volume']
moving_windows = [7,14,21]


# ###  Prepare the data
def expand_days(
    data
):

    # get all dates in the range
    date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max())

    # create df with all dates
    full_dates_df = pd.DataFrame({'Date': date_range})

    # merge dataframe with all dates and original 
    data = full_dates_df.merge(data, on='Date', how='left')
    
    return data

def calculate_features(data, window):
    
    data[f'diff_close_mean_{window}'] = data.Close - data[f'SMA_{window}']
    data[f'diff_close_mean_z_score_{window}'] = data[f'diff_close_mean_{window}'] / data[f'MSTD_{window}']
    data[f'diff_close_mean_z_score_{window}'] = data[f'diff_close_mean_z_score_{window}'].fillna(0)
    data[f'diff_close_mean_z_score_{window}_diff'] = data[f'diff_close_mean_z_score_{window}'].diff(1).fillna(0)
    data[f'meta_{window}'] = data[f'diff_close_mean_z_score_{window}'].apply(int)
    data[f'meta_{window}_diff'] = data[f'diff_close_mean_z_score_{window}']
    
    data[f'Close_diff'] = data[f'Close'].diff(1).fillna(0)
    
    
    return data 


def main():
    # #### create dataframe
    for asset in tqdm(list_assets):
        
        # get data
        data =  pd.read_excel(path_data_dir + f"raw/price_history_{asset.replace('.', '_')}.xlsx")[relevant_cols]
        
        data_original = data.copy()
        
        for window_size in moving_windows:
            # create moving average column    
            data[f'SMA_{window_size}'] = data[f'Close'].rolling(window = window_size).mean()
            # create moving standard deviation column
            data[f'MSTD_{window_size}'] = data[f'Close'].rolling(window = window_size).std()
        
        
        # remove n first rows that not enter in the SMA of the max window value (null data)
        max_window_size = max(moving_windows)
        data = data.iloc[max_window_size - 1:,:]
        
        # expand days
        data = expand_days(data)
        
        # ffill null data
        cols_to_ffill = [f'SMA_{window_size}',f'MSTD_{window_size}'] 
        data[cols_to_ffill] = data[cols_to_ffill].ffill()
        
        # create column with day of week
        data['Day_of_week'] = data.Date.dt.day_of_week
        # tag weekend
        data['Weekend'] = data.Day_of_week.apply(lambda x: 1 if x in [5,6] else 0)
        # data invalid days 
        data['Invalid_Days'] = data.Close.apply(lambda x: 1 if isnan(x) else 0)
        
        data.set_index('Date', inplace=True)
        
        data_filled = data.ffill()
        data_interpolate = data.interpolate()
        
        for window_size in moving_windows:
            data_filled = calculate_features(data_filled, window_size)
            data_interpolate = calculate_features(data_interpolate, window_size)
        
        data_filled.to_csv(path_data_dir + f"processed/price_history_{asset.replace('.', '_')}_meta_dataset_ffill.csv")
        data_interpolate.to_csv(path_data_dir + f"processed/price_history_{asset.replace('.', '_')}_meta_dataset_interpolate.csv")
        
        
if __name__ == "__main__":
    main()