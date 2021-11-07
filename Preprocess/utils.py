from datetime import datetime

'''
This file contains parameters that are used across all of our data frames. We define them here to 
make sure we use the same params. 
'''


class DataParams:
    def __init__(self):
        self.start_date = datetime(2021, 1, 17)
        self.end_date = datetime(2021, 11, 9)
        self.number_of_weeks_for_vaccination_stats = 4
        self.number_of_days_for_infected_stats = 14
        self.normalization_factor = 1
        self.not_normalized_columns = ['City_Name', 'City_Code', 'Date', 'colour', 'final_score', 'today_verified_cases']
        self.Y = 'today_verified_cases'
        self.split_test_size = 0.2
        self.split_random_state = 1


def generate_output_csv(df, output_name):
    """
   Parameters
   ----------
   df : dataframe object
       the dataframe we want to convert to csv
   output_name : str
       csv file name (no need to add .csv suffix)
    """
    df.to_csv(f'{output_name}.csv', index=False, encoding='utf-8-sig')

