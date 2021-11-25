from datetime import datetime
import pandas as pd
import math

'''
This file contains parameters that are used across all of our data frames. We define them here to 
make sure we use the same params. 
'''


class DataParams:
    def __init__(self):
        self.start_date = datetime(2021, 1, 5)
        self.end_date = datetime(2021, 11, 23)
        self.split_date = datetime(2021, 10, 15)
        self.number_of_weeks_for_vaccination_stats = 2
        self.number_of_days_for_infected_stats = 14
        self.normalization_factor = 1
        self.not_normalized_columns = ['City_Name', 'City_Code', 'Date', 'colour', 'final_score',
                                       'today_verified_cases', 'today_verified_cases_smoothed']
        self.Y = 'today_verified_cases'
        # self.Y = 'today_verified_cases_smoothed'
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


def preprocess_raw_dataset():
    # read main data frame
    corona_df = pd.read_csv('../Resources/corona_city_table_ver_00155.csv')
    corona_df['Date'] = pd.to_datetime(corona_df['Date'])
    params = DataParams()

    # reduce to the dates set by DataParams
    start_date = params.start_date
    end_date = params.end_date
    corona_df = corona_df[(corona_df['Date'] >= start_date) & (corona_df['Date'] <= end_date)]

    city_codes = list(dict.fromkeys(corona_df['City_Code'].values))
    # get rid of fields containing a "<15" value and replace them with ascending sequence of
    # number from 1 to 14
    for column in corona_df.filter(regex="Cumulative_.*|Cumulated_.*"):
        for city in city_codes:
            count = corona_df.loc[(corona_df[column] == "<15") & (corona_df['City_Code'] ==
                                                                  city), column].count()
            factor = count / 14
            # if factor is less than 1, put the mean value in all fields and call it a day..
            if 0 <= factor < 1:
                corona_df.loc[(corona_df[column] == "<15") & (corona_df['City_Code'] ==
                                                              city), column] = 7
            else:
                number_of_rows_for_each_value = math.floor(factor)
                counter = 0
                i = 1
                for j, row in corona_df.iterrows():
                    if row['City_Code'] == city and row[column] == "<15":
                        corona_df.at[j, column] = i
                        counter += 1
                        if counter == number_of_rows_for_each_value:
                            i += 1
                            counter = 0
                            if i == 15:
                                break
            print(f'processed column {column} for city {city}')
        corona_df[column].replace({"<15": 14}, inplace=True)
        corona_df[column] = pd.to_numeric(corona_df[column])

    # generate the output df
    generate_output_csv(corona_df, 'corona_city_table_preprocessed')


if __name__ == "__main__":
    preprocess_raw_dataset()

