import pandas as pd
from datetime import datetime
'''
right now every first row (first day) on each city has wrong value because of the shifts.
this bug is irellevant if we dont need the first few days - so i think its ok
if we want to calculate the first few days good, need to work on each city seperately or fix each first row with special
treatment.
for each row(for city_code in cities_df['City_Code'].unique())    
'''


def before_date():
    march20 = datetime(2020, 3, 28)
    march20_df = cities_df[cities_df['Date'] < march20]


if __name__ == "__main__":
    # output_csv = pd.read_csv('output.csv')
    cities_df = pd.read_csv('../Resources/corona_city_table_ver_00134.csv')
    int_columns = ['Cumulative_verified_cases', 'Cumulated_recovered', 'Cumulated_deaths', 'Cumulated_number_of_tests',
                   'Cumulated_number_of_diagnostic_tests']

    cities_df['Date'] = pd.to_datetime(cities_df['Date'])

    for column_name in int_columns:
        cities_df[column_name] = cities_df[column_name].replace('<15', 0)
        cities_df[column_name] = pd.to_numeric(cities_df[column_name])

    # Generate N columns of previous days new cases
    N = 16
    for i in range(1, N):
        cities_df[f'tmp_{i}'] = cities_df['Cumulative_verified_cases'].shift(periods=i)
        if i == 1:
            cities_df[f'verified_cases_{i-1}_days_ago'] = cities_df['Cumulative_verified_cases'] - cities_df[f'tmp_{i}']
        else:
            cities_df[f'verified_cases_{i-1}_days_ago'] = cities_df[f'tmp_{i-1}'] - cities_df[f'tmp_{i}']

    for i in range(1, N):
        cities_df.drop([f'tmp_{i}'], axis=1, inplace=True)
    cities_df.rename(columns={'verified_cases_0_days_ago': 'today_verified_cases'}, inplace=True)

    result_columns = ['City_Name', 'City_Code', 'Date', 'Cumulative_verified_cases', 'today_verified_cases']
    for i in range(2, N):
        result_columns.append(f'verified_cases_{i - 1}_days_ago')

    # Need min_date because of the bug told upper
    min_date = datetime(2020, 5, 1)
    result_df = cities_df[cities_df['Date'] > min_date][result_columns]
    result_df.to_csv('output.csv', index=False)
    # ~~~~~~ Finished ~~~~~~

    # tmp code to test new columns
    start_date = datetime(2020, 7, 20)
    end_date = datetime(2020, 11, 20)
    interesting_columns = ['Cumulative_verified_cases', 'today_verified_cases', 'verified_cases_2_days_ago',
                         'verified_cases_3_days_ago', 'verified_cases_4_days_ago', 'verified_cases_5_days_ago',
                         'verified_cases_6_days_ago', 'verified_cases_7_days_ago']

    # we work in specified date range
    result_df = cities_df[(cities_df['Date'] > start_date) & (cities_df['Date'] < end_date)]

    tlv_df = result_df[result_df['City_Code'] == 5000][interesting_columns]

    march_20 = datetime(2020, 3, 28)
    haifa_df = cities_df[(cities_df['Date'] < march_20) & (cities_df['City_Code'] == 4000)]

    print('~~~~')
