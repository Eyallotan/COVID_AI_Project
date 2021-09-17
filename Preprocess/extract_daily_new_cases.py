import pandas as pd
from datetime import datetime

def before_date():
    march20 = datetime(2020, 3, 28)
    march20_df = cities_df[cities_df['Date'] < march20]


if __name__ == "__main__":
    cities_df = pd.read_csv('../Resources/corona_city_table_ver_00134.csv')
    int_columns = ['Cumulative_verified_cases', 'Cumulated_recovered', 'Cumulated_deaths', 'Cumulated_number_of_tests',
                   'Cumulated_number_of_diagnostic_tests']

    cities_df['Date'] = pd.to_datetime(cities_df['Date'])

    for column_name in int_columns:
        cities_df[column_name] = cities_df[column_name].replace('<15', 0)
        cities_df[column_name] = pd.to_numeric(cities_df[column_name])

    # Generate N columns of previous days new cases
    # for city_code in cities_df['City_Code'].unique():
    #TODO: need to do this to every city seperately because of the shifts
    N = 8
    for i in range(1, N):
        cities_df[f'tmp_{i}'] = cities_df['Cumulative_verified_cases'].shift(periods=i)
        if i == 1:
            cities_df[f'verified_cases_{i}_days_ago'] = cities_df['Cumulative_verified_cases'] - cities_df[f'tmp_{i}']
        else:
            cities_df[f'verified_cases_{i}_days_ago'] = cities_df[f'tmp_{i-1}'] - cities_df[f'tmp_{i}']

    # tmp code to test new columns
    march20 = datetime(2020, 3, 28)
    tmp_df = cities_df[(cities_df['Date'] < march20) & (cities_df['City_Code'] == 4000)]

    tmp_df = tmp_df[['Cumulative_verified_cases', 'verified_cases_1_days_ago', 'verified_cases_2_days_ago',
                     'verified_cases_3_days_ago', 'verified_cases_4_days_ago', 'verified_cases_5_days_ago',
                     'verified_cases_6_days_ago', 'verified_cases_7_days_ago']]

    print('~~~~')
