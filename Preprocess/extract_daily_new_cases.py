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

    print(cities_df.dtypes)
    print('~~~~')

