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


def generate_output(df, output_name):
    df.to_csv(f'{output_name}.csv', index=False, encoding='utf-8-sig')

