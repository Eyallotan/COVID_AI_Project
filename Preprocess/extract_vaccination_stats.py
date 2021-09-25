import pandas as pd
from Preprocess.utils import DataParams


'''
This code extracts vaccination stats from for each city. The stats are calculated in the following
way:
For each of the vaccinations (1/2/3), there are N weekly stats columns.
If a row represents the date X , then column 1 (out of N) for vaccine 1 represents the accumulated 
number of people that were vaccinated with the first vaccine in 1 week prior to X. 
Column 2 represents 2 weeks prior to X and so on, up to N weeks prior to the date X.
The data frame also contains the total numbers of vaccinated people for each of the vaccines.

Date boundaries: 
The weekly sum columns are calculated based on N previous weeks. Since the data frame contains 
all cities together, we get an overlapping between the the start on one city to the end of the 
previous one. In the current implementation this overlapping DOES exists, so in order to get 
correct data we must choose the date boundaries using the following rule: If X is earliest date
in the data frame, and N is the number of weeks we use for weekly sums, the minimal day that 
will contain valid data is X+7*N days after X. This will also assure that all values are indeed 
based on N weeks. 
A better approach is to group by cities (a code demonstrating this approach was added in comment 
for now, in case we want to use it later). 
'''


def generate_vaccination_columns():
    # read csv file
    vaccinated_df = pd.read_csv('../Resources/vaccinated_city_table_ver_00156.csv')
    vaccinated_df['Date'] = pd.to_datetime(vaccinated_df['Date'])
    params = DataParams()

    # get rid of fields containing a "<15" value and replace them with 0
    for column in vaccinated_df:
        vaccinated_df[column].replace({"<15": 0}, inplace=True)
        if vaccinated_df.columns.get_loc(column) > 2:
            vaccinated_df[column] = pd.to_numeric(vaccinated_df[column])

    # the df contains vaccination stats by age. We want the total number of vaccinated so we sum
    # all ages for each vaccine. I also create a diff col to help with calculations later on.
    numbers = ['first', 'second', 'third']
    for i in range(1, 4):
        vaccinated_df[f'vaccinated_dose_{i}_total'] = vaccinated_df.filter(
            regex=f"{numbers[i-1]}_dose.*").sum(axis=1)
        vaccinated_df[f'dose_{i}_diff'] = \
            vaccinated_df[f'vaccinated_dose_{i}_total'] - vaccinated_df.shift(periods=1)[
                f'vaccinated_dose_{i}_total'].fillna(0)

    # add sum by week columns
    number_of_weeks = params.number_of_weeks_for_vaccination_stats
    for vaccine_number in range(1, 4):
        for week_number in range(1, number_of_weeks+1):
            vaccinated_df[f'dose_{vaccine_number}_in_last_{week_number}_week'] = vaccinated_df[
                f'dose_{vaccine_number}_diff'].rolling(min_periods=1, window=week_number*7).sum()

    # drop age and temp columns
    vaccinated_df = vaccinated_df[vaccinated_df.columns.drop(list(vaccinated_df.filter(
        regex="first_dose.*|second_dose.*|third_dose.*|_diff.*")))]

    '''
     # a better approach (no overlapping errors)
    grouped_cities = vaccinated_df.groupby(['CityName', 'CityCode'])
    for (_, _), df in grouped_cities:
        for vaccine_number in range(1, 4):
            for week_number in range(1, 5):
                df[f'dose_{vaccine_number}_in_last_{week_number}_week'] =\
                    df[f'dose_{vaccine_number}_diff'].rolling(min_periods=1, 
                    window=week_number*7).sum()
                # concat result to the main data frame
    '''

    # choose date ranges (see file prolog for more details)
    vaccinated_df = vaccinated_df[(vaccinated_df['Date'] >= params.start_date) &
                                  (vaccinated_df['Date'] <= params.end_date)]

    # rename first two columns. Since they are used as keys later on when we merge the data frames
    # together, they need to have identical names.
    col_names = {"CityName": "City_Name", "CityCode": "City_Code"}
    result_df = vaccinated_df.rename(columns=col_names)
    return result_df
