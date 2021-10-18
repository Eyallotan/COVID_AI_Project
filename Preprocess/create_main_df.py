import pandas as pd
from Preprocess import utils
from Preprocess import extract_daily_new_cases
from Preprocess import extract_vaccination_stats
from Preprocess import normalize_df

'''
This code generates the main corona data frame. We use external code that generates all columns 
and then aggregate all of the data. 
The only manual preprocessing we do is changing the color values to numbers. We do this manually 
since these values are written in Hebrew and it is a pain to replace it in the code.
Note that we join all data frames together based on the following keys: [City_Name,City_Code,
Date]. Therefore if you add another external data frame make sure it has exactly those keys to 
assure that the join works correctly.
'''

if __name__ == "__main__":
    # read main data frame
    corona_df = pd.read_csv('../Resources/corona_city_table_ver_00134.csv')
    corona_df['Date'] = pd.to_datetime(corona_df['Date'])
    params = utils.DataParams()

    # get rid of fields containing a "<15" value and replace them with 0
    for column in corona_df.filter(regex="Cumulative_.*|Cumulated_.*"):
        corona_df[column].replace({"<15": 0}, inplace=True)
        corona_df[column] = pd.to_numeric(corona_df[column])

    # reduce to the dates set by DataParams
    start_date = params.start_date
    end_date = params.end_date
    corona_df = corona_df[(corona_df['Date'] >= start_date) & (corona_df['Date'] <= end_date)]

    # get external columns
    vaccination_df = extract_vaccination_stats.generate_vaccination_columns()
    daily_new_cases_df = extract_daily_new_cases.generate_daily_new_cases_df()

    # join all data frames together. The key for the join op is [City_Name,City_Code,Date]
    temp_df = pd.merge(corona_df, vaccination_df, how="inner", on=["City_Name", "City_Code",
                                                                   "Date"])
    result_df = pd.merge(temp_df, daily_new_cases_df, how="inner", on=["City_Name", "City_Code",
                                                                       "Date"])

    result_df = normalize_df.normalize_data_set(result_df)

    # generate the output df
    utils.generate_output_csv(result_df, 'corona_df')
