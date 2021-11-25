import pandas as pd
from Preprocess import utils


def normalize_data_set(result_df):
    # read population csv file
    population_df = pd.read_csv('../Resources/population_table.csv')
    params = utils.DataParams()

    # create a new df with population column included
    result_normalized_df = pd.merge(result_df, population_df, how="inner", on=["City_Code"])

    # append population column to columns we don't want to normalize
    params.not_normalized_columns.append('population')

    # normalizing df by calculating percentage per normalization_factor
    for column in result_normalized_df:
        if column not in params.not_normalized_columns:
            result_normalized_df[column] = (result_normalized_df[column] * (params.normalization_factor)) / result_normalized_df['population']

    # remove population column from columns we don't want to normalize
    params.not_normalized_columns.remove('population')
    # remove population column from result normalized df
    del result_normalized_df['population']

    return result_normalized_df
