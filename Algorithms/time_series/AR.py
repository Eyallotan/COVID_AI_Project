# External includes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
# Statsmodels
from statsmodels.tsa.stattools import adfuller

# Internal includes
from DataTransformation import DataTransformation
from AlgoRunner import AlgoRunner
from Preprocess import utils
from Preprocess.utils import DataParams

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'


def generate_daily_new_cases_csv(city_code):
    """
    Generate a csv containing daily new cases by date.
    :param city_code: City code of the desired city.
    :return: No return. The function will generate a csv file called
    'daily_new_cases_time_series' that will contain two columns - date and number of daily cases.
    """
    # read csv file
    cities_df = pd.read_csv('../../Resources/corona_city_table_ver_00155.csv')

    # filter columns and rows
    cities_df = cities_df[['Date', 'City_Code', 'Cumulative_verified_cases']]
    out_df = cities_df.loc[cities_df['City_Code'] == city_code]

    city_codes = list(dict.fromkeys(cities_df['City_Code'].values))
    # get rid of fields containing a "<15" value and replace them with ascending sequence of
    # number from 1 to 14
    column = 'Cumulative_verified_cases'
    for city in city_codes:
        count = cities_df.loc[(cities_df[column] == "<15") & (cities_df['City_Code'] ==
                                                              city), column].count()
        factor = count / 14
        # if factor is less than 1, put the mean value in all fields and call it a day..
        if 0 <= factor < 1:
            cities_df.loc[(cities_df[column] == "<15") & (cities_df['City_Code'] ==
                                                          city), column] = 7
        else:
            number_of_rows_for_each_value = math.floor(factor)
            counter = 0
            i = 1
            for j, row in out_df.iterrows():
                if row['City_Code'] == city and row[column] == "<15":
                    out_df.at[j, column] = i
                    counter += 1
                    if counter == number_of_rows_for_each_value:
                        i += 1
                        counter = 0
                        if i == 15:
                            break
        print(f'processed column {column} for city {city}')
    out_df[column].replace({"<15": 14}, inplace=True)
    out_df[column] = pd.to_numeric(out_df[column])

    out_df.reset_index(inplace=True)

    # generate the daily new cases column
    out_df['daily_new_cases'] = \
        out_df['Cumulative_verified_cases'] - out_df.shift(periods=1)[
            'Cumulative_verified_cases']
    out_df = out_df.dropna()

    # drop redundant columns
    out_df.drop(columns=['index', 'City_Code', 'Cumulative_verified_cases'], inplace=True)

    # If you want to change the dates in the time series use the following code:
    # params = DataParams()
    # out_df['Date'] = pd.to_datetime(out_df['Date'])
    # out_df = out_df[(out_df['Date'] >= params.start_date) & (out_df['Date'] <= params.end_date)]

    utils.generate_output_csv(out_df, 'daily_new_cases_time_series')


class StationarityTests:
    def __init__(self, significance=.05):
        self.significance_level = significance
        self.p_value = None
        self.is_stationary = None

    def ADF_Stationarity_Test(self, time_series, printResults=True):
        # Dickey-Fuller test:
        adf_test = adfuller(time_series, autolag='AIC')

        self.p_value = adf_test[1]

        if self.p_value < self.significance_level:
            self.is_stationary = True
        else:
            self.is_stationary = False

        if printResults:
            df_results = pd.Series(adf_test[0:4],
                                  index=['ADF Test Statistic', 'P-Value', '# Lags Used',
                                         '# Observations Used'])

            # Add Critical Values
            for key, value in adf_test[4].items():
                df_results['Critical Value (%s)' % key] = value

            print('Augmented Dickey-Fuller Test Results:')
            print(df_results)


def test_transformations(time_series):
    plot_time_series(time_series, 'Original time series', 'Values')
    # Check for patterns by plotting histogram
    plt.hist(time_series, bins=30)
    plt.show()
    transformer = DataTransformation(time_series.copy())
    s_test = StationarityTests()

    # Difference transformation
    time_series_diff = transformer.difference(lags=1)
    plot_time_series(time_series_diff, 'Diff(1) Transformation', 'Diff values')
    s_test.ADF_Stationarity_Test(time_series_diff, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # invert diff
    transformer.invert_difference(time_series_diff, time_series.copy(), lags=1)
    # TODO: Find the best diff operator

    # Square root transformation
    sqrt_time_series = transformer.sqrt()
    print('Sqrt time series:\n', sqrt_time_series.head())
    plot_time_series(sqrt_time_series, 'Square root Transformation', 'Square root values')
    s_test.ADF_Stationarity_Test(sqrt_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # take diff on top of sqrt transformation
    sqrt_diff_time_series = transformer.difference(1)
    plot_time_series(sqrt_diff_time_series, 'Square root and Diff(1) Transformation',
                     'Transformation values')
    s_test.ADF_Stationarity_Test(sqrt_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # invert transformations
    transformer.invert_difference(sqrt_diff_time_series, sqrt_time_series, lags=1)
    inverted = transformer.pow()
    # print inverted time series
    plot_time_series(inverted, 'Inverted', 'Values')

    # Power transformation
    pow_transformation = transformer.pow()
    print('Power time series:\n', pow_transformation.head())
    plot_time_series(pow_transformation, 'Power of 2 Transformation', 'Power values')
    s_test.ADF_Stationarity_Test(pow_transformation, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # invert
    inverted = transformer.sqrt()
    # print inverted time series
    plot_time_series(inverted, 'Inverted', 'Values')

    # Log transformation
    log_transformation = transformer.log(increment_val=1)
    print('ln time series:\n', log_transformation.head())
    plot_time_series(log_transformation, 'ln(1) Transformation', 'ln(1) values')
    s_test.ADF_Stationarity_Test(log_transformation, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # take diff on top of log transformation
    log_diff_time_series = transformer.difference(lags=1)
    plot_time_series(log_diff_time_series, 'ln(1) and Diff(1) Transformation',
                     'Transformation values')
    s_test.ADF_Stationarity_Test(log_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # invert transformations
    transformer.invert_difference(log_diff_time_series, log_transformation, lags=1)
    inverted = transformer.exp(1)
    # print inverted time series
    plot_time_series(inverted, 'Inverted', 'Values')

    # Standardization transformation
    standardized_time_series = transformer.standardization()
    print('Standardized time series:\n', standardized_time_series.head())
    plot_time_series(standardized_time_series, 'Standardization transformation',
                     'Standardized values')

    s_test.ADF_Stationarity_Test(standardized_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # take log on top of standardization transformation
    standardized_log_time_series = transformer.log(increment_val=1)
    plot_time_series(standardized_log_time_series, 'Standardization and ln(1) Transformation',
                     'Transformation values')
    s_test.ADF_Stationarity_Test(standardized_log_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # take diff on top of standardization and log transformation
    standardized_log_diff_time_series = transformer.difference(lags=1)
    plot_time_series(standardized_log_diff_time_series, 'Standardization, ln(1) and Diff(1) '
                                                        'Transformation', 'Transformation values')
    s_test.ADF_Stationarity_Test(standardized_log_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # invert transformations
    transformer.invert_difference(standardized_log_diff_time_series,
                                  standardized_log_time_series, lags=1)
    transformer.exp(decrement_val=1)
    inverted = transformer.invert_standardization()
    plot_time_series(inverted, 'Inverted', 'Values')


def plot_time_series(time_series, title=None, ylabel=None):
    """
    Plot a time series graph. Plotting plot lines from March-2020 until November-2021.
    TODO: make the plotting dates customizable.
    :param time_series: Time series to plot.
    :param title: Plot title.
    :param ylabel: Y axis label.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time_series)
    plt.title(title, fontsize=20)
    plt.ylabel(ylabel, fontsize=16)
    for month in range(3, 13):
        plt.axvline(pd.to_datetime('2020' + '-' + str(month) + '-01'), color='k',
                    linestyle='--', alpha=0.2)
    for month in range(1, 10):
        plt.axvline(pd.to_datetime('2021' + '-' + str(month) + '-01'), color='k',
                    linestyle='--', alpha=0.2)
    plt.axhline(time_series.iloc[:, 0].mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(time_series.iloc[:, 0].std(), color='b', alpha=0.2, linestyle='--')

    plt.show()


def generate_time_series_for_city(city_code):
    """
    A simple function that creates a time series for a given city.
    :param city_code: The city that we want to create this time series for.
    :return: Time series dataset.
    """
    # generate_daily_new_cases_csv(city_code)
    time_series = pd.read_csv('daily_new_cases_time_series.csv', index_col=0, parse_dates=True)
    time_series = time_series.asfreq(pd.infer_freq(time_series.index))

    print('Shape of data \t', time_series.shape)
    print('Time series:\n', time_series)

    return time_series


def generate_rolling_average_series():
    """
    A simple function that creates a time series that holds for each day the 7 day rolling
    average of new cases.
    :return: Time rolling average time series.
    """
    time_series = pd.read_csv('../../Resources/7_days_rolling_avg_global.csv', index_col=0)
    time_series.index = pd.to_datetime(time_series.index, format='%d-%m-%Y')
    start_date = datetime(2020, 2, 12)
    end_date = datetime(2021, 11, 22)
    time_series = time_series[(time_series.index >= start_date) & (time_series.index <=
                                                                   end_date)]

    print('Shape of data \t', time_series.shape)
    print('Time series:\n', time_series)

    return time_series

def generate_daily_cases_national():
    """
    A simple function that creates a time series that holds for each day the total number of new
    cases.
    :return: Time daily new cases time series.
    """
    time_series = pd.read_csv('../../Resources/daily_cases_global.csv', index_col=0)
    time_series.index = pd.to_datetime(time_series.index, format='%d-%m-%Y')
    start_date = datetime(2020, 2, 12)
    end_date = datetime(2021, 11, 22)
    time_series = time_series[(time_series.index >= start_date) & (time_series.index <=
                                                                     end_date)]
    print('Shape of data \t', time_series.shape)
    print('Time series:\n', time_series)

    return time_series


if __name__ == "__main__":
    # Define the time series we want to work with
    time_series = generate_time_series_for_city(city_code=5000)
    rolling_average_series = generate_rolling_average_series()
    national_daily_cases = generate_daily_cases_national()

    # plot_time_series(national_daily_cases, 'Daily new cases', 'New Cases')
    # plot_time_series(rolling_average_series, '7 Days rolling average', 'Avg value')

    # Apply transformations (if needed)
    transformer = DataTransformation(national_daily_cases.copy())
    time_series_log = transformer.log(increment_val=1)
    time_series_transformed = transformer.difference(lags=1)
    # Test for stationarity
    s_test = StationarityTests()
    s_test.ADF_Stationarity_Test(time_series_transformed, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # Define training and test sets
    train_end = datetime(2021, 8, 10)
    test_end = datetime(2021, 8, 17)

    # Run the AlgoRunner with original time series
    runner1 = AlgoRunner(time_series_transformed, train_end, test_end)
    runner1.plot_correlation_plots(number_of_lags=25)
    runner1.run_ma_regressor(7, use_rolling_forecast=True)
    # Run the AlgoRunner with transformed time series
    runner2 = AlgoRunner(time_series_transformed, train_end, test_end,
                         original_time_series=rolling_average_series,
                         transformations=['log', 'difference'],
                         diff_params=(1, time_series_log),
                         log_exp_delta=1)
    runner2.plot_correlation_plots(number_of_lags=25)
    runner2.run_ma_regressor(7, use_rolling_forecast=True)









