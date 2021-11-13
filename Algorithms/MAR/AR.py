# External includes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from time import time
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
    cities_df = pd.read_csv('../../Resources/corona_city_table_ver_00134.csv')

    # filter columns and rows
    cities_df = cities_df[['Date', 'City_Code', 'Cumulative_verified_cases']]
    out_df = cities_df.loc[cities_df['City_Code'] == city_code]

    # get rid of fields containing a "<15" value and replace them with 0
    out_df['Cumulative_verified_cases'].replace({"<15": 0}, inplace=True)
    out_df['Cumulative_verified_cases'] = pd.to_numeric(out_df['Cumulative_verified_cases'])
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
    transformer = DataTransformation(time_series.copy())
    s_test = StationarityTests()

    # Difference transformation
    time_series_diff = transformer.difference()
    plot_time_series(time_series_diff, 'Differenced time series', 'Diff values')
    s_test.ADF_Stationarity_Test(time_series_diff, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # TODO: Find the best diff operator

    # Square root transformation
    # Check for square root pattern by plotting histogram
    plt.hist(time_series, bins=30)
    plt.show()
    sqrt_time_series = transformer.sqrt()
    print('Sqrt time series:\n', sqrt_time_series.head())
    plot_time_series(sqrt_time_series, 'Square root transformation', 'Square root values')
    s_test.ADF_Stationarity_Test(sqrt_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # take diff on top of sqrt transformation
    sqrt_transform = DataTransformation(sqrt_time_series)
    sqrt_diff_time_series = sqrt_transform.difference(1)
    plot_time_series(sqrt_diff_time_series, 'Square root and Diff(1) transformation',
                     'Transformation values')
    s_test.ADF_Stationarity_Test(sqrt_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # Power transformation
    pow_transformation = transformer.pow()
    print('Power time series:\n', pow_transformation.head())
    plot_time_series(pow_transformation, 'Power of 2 transformation', 'Power values')
    s_test.ADF_Stationarity_Test(pow_transformation, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # Log transformation
    log_transformation = transformer.log(1)
    print('Log2 time series:\n', log_transformation.head())
    plot_time_series(log_transformation, 'Log transformation', 'Log values')
    s_test.ADF_Stationarity_Test(log_transformation, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # take diff on top of log transformation
    log_transformation = DataTransformation(log_transformation)
    log_diff_time_series = log_transformation.difference(1)
    plot_time_series(log_diff_time_series, 'Log and Diff(1) transformation', 'Transformation '
                                                                             'values')
    s_test.ADF_Stationarity_Test(log_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # Standardization transformation
    standardized_time_series = transformer.standardization()
    print('Standardized time series:\n', standardized_time_series.head())
    plot_time_series(standardized_time_series, 'Standardization transformation',
                     'Standardized values')

    s_test.ADF_Stationarity_Test(standardized_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))
    # take log on top of standardization transformation
    standardized_time_series = DataTransformation(standardized_time_series)
    standardized_log_time_series = standardized_time_series.log(1)
    plot_time_series(standardized_log_time_series, 'Standardization and Log transformation',
                     'Transformation values')
    s_test.ADF_Stationarity_Test(standardized_log_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # take diff on top of standardization and log transformation
    standardized_log_time_series = DataTransformation(standardized_log_time_series)
    standardized_log_diff_time_series = standardized_log_time_series.difference(1)
    plot_time_series(standardized_log_diff_time_series, 'Standardization, Log and Diff(1) '
                                                        'transformation', 'Transformation values')

    s_test.ADF_Stationarity_Test(standardized_log_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(s_test.is_stationary))

    # generate standardized time series and then apply inverse transformation to get back the
    # original time series
    transformer = DataTransformation(time_series.copy())
    standardized_time_series = transformer.standardization()
    print('Standardized time series:\n', standardized_time_series.head())
    original_time_series = transformer.invert_standardization(standardized_time_series)
    print('Original time series:\n', original_time_series.head())


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
    for month in range(1, 12):
        plt.axvline(pd.to_datetime('2021' + '-' + str(month) + '-01'), color='k',
                    linestyle='--', alpha=0.2)
    plt.axhline(time_series.iloc[:, 0].mean(), color='r', alpha=0.2, linestyle='--')

    plt.show()


def generate_time_series(city_code):
    """
    A simple function that creates a time series for a given city.
    :param city_code: The city that we want to create this time series for.
    :return: Time series dataset.
    """
    generate_daily_new_cases_csv(city_code)
    time_series = pd.read_csv('daily_new_cases_time_series.csv', index_col=0, parse_dates=True)
    time_series = time_series.asfreq(pd.infer_freq(time_series.index))

    print('Shape of data \t', time_series.shape)
    print('Time series:\n', time_series.head())

    return time_series


if __name__ == "__main__":
    time_series = generate_time_series(city_code=5000)

    # run transformations
    # test_transformations(time_series)

    transformer = DataTransformation(time_series)
    diff_time_series = transformer.difference(1)
    # Get training and test sets
    train_end = datetime(2021, 8, 5)
    test_end = datetime(2021, 8, 8)
    runner = AlgoRunner(diff_time_series, train_end, test_end)
    runner.run_arma_regressor(5, 6)







