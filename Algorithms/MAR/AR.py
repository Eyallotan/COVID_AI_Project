import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler

from Preprocess import utils
from Preprocess.utils import DataParams

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from statsmodels.tsa.ar_model import AutoReg


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
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    def ADF_Stationarity_Test(self, time_series, printResults=True):
        # Dickey-Fuller test:
        adfTest = adfuller(time_series, autolag='AIC')

        self.pValue = adfTest[1]

        if self.pValue < self.SignificanceLevel:
            self.isStationary = True
        else:
            self.isStationary = False

        if printResults:
            dfResults = pd.Series(adfTest[0:4],
                                  index=['ADF Test Statistic', 'P-Value', '# Lags Used',
                                         '# Observations Used'])

            # Add Critical Values
            for key, value in adfTest[4].items():
                dfResults['Critical Value (%s)' % key] = value

            print('Augmented Dickey-Fuller Test Results:')
            print(dfResults)


class DataTransformation:
    """
    Class for performing transformations on time series data sets.
    For each transformation operator there is a inverse operator that can transform the data
    back to it's original form.
    """
    def __init__(self, data, value_name):
        self.TimeSeries = data
        self.value_name = value_name
        self.scaler = StandardScaler()

    def difference(self, interval=1):
        """
        Transform to a difference time series with a certain interval.
        :param interval: The difference operator.
        :return: Time series with applied diff.
        """
        assert interval > 0
        return self.TimeSeries.diff(interval).dropna()

    def sqrt(self):
        """
        Transform by taking a square root of all values.
        :return: Time series with applied square root transformation.
        """
        sqrt_time_series = self.TimeSeries.copy()
        sqrt_time_series[self.value_name] = np.sqrt(sqrt_time_series[self.value_name])
        return sqrt_time_series

    def pow(self):
        """
        Transform by applying power of 2 to all values.
        :return: Time series with applied pow transformation.
        """
        pow_time_series = self.TimeSeries.copy()
        pow_time_series[self.value_name] = np.power((pow_time_series[self.value_name]), 2)
        return pow_time_series

    def log(self, increment_val=0):
        """
        Transform by applying log (base 2) to all values.
        :param increment_val: Log function can only be applied to numbers higher than zero,
        so if the time series has values <= 0 you should provide the increment val that will be
        added to all values in order for the log function to work properly. Note that the same
        increment_val should be provided when inverting the time series back.
        :return: Time series with applied log transformation.
        """
        log_time_series = self.TimeSeries.copy()
        log_time_series += increment_val
        log_time_series[self.value_name] = np.log2((log_time_series[self.value_name]))
        return log_time_series

    def exp(self, decrement_val=0):
        """
        Transform by applying exponent(raise to the power of the natural exponent e) to all values.
        :param decrement_val: If this function is used to revert the log operator, a decrement
        value will be subtracted after applying the exponent function (used to restore original
        values that might have been <= 0).
        :return: Time series with applied exp transformation.
        """
        exp_time_series = self.TimeSeries.copy()
        exp_time_series[self.value_name] = np.exp((exp_time_series[self.value_name]))
        exp_time_series -= decrement_val
        return exp_time_series

    def standardization(self):
        """
        Transform by applying standardization to the data. For each value x in our time series we
        produce a transformation value y that is given by:
        y = (x - mean) / standard_deviation
        :return: Time series with applied standardization.
        """
        standardized_time_series = self.TimeSeries.copy()
        values = standardized_time_series.values
        values = values.reshape((len(values), 1))
        # train the standardization
        self.scaler = self.scaler.fit(values)
        # print('Mean: %f, StandardDeviation: %f' % (self.scaler.mean_, math.sqrt(
        # self.scaler.var_)))

        normalized = self.scaler.transform(values)
        standardized_time_series.iloc[:, 0] = normalized
        return standardized_time_series

    def invert_standardization(self, standardized_time_series):
        """
        Apply the inverse transformation on a standardized time series.
        :param standardized_time_series
        :return: Original time series.
        """
        inverse_time_series = standardized_time_series.copy()
        values = inverse_time_series.values
        values = values.reshape((len(values), 1))
        inversed = self.scaler.inverse_transform(values)
        inverse_time_series.iloc[:, 0] = inversed
        return inverse_time_series


def test_transformations(time_series, val_name):
    transformer = DataTransformation(time_series.copy(), val_name)
    sTest = StationarityTests()

    # Difference transformation
    time_series_diff = transformer.difference()
    plot_time_series(time_series_diff, 'Differenced time series', 'Diff values')
    sTest.ADF_Stationarity_Test(time_series_diff, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))
    # TODO: Find the best diff operator

    # Square root transformation
    # Check for square root pattern by plotting histogram
    plt.hist(time_series, bins=30)
    plt.show()
    sqrt_time_series = transformer.sqrt()
    print('Sqrt time series:\n', sqrt_time_series.head())
    plot_time_series(sqrt_time_series, 'Square root transformation', 'Square root values')
    sTest.ADF_Stationarity_Test(sqrt_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))

    # take diff on top of sqrt transformation
    sqrt_transform = DataTransformation(sqrt_time_series, val_name)
    sqrt_diff_time_series = sqrt_transform.difference(1)
    plot_time_series(sqrt_diff_time_series, 'Square root and Diff(1) transformation',
                     'Transformation values')
    sTest.ADF_Stationarity_Test(sqrt_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))

    # Power transformation
    pow_transformation = transformer.pow()
    print('Power time series:\n', pow_transformation.head())
    plot_time_series(pow_transformation, 'Power of 2 transformation', 'Power values')
    sTest.ADF_Stationarity_Test(pow_transformation, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))

    # Log transformation
    log_transformation = transformer.log(1)
    print('Log2 time series:\n', log_transformation.head())
    plot_time_series(log_transformation, 'Log transformation', 'Log values')
    sTest.ADF_Stationarity_Test(log_transformation, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))
    # take diff on top of log transformation
    log_transformation = DataTransformation(log_transformation, val_name)
    log_diff_time_series = log_transformation.difference(1)
    plot_time_series(log_diff_time_series, 'Log and Diff(1) transformation', 'Transformation '
                                                                             'values')
    sTest.ADF_Stationarity_Test(log_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))

    # Standardization transformation
    standardized_time_series = transformer.standardization()
    print('Standardized time series:\n', standardized_time_series.head())
    plot_time_series(standardized_time_series, 'Standardization transformation',
                     'Standardized values')

    sTest.ADF_Stationarity_Test(standardized_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))
    # take log on top of standardization transformation
    standardized_time_series = DataTransformation(standardized_time_series, val_name)
    standardized_log_time_series = standardized_time_series.log(1)
    plot_time_series(standardized_log_time_series, 'Standardization and Log transformation',
                     'Transformation values')
    sTest.ADF_Stationarity_Test(standardized_log_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))

    # take diff on top of standardization and log transformation
    standardized_log_time_series = DataTransformation(standardized_log_time_series, val_name)
    standardized_log_diff_time_series = standardized_log_time_series.difference(1)
    plot_time_series(standardized_log_diff_time_series, 'Standardization, Log and Diff(1) '
                                                        'transformation', 'Transformation values')

    sTest.ADF_Stationarity_Test(standardized_log_diff_time_series, printResults=True)
    print("Is the time series stationary? {0}".format(sTest.isStationary))

    # generate standardized time series and then apply inverse transformation to get back the
    # original time series
    transformer = DataTransformation(time_series.copy(), val_name)
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


def plot_correlation_plots(time_series, lags=20):
    """
    This function plots the ACF and PACF plots for a given time series.
    :param time_series: The time series to examine.
    :param lags: The number of lags to be printed for the correlation functions.
    """
    plot_acf(time_series, lags)
    plt.show()
    plot_pacf(time_series, lags)
    plt.show()

def generate_train_and_test_set(time_series, date_threshold, test_set_length):
    """
    This function splits the time series into a train and test set based on a date threshold.
    :param time_series: The time series to split.
    :param date_threshold: The date that marks the end of the train set. The train set will
    contain all values from the first value in the time series up until the date threshold value.
    :param test_set_length: The length of the test set, i.e., the number of values we want to
    predict. The resulting test set will be a time series starting from one day after the
    date_threshold and going on for test_set_length days.
    :return: A tuple containing (train_set, test_set)
    """


def run_ar_simulation(time_series, ar_order=1):
    """
    Train an auto regressive (AR) model and use it to predict future values.
    :param time_series: The time series to run the model on.
    :param ar_order: The order of the the AR regressor.
    """


def run_ma_simulation(time_series, ma_order=1):
    """
    Train an moving average (MA) model and use it to predict future values.
    :param time_series: The time series to run the model on.
    :param ma_order: The order of the the MA regressor.
    """


def run_arma_simulation(time_series, ar_order=1, ma_order=1):
    """
    Train an ARMA model and use it to predict future values.
    :param time_series: The time series to run the model on.
    :param ar_order: The order of the the AR part.
    :param ma_order: The order of the the MA part.
    """


def run_arima_simulation(time_series, ar_order=1, ma_order=1):
    """
    Train an ARIMA model and use it to predict future values.
    :param time_series: The time series to run the model on.
    :param ar_order: The order of the the AR part.
    :param ma_order: The order of the the MA part.
    """


if __name__ == "__main__":
    time_series = generate_time_series(city_code=5000)

    # run transformations
    test_transformations(time_series, 'daily_new_cases')







