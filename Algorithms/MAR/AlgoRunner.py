import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from time import time


class AlgoRunner:
    def __init__(self, data, train_end, test_end):
        """
        :param data: The time series for the runner.
        :param train_end: The date that marks the end of the train set.
        :param test_end: The end date of the test set.
        Note: Date input should be in the format (year, month, day)
        """
        self.time_series = data
        self.train_end = train_end
        self.test_end = test_end
        self.train_data, self.test_data = self.generate_train_and_test_set()

    @staticmethod
    def fit_model(model, print_summary=True):
        """
        Fit the given model.
        :param model: The model to fit.
        :param print_summary: Whether or not to print the model fit summary.
        :return: The fitted model object.
        """
        # fit the model
        start = time()
        model_fit = model.fit()
        end = time()
        if print_summary:
            print('Model Fitting Time:', end - start)
            # summary of the model
            print(model_fit.summary())
        return model_fit

    def predict(self, model_fit, model_name, print_plots=True):
        """
        Predict the test data values based on the trained model object.
        :param model_fit: Trained model object
        :param model_name: The model name
        :param print_plots: Whether to print the residuals plot and data vs. predictions plot.
        :return: The predictions series.
        """
        predictions = model_fit.forecast(len(self.test_data))
        predictions = pd.Series(predictions, index=self.test_data.index)
        residuals = self.test_data.iloc[:, 0] - predictions
        if print_plots:
            # plot residuals
            plt.figure(figsize=(10, 4))
            plt.plot(residuals)
            plt.axhline(0, linestyle='--', color='k')
            plt.title(f'Residuals from {model_name} Model', fontsize=20)
            plt.ylabel('Error', fontsize=16)
            plt.show()
            # plot data vs. predictions
            plt.figure(figsize=(10, 4))
            plt.plot(self.test_data)
            plt.plot(predictions)
            plt.legend(('Data', 'Predictions'), fontsize=16)
            plt.title('Data vs. Predictions', fontsize=20)
            plt.ylabel('Values', fontsize=16)
            plt.show()

        return predictions

    def plot_correlation_plots(self, number_of_lags=20):
        """
        This function plots the ACF and PACF plots for a given time series.
        :param number_of_lags: The number of lags to be printed for the correlation functions.
        """
        plot_acf(self.time_series, lags=number_of_lags)
        plt.show()
        plot_pacf(self.time_series, lags=number_of_lags)
        plt.show()

    def generate_train_and_test_set(self):
        """
        This function splits the time series into a train and test set based on a date threshold.
        The train set will contain all values from the first value in the time series up until
        the train_end value. The resulting test set will be a time series
        starting from one day after train_end until test_end.
        :return: A tuple containing (train_set, test_set)
        """
        train_data = self.time_series[:self.train_end]
        test_data = self.time_series[self.train_end + timedelta(days=1):self.test_end]
        return train_data, test_data

    def run_ar_regressor(self, ar_order=1):
        """
        Train an auto regressive (AR) model and use it to predict future values.
        :param ar_order: The order of the the AR regressor.
        """
        # define the model
        model = ARIMA(self.train_data, order=(ar_order, 0, 0))
        model_fit = self.fit_model(model)
        self.predict(model_fit, 'AR')

    def run_ma_regressor(self, ma_order=1):
        """
        Train an moving average (MA) model and use it to predict future values.
        :param ma_order: The order of the the MA regressor.
        """
        # define the model
        model = ARIMA(self.train_data, order=(0, 0, ma_order))
        model_fit = self.fit_model(model)
        self.predict(model_fit, 'MA')

    def run_arma_regressor(self, ar_order=1, ma_order=1):
        """
        Train an ARMA model and use it to predict future values.
        :param ar_order: The order of the the AR part.
        :param ma_order: The order of the the MA part.
        """
        # define the model
        model = ARIMA(self.train_data, order=(ar_order, 0, ma_order))
        model_fit = self.fit_model(model)
        self.predict(model_fit, 'ARMA')

    def run_arima_regressor(self, ar_order=1, diff_order=1, ma_order=1):
        """
        Train an ARIMA model and use it to predict future values.
        :param ar_order: The order of the the AR part.
        :param diff_order: The order of the integrated part (the diff param).
        :param ma_order: The order of the the MA part.
        """
        model = ARIMA(self.train_data, order=(ar_order, diff_order, ma_order))
        model_fit = self.fit_model(model)
        self.predict(model_fit, 'ARIMA')
