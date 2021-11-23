import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from time import time
from DataTransformation import DataTransformation


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
        start = time()
        model_fit = model.fit()
        end = time()
        if print_summary:
            print('Model Fitting Time:', end - start)
            # summary of the model
            print(model_fit.summary())
        return model_fit

    def predict(self, model_fit):
        """
        Predict the test data values based on the trained model object.
        :param model_fit: Trained model object
        :return The model's predictions.
        """
        predictions = model_fit.forecast(len(self.test_data))
        predictions = pd.Series(predictions, index=self.test_data.index)
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

    def run_ar_regressor(self, ar_order=1, use_rolling_forecast=False, print_plots=True):
        """
        Train an auto regressive (AR) model and use it to predict future values.
        :param ar_order: The order of the the AR regressor.
        :param use_rolling_forecast: Whether or not to use a rolling forecast for predictions.
        :param print_plots: Whether or not to print result plots.
        """
        model_order = (ar_order, 0, 0)
        if use_rolling_forecast:
            predictions = self.rolling_forecast(model_order)
        else:
            model = ARIMA(self.train_data, order=model_order)
            model_fit = self.fit_model(model)
            predictions = self.predict(model_fit)
        self.print_results_and_accuracy(predictions, 'AR', print_plots)

    def run_ma_regressor(self, ma_order=1, use_rolling_forecast=False, print_plots=True):
        """
        Train an moving average (MA) model and use it to predict future values.
        :param ma_order: The order of the the MA regressor.
        :param use_rolling_forecast: Whether or not to use a rolling forecast for predictions.
        :param print_plots: Whether or not to print result plots.
        """
        model_order = (0, 0, ma_order)
        if use_rolling_forecast:
            predictions = self.rolling_forecast(model_order)
        else:
            model = ARIMA(self.train_data, order=model_order)
            model_fit = self.fit_model(model)
            predictions = self.predict(model_fit)
        self.print_results_and_accuracy(predictions, 'MA', print_plots)

    def run_arma_regressor(self, ar_order=1, ma_order=1, use_rolling_forecast=False,
                           print_plots=True):
        """
        Train an ARMA model and use it to predict future values.
        :param ar_order: The order of the the AR part.
        :param ma_order: The order of the the MA part.
        :param use_rolling_forecast: Whether or not to use a rolling forecast for predictions.
        :param print_plots: Whether or not to print result plots.
        """
        model_order = (ar_order, 0, ma_order)
        if use_rolling_forecast:
            predictions = self.rolling_forecast(model_order)
        else:
            model = ARIMA(self.train_data, order=model_order)
            model_fit = self.fit_model(model)
            predictions = self.predict(model_fit)
        self.print_results_and_accuracy(predictions, 'ARMA', print_plots)

    def run_arima_regressor(self, ar_order=1, diff_order=1, ma_order=1,
                            use_rolling_forecast=False, print_plots=True):
        """
        Train an ARIMA model and use it to predict future values.
        :param ar_order: The order of the the AR part.
        :param diff_order: The order of the integrated part (the diff param).
        :param ma_order: The order of the the MA part.
        :param use_rolling_forecast: Whether or not to use a rolling forecast for predictions.
        :param print_plots: Whether or not to print result plots.
        """
        model_order = (ar_order, diff_order, ma_order)
        if use_rolling_forecast:
            predictions = self.rolling_forecast(model_order)
        else:
            model = ARIMA(self.train_data, order=model_order)
            model_fit = self.fit_model(model)
            predictions = self.predict(model_fit)
        self.print_results_and_accuracy(predictions, 'ARIMA', print_plots)

    def print_results_and_accuracy(self, predictions, model_name, print_plots=True):
        """
        Print the model accuracy using various metrics.
        :param predictions: The model predictions.
        :param model_name: The model name.
        :param print_plots: Whether or not to print the residual plot and the data vs.
        predictions plot.
        """
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
        print('###----Metrics for model accuracy---###')
        MAPE = mean_absolute_percentage_error(self.test_data, predictions)
        MAE = mean_absolute_error(self.test_data, predictions)
        MSE = mean_squared_error(self.test_data, predictions)
        print('Mean Absolute Percentage Error: ', MAPE)
        print('Test Data Mean: ', np.mean(self.test_data.iloc[:, 0]))
        print('Mean Absolute Error', MAE)
        print('Mean Squared Error:', MSE)
        print('Root Mean Squared Error:', np.sqrt(MSE))

    def rolling_forecast(self, model_orders):
        """
        Run the selected model using a rolling forecast. The rolling forecast will predict one
        day at a time, and then train again with all data up to the day that was just forecasted
        in order to predict the next day, and so on.
        :param model_orders: The model order parameters (ar_order, diff_order, ma_order).
        :return: The predictions that were generated using the rolling forecast.
        """
        prediction_rolling = pd.Series()
        for end_date in self.test_data.index:
            train_data = self.time_series[:end_date - timedelta(days=1)]
            model = ARIMA(train_data, order=model_orders)
            model_fit = model.fit()
            # forecast one day ahead
            pred = model_fit.forecast(1)
            prediction_rolling.loc[end_date] = pred
        return prediction_rolling
