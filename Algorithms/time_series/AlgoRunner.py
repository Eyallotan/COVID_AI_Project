import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn import metrics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta
from time import time
from DataTransformation import DataTransformation
from Preprocess import utils


class AlgoRunner:
    def __init__(self, time_series, train_end, test_end, original_time_series=None,
                 transformations=None, diff_params=None, log_exp_delta=None):
        """
        :param time_series: The time series for the runner.
        :param train_end: The date that marks the end of the train set.
        :param test_end: The end date of the test set.
        :param original_time_series: If the time_series provided was transformed, this optional
        parameter will hold the original time series. Providing this parameter signals that we
        want to get all results after inverting back to the original time series.
        :param transformations: If the time_series provided was transformed, this optional
        parameter will hold the transformations that were applied as a list of transformations.
        For example: ['difference', 'sqrt'].
        :param diff_params: Diff params for difference transformation (for transformed time series).
        This is a tuple containing (diff_lags, original_time_series_for_inverting_diff). This
        AlgoRunner supports only one difference operator in the transformations list.
        :param log_exp_delta: Delta value for log and exp transformations
        (for transformed time series).
        Note: Date input should be in the format (year, month, day)
        """
        self.time_series = time_series
        self.train_end = train_end
        self.test_end = test_end
        self.train_data, self.test_data = self.generate_train_and_test_set()
        self.original_time_series = original_time_series
        self.transformations = transformations
        self.diff_params = diff_params
        self.log_exp_delta = log_exp_delta
        self.predictions = None

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

    def run_ar_regressor(self, ar_order=1, use_rolling_forecast=False, print_results=True):
        """
        Train an auto regressive (AR) model and use it to predict future values.
        :param ar_order: The order of the the AR regressor.
        :param use_rolling_forecast: Whether or not to use a rolling forecast for predictions.
        :param print_results: Whether or not to print result plots.
        """
        model_order = (ar_order, 0, 0)
        self.predictions = self.fit_and_predict(model_order, use_rolling_forecast)
        if print_results is True:
            self.print_results('AR')

    def run_ma_regressor(self, ma_order=1, use_rolling_forecast=False, print_results=True):
        """
        Train an moving average (MA) model and use it to predict future values.
        :param ma_order: The order of the the MA regressor.
        :param use_rolling_forecast: Whether or not to use a rolling forecast for predictions.
        :param print_results: Whether or not to print result plots and metrics.
        """
        model_order = (0, 0, ma_order)
        self.predictions = self.fit_and_predict(model_order, use_rolling_forecast)
        if print_results is True:
            self.print_results('MA')

    def run_arma_regressor(self, ar_order=1, ma_order=1, use_rolling_forecast=False,
                           print_results=True):
        """
        Train an ARMA model and use it to predict future values.
        :param ar_order: The order of the the AR part.
        :param ma_order: The order of the the MA part.
        :param use_rolling_forecast: Whether or not to use a rolling forecast for predictions.
        :param print_results: Whether or not to print result plots.
        """
        model_order = (ar_order, 0, ma_order)
        self.predictions = self.fit_and_predict(model_order, use_rolling_forecast)
        if print_results is True:
            self.print_results('ARMA')

    def run_arima_regressor(self, ar_order=1, diff_order=1, ma_order=1,
                            use_rolling_forecast=False, print_results=True):
        """
        Train an ARIMA model and use it to predict future values.
        :param ar_order: The order of the the AR part.
        :param diff_order: The order of the integrated part (the diff param).
        :param ma_order: The order of the the MA part.
        :param use_rolling_forecast: Whether or not to use a rolling forecast for predictions.
        :param print_results: Whether or not to print result plots.
        """
        model_order = (ar_order, diff_order, ma_order)
        self.predictions = self.fit_and_predict(model_order, use_rolling_forecast)
        if print_results is True:
            self.print_results('ARIMA')

    def print_results_and_accuracy_intrn(self, predictions, model_name):
        """
        Print the model accuracy using various metrics and print plots. The plots that are
        generated are the residuals plot and the data vs. predictions plot.
        :param predictions: The model predictions.
        :param model_name: The model name.
        """
        residuals = self.test_data - predictions
        # plot residuals
        plt.figure(figsize=(10, 4))
        plt.plot(residuals)
        plt.axhline(0, linestyle='--', color='k')
        plt.title(f'Residuals from {model_name} Model', fontsize=20)
        plt.ylabel('Error', fontsize=16)
        plt.show()
        # plot data vs. predictions
        plt.figure(figsize=(10, 4))
        plt.plot(self.time_series if self.original_time_series is None else self.original_time_series)
        plt.plot(predictions)
        plt.legend(('Data', 'Predictions'), fontsize=16)
        plt.title('Data vs. Predictions', fontsize=20)
        plt.ylabel('Values', fontsize=16)
        plt.show()
        utils.print_result_metrics(self.test_data, predictions)

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
            # with model.fix_params({'ma.L6': 0.0}):
            model_fit = model.fit()
            # print model summary
            print(model_fit.summary())
            # forecast one day ahead
            next_prediction = model_fit.forecast(1)
            prediction_rolling = prediction_rolling.append(next_prediction)
        return prediction_rolling

    def fit_and_predict(self, model_order, use_rolling_forecast):
        """
        An execution method for the ARIMA model.
        :param model_order: The ARIMA model orders. This argument is in the following format:
        (ar_order, diff_order, ma_order)
        :param use_rolling_forecast: Whether or not to use a rolling forecast for predictions.
        :return: The model predictions (converted to dataframe format)
        """
        if use_rolling_forecast:
            predictions = self.rolling_forecast(model_order)
        else:
            model = ARIMA(self.train_data, order=model_order)
            model_fit = self.fit_model(model)
            predictions = self.predict(model_fit)
        # convert prediction series to dataframe
        predictions = predictions.to_frame(self.time_series.columns[0])
        return predictions

    def print_results(self, model_name):
        """
        A wrapper for printing the model results. This method will convert the predictions
        according to data transformations if needed and then call the actual print result method.
        :param model_name: The model name for plotting.
        """
        assert self.predictions is not None
        if self.transformations is not None:
            self.predictions, self.test_data = self.invert_regressor_results(self.predictions)
        self.print_results_and_accuracy_intrn(self.predictions, model_name)

    def invert_regressor_results(self, predictions):
        """
        Invert the regressor predictions and translate the results to correlate with the original
        time series values. All of the transformations that were done on the original time series
        will be inverted by applying the inverse transformation in backwards order.
        :param predictions: The predictions from the regressor.
        :return: A tuple containing the inverted predictions and inverted test data.
        """
        assert self.transformations is not None
        pred_transformer = DataTransformation(predictions)
        test_transformer = DataTransformation(self.test_data)
        transformations = self.transformations
        predictions_inverted, test_inverted = predictions, self.test_data
        for transformation in reversed(transformations):
            if transformation == 'difference':
                # A little abuse of notation, forgive me (or fix it..)
                predictions_inverted = pred_transformer.invert_difference(predictions_inverted,
                                                                          original_time_series=
                                                                          self.diff_params[1],
                                                                          lags=self.diff_params[0])
                test_inverted = test_transformer.invert_difference(test_inverted,
                                                                   original_time_series=
                                                                   self.diff_params[1],
                                                                   lags=self.diff_params[0])
            if transformation == 'sqrt':
                predictions_inverted = pred_transformer.pow()
                test_inverted = test_transformer.pow()
            if transformation == 'pow':
                predictions_inverted = pred_transformer.sqrt()
                test_inverted = test_transformer.sqrt()
            if transformation == 'log':
                predictions_inverted = pred_transformer.exp(decrement_val=self.log_exp_delta)
                test_inverted = test_transformer.exp(decrement_val=self.log_exp_delta)
            if transformation == 'exp':
                predictions_inverted = pred_transformer.log(increment_val=self.log_exp_delta)
                test_inverted = test_transformer.log(increment_val=self.log_exp_delta)
            if transformation == 'standardization':
                predictions_inverted = pred_transformer.invert_standardization()
                test_inverted = test_transformer.invert_standardization()
        return predictions_inverted, test_inverted
