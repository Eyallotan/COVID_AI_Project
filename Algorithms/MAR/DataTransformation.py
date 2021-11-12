import numpy as np
from sklearn.preprocessing import StandardScaler


class DataTransformation:
    """
    Class for performing transformations on time series data sets.
    For each transformation operator there is a inverse operator that can transform the data
    back to it's original form.
    """
    def __init__(self, data, value_name):
        self.time_series = data
        self.value_name = value_name
        self.scaler = StandardScaler()

    def difference(self, interval=1):
        """
        Transform to a difference time series with a certain interval.
        :param interval: The difference operator.
        :return: Time series with applied diff.
        """
        assert interval > 0
        return self.time_series.diff(interval).dropna()

    def sqrt(self):
        """
        Transform by taking a square root of all values.
        :return: Time series with applied square root transformation.
        """
        sqrt_time_series = self.time_series.copy()
        sqrt_time_series[self.value_name] = np.sqrt(sqrt_time_series[self.value_name])
        return sqrt_time_series

    def pow(self):
        """
        Transform by applying power of 2 to all values.
        :return: Time series with applied pow transformation.
        """
        pow_time_series = self.time_series.copy()
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
        log_time_series = self.time_series.copy()
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
        exp_time_series = self.time_series.copy()
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
        standardized_time_series = self.time_series.copy()
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

