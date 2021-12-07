# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import KFold
# from Algorithms.DTTS import DecisionTreeAuxiliaries
# from Algorithms.DTTS import DecisionTreePrinting


#######################################################################################################################
#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################


# forecast monthly births with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
import pandas as pd
from sklearn import metrics
from datetime import datetime


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(time_series_corona_df, n_input=1, n_output=1, drop_nan=True):
    n_vars = 1 if type(time_series_corona_df) is list else time_series_corona_df.shape[1]
    df = DataFrame(time_series_corona_df)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_input, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_output):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if drop_nan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]


# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    # transform list into array
    train = asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = RandomForestRegressor(n_estimators=10, random_state=1)
    model.fit(trainX, trainy)
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = random_forest_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    # estimate prediction error
    print(test[:, -2])
    mae = mean_absolute_error(test[:, -2], predictions)
    return mae, test[:, -1], predictions


def run_part_A():
    # load the dataset
    corona_df = pd.read_csv('../../Preprocess/corona_df.csv')
    corona_df['Date'] = corona_df['Date'].str.replace('-', '').astype(float)
    today_verified_cases_df = DataFrame(corona_df[[col for col in corona_df if col == 'today_verified_cases' or col == 'City_Name' or col == 'City_Code' or col == 'Date']])
    corona_df = corona_df.drop(['today_verified_cases'], axis=1)

    corona_df = pd.merge(corona_df, today_verified_cases_df, how="inner", on=["City_Name", "City_Code",
                                                                       "Date"])

    features = [col for col in corona_df]
    for feature in features:
        if feature == 'today_verified_cases' or feature == 'Date' or feature == 'City_Code' or feature == 'verified_cases_7_days_ago':
            continue
        corona_df = corona_df.drop([feature], axis=1)

    values = corona_df.values
    # transform the time series data into supervised learning
    time_series_corona_df = series_to_supervised(values, n_input=2)
    # evaluate
    mae, y, yhat = walk_forward_validation(time_series_corona_df, 5)
    print('MAE: %.3f' % mae)
    # plot expected vs predicted
    pyplot.plot(y, label='Expected')
    pyplot.plot(yhat, label='Predicted')
    pyplot.legend()
    pyplot.show()


#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################
#######################################################################################################################


if __name__ == "__main__":
    run_part_A()








