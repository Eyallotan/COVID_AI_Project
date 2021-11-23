import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from Algorithms.DecisionTreeAlgo import DecisionTreeAuxiliaries
from sklearn import metrics


def get_best_decision_tree_regressor():
    RF_best_min_samples_leaf = 1
    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']
    train_df, test_df = DecisionTreeAuxiliaries.get_train_and_test_df()

    for col in train_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            train_df = train_df.drop([col], axis=1)

    X_train, Y_train, X_test, Y_test = DecisionTreeAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=RF_best_min_samples_leaf)
    RF_regressor.fit(X_train, Y_train)

    return RF_regressor


def run_decision_tree_on_small_cities(population):
    test_df = pd.read_csv('../../Preprocess/test_df.csv')

    population_df = pd.read_csv('../../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])
    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    test_df = test_df[test_df['population'] <= population]
    test_df.drop(['population'], axis=1, inplace=True)

    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    for col in test_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            test_df = test_df.drop([col], axis=1)

    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    best_regressor = get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)
    Y_pred = best_regressor.predict(X_test)

    Avg_Y_test_arr = []
    for i in range(len(Y_test)):
        Avg_Y_test_arr.append(np.mean(Y_test))

    print(f'City Population Score: population <= {population} ===> {score} ')
    print(f'City Population SSres: population <= {population} ===> {metrics.mean_squared_error(Y_test, Y_pred)}')
    print(f'City Population SStotal: population <= {population} ===> {metrics.mean_squared_error(Y_test, Avg_Y_test_arr)}')
    print(f'City Population MAE: population <= {population} ===> {metrics.mean_absolute_error(Y_test, Y_pred)}')
    # print(f'City Population Y_test Avg: {np.mean(Y_test)}')
    # print(f'City Population Y_pred Avg: {np.mean(Y_pred)}')
    print(f'City Population Y_pred Avg - Y_test Avg (abs): population <= {population} ===> {abs(np.mean(Y_pred) - np.mean(Y_test))}')


def run_decision_tree_on_small_new_cases(new_cases):
    test_df = pd.read_csv('../../Preprocess/test_df.csv')
    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    test_df = test_df[test_df['today_verified_cases'] <= new_cases]

    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    for col in test_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            test_df = test_df.drop([col], axis=1)

    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    best_regressor = get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)
    Y_pred = best_regressor.predict(X_test)

    Avg_Y_test_arr = []
    for i in range(len(Y_test)):
        Avg_Y_test_arr.append(np.mean(Y_test))

    print(f'Today Verified Cases Score: new_cases <= {new_cases} ===> {score} ')
    print(f'SSres: {metrics.mean_squared_error(Y_test, Y_pred)}')
    print(f'SStotal: {metrics.mean_squared_error(Y_test, Avg_Y_test_arr)}')
    print(f'Today Verified Cases MAE: {metrics.mean_absolute_error(Y_test, Y_pred)}')
    # print(f'Y_test Avg: {np.mean(Y_test)}')
    # print(f'Y_pred Avg: {np.mean(Y_pred)}')
    print(f'Today Verified Cases Y_pred Avg - Y_test Avg (abs): {abs(np.mean(Y_pred) - np.mean(Y_test))}')


def run_decision_tree_on_colour(colour):
    test_df = pd.read_csv('../../Preprocess/test_df.csv')
    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    test_df = test_df[test_df['colour'] == colour]

    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    for col in test_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            test_df = test_df.drop([col], axis=1)

    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    best_regressor = get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)
    Y_pred = best_regressor.predict(X_test)

    Avg_Y_test_arr = []
    for i in range(len(Y_test)):
        Avg_Y_test_arr.append(np.mean(Y_test))

    print(f'Colour Score: colour == {colour} ===> {score} ')
    print(f'Colour SSres: colour == {colour} ===> {metrics.mean_squared_error(Y_test, Y_pred)}')
    print(f'Colour SStotal: colour == {colour} ===> {metrics.mean_squared_error(Y_test, Avg_Y_test_arr)}')
    print(f'Colour MAE: colour == {colour} ===> {metrics.mean_absolute_error(Y_test, Y_pred)}')
    # print(f'Colour Y_test Avg: {np.mean(Y_test)}')
    # print(f'Colour Y_pred Avg: {np.mean(Y_pred)}')
    print(f'Colour Y_pred Avg - Y_test Avg (abs): colour == {colour} ===> {abs(np.mean(Y_pred) - np.mean(Y_test))}')

    # results_df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred, 'Diff': abs(Y_test-Y_pred)})
    # if colour == 0:
    #     results_df.to_csv('color_0_results.csv')
    # if colour == 3:
    #     results_df.to_csv('color_3_results.csv')
    #
    # print(np.mean(abs(Y_test-Y_pred)))


def run_decision_tree_on_dates(start_date, end_date):
    test_df = pd.read_csv('../../Preprocess/test_df.csv')

    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df = test_df[(test_df['Date'] >= start_date) & (test_df['Date'] <= end_date)]

    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    for col in test_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            test_df = test_df.drop([col], axis=1)

    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    best_regressor = get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)

    print(f'decision_tree_on_dates accuracy: {score}, start_date {start_date}, end_date {end_date}')


def run_decision_tree_on_first_veccinated_percentage(min_percentage, max_percentage):
    test_df = pd.read_csv('../../Preprocess/test_df.csv')

    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df = test_df[(test_df['vaccinated_dose_1_total'] >= min_percentage) & (test_df['vaccinated_dose_1_total'] <= max_percentage)]

    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    for col in test_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            test_df = test_df.drop([col], axis=1)

    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    best_regressor = get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)

    print(f'decision_tree_on_first_veccinated_percentage accuracy: {score}, min_percentage {min_percentage}, max_percentage {max_percentage}')


def run_decision_tree_on_second_veccinated_percentage(min_percentage, max_percentage):
    test_df = pd.read_csv('../../Preprocess/test_df.csv')

    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df = test_df[(test_df['vaccinated_dose_2_total'] >= min_percentage) & (test_df['vaccinated_dose_2_total'] <= max_percentage)]

    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    for col in test_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            test_df = test_df.drop([col], axis=1)

    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    best_regressor = get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)

    print(f'decision_tree_on_second_veccinated_percentage accuracy: {score}, min_percentage {min_percentage}, max_percentage {max_percentage}')


def run_decision_tree_on_yesterday_verified_cases(verified_cases):
    test_df = pd.read_csv('../../Preprocess/test_df.csv')

    test_df['Date'] = pd.to_datetime(test_df['Date'])
    if verified_cases == 0:
        test_df = test_df[(test_df['verified_cases_1_days_ago'] == verified_cases)]
    if verified_cases != 0:
        test_df = test_df[(test_df['verified_cases_1_days_ago'] > 0)]

    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    for col in test_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            test_df = test_df.drop([col], axis=1)

    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    best_regressor = get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)
    Y_pred = best_regressor.predict(X_test)

    results_df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred, 'Diff': abs(Y_test-Y_pred)})
    if verified_cases == 0:
        results_df.to_csv('verified_cases_equals_0_results.csv')
        print(f'verified_cases_bigger_than_0 accuracy: {score}')
    if verified_cases != 0:
        results_df.to_csv('verified_cases_bigger_than_0_results.csv')
        print(f'verified_cases_equals_0 accuracy: {score}')

    print(f'MAE: {metrics.mean_absolute_error(Y_test, Y_pred)}')
    print(f'Y_test: {np.mean(Y_test)}')
    print(f'Y_pred: {np.mean(Y_pred)}')


def run_sub_test_sets_experiments():
    run_decision_tree_on_small_cities(10000)
    run_decision_tree_on_small_cities(50000)
    run_decision_tree_on_small_cities(100000)
    # run_decision_tree_on_small_new_cases(10)
    # run_decision_tree_on_small_new_cases(30)
    # run_decision_tree_on_small_new_cases(400)
    run_decision_tree_on_colour(0)
    run_decision_tree_on_colour(1)
    run_decision_tree_on_colour(2)
    run_decision_tree_on_colour(3)
    # run_decision_tree_on_first_veccinated_percentage(0, 0.25)
    # run_decision_tree_on_first_veccinated_percentage(0.25, 0.50)
    # run_decision_tree_on_first_veccinated_percentage(0.50, 0.75)
    # run_decision_tree_on_first_veccinated_percentage(0.75, 1)
    # run_decision_tree_on_second_veccinated_percentage(0, 0.25)
    # run_decision_tree_on_second_veccinated_percentage(0.25, 0.50)
    # run_decision_tree_on_second_veccinated_percentage(0.50, 0.75)
    # run_decision_tree_on_second_veccinated_percentage(0.75, 1)
    #
    # run_decision_tree_on_yesterday_verified_cases(0)
    # run_decision_tree_on_yesterday_verified_cases(1)

    # start_date = datetime(2021, 1, 20)
    # end_date = datetime(2021, 3, 20)
    # run_decision_tree_on_dates(start_date, end_date)
    #
    # start_date = datetime(2021, 3, 20)
    # end_date = datetime(2021, 5, 20)
    # run_decision_tree_on_dates(start_date, end_date)
    #
    # start_date = datetime(2021, 5, 20)
    # end_date = datetime(2021, 7, 20)
    # run_decision_tree_on_dates(start_date, end_date)
    #
    # start_date = datetime(2021, 7, 20)
    # end_date = datetime(2021, 9, 11)
    # run_decision_tree_on_dates(start_date, end_date)


if __name__ == "__main__":
    run_sub_test_sets_experiments()







