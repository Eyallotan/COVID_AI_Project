import pandas as pd
from datetime import datetime
from Algorithms.DT import DecisionTreePartB


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

    best_regressor = DecisionTreePartB.get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)

    print(f'knn_on_small_cities accuracy: {score}, population={population}')


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

    best_regressor = DecisionTreePartB.get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)

    print(f'knn_on_big_new_cases accuracy: {score}, new_cases {new_cases}')


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

    best_regressor = DecisionTreePartB.get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)

    print(f'knn_on_big_new_cases accuracy: {score}, colour {colour}')


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

    best_regressor = DecisionTreePartB.get_best_decision_tree_regressor()
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

    best_regressor = DecisionTreePartB.get_best_decision_tree_regressor()
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

    best_regressor = DecisionTreePartB.get_best_decision_tree_regressor()
    score = best_regressor.score(X_test, Y_test)

    print(f'decision_tree_on_second_veccinated_percentage accuracy: {score}, min_percentage {min_percentage}, max_percentage {max_percentage}')


def run_sub_test_sets_experiments():
    run_decision_tree_on_small_cities(10000)
    run_decision_tree_on_small_cities(50000)
    run_decision_tree_on_small_cities(100000)
    run_decision_tree_on_small_new_cases(10)
    run_decision_tree_on_small_new_cases(30)
    run_decision_tree_on_small_new_cases(400)
    run_decision_tree_on_colour(0)
    run_decision_tree_on_colour(1)
    run_decision_tree_on_colour(2)
    run_decision_tree_on_colour(3)
    run_decision_tree_on_first_veccinated_percentage(0, 0.25)
    run_decision_tree_on_first_veccinated_percentage(0.25, 0.50)
    run_decision_tree_on_first_veccinated_percentage(0.50, 0.75)
    run_decision_tree_on_first_veccinated_percentage(0.75, 1)
    run_decision_tree_on_second_veccinated_percentage(0, 0.25)
    run_decision_tree_on_second_veccinated_percentage(0.25, 0.50)
    run_decision_tree_on_second_veccinated_percentage(0.50, 0.75)
    run_decision_tree_on_second_veccinated_percentage(0.75, 1)

    start_date = datetime(2021, 1, 20)
    end_date = datetime(2021, 3, 20)
    run_decision_tree_on_dates(start_date, end_date)

    start_date = datetime(2021, 3, 20)
    end_date = datetime(2021, 5, 20)
    run_decision_tree_on_dates(start_date, end_date)

    start_date = datetime(2021, 5, 20)
    end_date = datetime(2021, 7, 20)
    run_decision_tree_on_dates(start_date, end_date)

    start_date = datetime(2021, 7, 20)
    end_date = datetime(2021, 9, 11)
    run_decision_tree_on_dates(start_date, end_date)


if __name__ == "__main__":
    run_sub_test_sets_experiments()







