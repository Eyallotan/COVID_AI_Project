import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from Algorithms.DTTS import DecisionTreePartB
from Algorithms.DTTS import DecisionTreePrinting


#######################################################################################################################
#######################################################################################################################
################################################### PART C ############################################################
#######################################################################################################################


def run_experiment(experiment_type, population, colour):
    test_df = pd.read_csv('../../Preprocess/test_df.csv')

    population_df = pd.read_csv('../../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])
    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    if experiment_type == 'City Population':
        test_df = test_df[test_df['population'] <= population]
    if experiment_type == 'Colour':
        test_df = test_df[test_df['colour'] == colour]

    test_df.drop(['population'], axis=1, inplace=True)

    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    for col in test_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            test_df = test_df.drop([col], axis=1)

    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    best_regressor = DecisionTreePartB.get_best_decision_tree_regressor()
    r2_score = best_regressor.score(X_test, Y_test)
    Y_pred = best_regressor.predict(X_test)
    MAE_score = metrics.mean_absolute_error(Y_test, Y_pred)

    Avg_Y_test_arr = []
    for i in range(len(Y_test)):
        Avg_Y_test_arr.append(np.mean(Y_test))

    if experiment_type == 'City Population':
        print(f'City Population R^2 Score: population <= {population} ===> {r2_score} ')
        print(f'City Population MAE Score: population <= {population} ===> {MAE_score}')
        print(f'City Population SSres: population <= {population} ===> {metrics.mean_squared_error(Y_test, Y_pred)}')
        print(f'City Population SStotal: population <= {population} ===> {metrics.mean_squared_error(Y_test, Avg_Y_test_arr)}')
    if experiment_type == 'Colour':
        print(f'Colour R^2 Score: colour == {colour} ===> {r2_score} ')
        print(f'Colour MAE Score: colour == {colour} ===> {MAE_score}')
        print(f'Colour SSres: colour == {colour} ===> {metrics.mean_squared_error(Y_test, Y_pred)}')
        print(f'Colour SStotal: colour == {colour} ===> {metrics.mean_squared_error(Y_test, Avg_Y_test_arr)}')

    return r2_score, MAE_score


#######################################################################################################################
################################################### PART C ############################################################
#######################################################################################################################
#######################################################################################################################


def run_part_C():
    city_population_r2_results = []
    city_population_MAE_results = []
    colour_r2_results = []
    colour_MAE_results = []
    i = 50000
    while i <= 100000:
        r2_score, MAE_score = run_experiment('City Population', population=i, colour=None)
        city_population_r2_results.append(r2_score)
        city_population_MAE_results.append(MAE_score)
        i += 10000

    i = 0
    while i <= 3:
        r2_score, MAE_score = run_experiment('Colour', population=None, colour=i)
        colour_r2_results.append(r2_score)
        colour_MAE_results.append(MAE_score)
        i += 1

    DecisionTreePrinting.print_r2_score_vs_MAE_score('R^2', 'City Population', city_population_r2_results)
    DecisionTreePrinting.print_r2_score_vs_MAE_score('R^2', 'Colour', colour_r2_results)
    DecisionTreePrinting.print_r2_score_vs_MAE_score('MAE', 'City Population', city_population_MAE_results)
    DecisionTreePrinting.print_r2_score_vs_MAE_score('MAE', 'Colour', colour_MAE_results)


if __name__ == "__main__":
    run_part_C()







