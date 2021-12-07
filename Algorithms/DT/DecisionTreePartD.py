import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from Algorithms.DT import DecisionTreePartB
from Algorithms.DT import DecisionTreeAuxiliaries
from Algorithms.DT import DecisionTreePrinting
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold


#######################################################################################################################
#######################################################################################################################
################################################### PART D ############################################################
#######################################################################################################################

def print_train_set(train_set):
    plt.plot([index for index in range(len(train_set[train_set['City_Code'] == 5000]['today_verified_cases']))], train_set[train_set['City_Code'] == 5000]['today_verified_cases'].values, label="y_true")
    plt.axhline(train_set[train_set['City_Code'] == 5000]['today_verified_cases'].values.max(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(train_set[train_set['City_Code'] == 5000]['today_verified_cases'].values.min(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('israel new cases')
    plt.show()

def bin_data_set():
    # train_df, test_df = DecisionTreeAuxiliaries.get_train_and_test_df_part_D()
    # city_codes = train_df['City_Code'].values
    # # city_codes = np.sort(city_codes)
    # city_codes = list(dict.fromkeys(city_codes))
    # new_train_df = train_df
    # new_test_df = test_df
    # new_train_df = train_df[train_df['City_Code'] == city_codes[0]]
    for bin_size in range(1, 11):
        train_df, test_df = DecisionTreeAuxiliaries.get_train_and_test_df_part_D()
        city_codes = train_df['City_Code'].values
        # city_codes = np.sort(city_codes)
        city_codes = list(dict.fromkeys(city_codes))
        new_train_df = train_df
        for i in range(len(city_codes)):
            new_train_df.reset_index(drop=True, inplace=True)
            new_train_df = new_train_df[new_train_df['City_Code'] != city_codes[i]]
            new_train_df.reset_index(drop=True, inplace=True)
            temp_train_df = train_df[train_df['City_Code'] == city_codes[i]]
            temp_train_df = temp_train_df.sort_values(ascending=False, by=['today_verified_cases'])
            temp_train_df.reset_index(drop=True, inplace=True)
            for j in range(0, len(temp_train_df), bin_size):
                mean = 0
                for k in range(bin_size):
                    if (j + k) >= len(temp_train_df):
                        break
                    mean += temp_train_df['today_verified_cases'][j + k]
                mean = mean / bin_size
                if mean == 0:
                    break
                for k in range(bin_size):
                    if (j + k) >= len(temp_train_df):
                        break
                    temp_train_df.loc[j + k, 'today_verified_cases'] = mean
                    # instead
                    # temp_train_df['today_verified_cases'][j + k] = mean

            new_train_df = new_train_df.append(temp_train_df)

        new_train_df.reset_index(drop=True, inplace=True)
        #
        # new_train_df = new_train_df.sort_values(ascending=True, by=['City_Code', 'Date'])
        # new_train_df = new_train_df.drop(['Date'], axis=1)
        new_train_df_verified_cases = new_train_df[[col for col in new_train_df if col == 'City_Code' or col == 'Date' or
                                                    col == 'today_verified_cases']]
        train_df = train_df.drop(['today_verified_cases'], axis=1)
        train_df = pd.merge(train_df, new_train_df_verified_cases, how="inner", on=["City_Code", "Date"])

        # train_df_sorted_by_city_code_and_date = train_df
        # train_df_sorted_by_city_code_and_date = train_df_sorted_by_city_code_and_date.sort_values(ascending=True, by=['City_Code', 'Date'])
        # print_train_set(train_df_sorted_by_city_code_and_date)

        train_df_sorted_by_verified_cases = train_df
        train_df_sorted_by_verified_cases = train_df_sorted_by_verified_cases.sort_values(ascending=False, by=['today_verified_cases'])
        print_train_set(train_df_sorted_by_verified_cases)

        train_df = train_df.drop(['Date'], axis=1)



        RF_best_min_samples_leaf = 1
        RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

        for col in train_df:
            if col not in RF_best_features and col != 'today_verified_cases':
                train_df = train_df.drop([col], axis=1)
                test_df = test_df.drop([col], axis=1)

        X_train, Y_train, X_test, Y_test = DecisionTreeAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

        RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=RF_best_min_samples_leaf)
        RF_regressor.fit(X_train, Y_train)
        RF_test_set_res = RF_regressor.score(X_test, Y_test)

        print(f'BS == {bin_size}, Score == {RF_test_set_res}')


    x = 1

    # cv = KFold(n_splits=5, random_state=204098784, shuffle=True)
    #
    # print("PART A Final Results: ")
    # DecisionTreePrinting.print_k_fold_results('DecisionTreeRegressor', DT_regressor, X_train, Y_train, cv)
    # print(f"DecisionTreeRegressor Test Set Score : {DT_test_set_res}")
    # DecisionTreePrinting.print_k_fold_results('RandomForestRegressor', RF_regressor, X_train, Y_train, cv)
    # print(f"RandomForestRegressor Test Set Score : {RF_test_set_res}")



#######################################################################################################################
################################################### PART D ############################################################
#######################################################################################################################
#######################################################################################################################


def run_part_D():
    bin_data_set()

if __name__ == "__main__":
    run_part_D()
    # print_train_set()







