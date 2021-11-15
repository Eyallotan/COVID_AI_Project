from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

font = {'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

def get_train_and_test_df():
    train_df = pd.read_csv('../Preprocess/train_df.csv')
    # train_df = train_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    train_df = train_df.drop(['City_Name', 'Date'], axis=1)
    test_df = pd.read_csv('../Preprocess/test_df.csv')
    # test_df = test_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    return train_df, test_df


def get_X_and_Y_tarin_test_sets(train_df, test_df):
    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    return X_train, Y_train, X_test, Y_test


#######################################################################################################################
#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################


def print_k_fold_results(regressor_type, regressor, X_train, Y_train, cv):
    scores = cross_val_score(regressor, X_train, Y_train, cv=cv, n_jobs=-1)
    k_fold_res = np.mean(scores)
    if regressor_type == "DecisionTreeRegressor":
        print(f"DecisionTreeRegressor K-fold Score : {k_fold_res}")
    if regressor_type == "RandomForestRegressor":
        print(f"RandomForestRegressor K-fold Score : {k_fold_res}")


def print_y_test_vs_y_predict(Y_test, DT_regressor_results, RF_regressor_results):
    x_ax = range(len(Y_test[30:71]))
    plt.plot(x_ax, Y_test[30:71], linewidth=1, label="Original")
    plt.plot(x_ax, DT_regressor_results[30:71], linewidth=1.1, label="DecisionTreeRegressor Predicted")
    plt.plot(x_ax, RF_regressor_results[30:71], linewidth=1.1, label="RandomForestRegressor Predicted")
    plt.title("DecisionTreeRegressor and RandomForestRegressor - Y_test vs. Y_predicted")
    plt.xlabel('Number of Sample')
    plt.ylabel('Value')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


# decision tree regressor vs. random forest regressor #
def run_part_A():
    train_df, test_df = get_train_and_test_df()
    X_train, Y_train, X_test, Y_test = get_X_and_Y_tarin_test_sets(train_df, test_df)

    DT_regressor = DecisionTreeRegressor(random_state=1)
    DT_regressor.fit(X_train, Y_train)

    RF_regressor = RandomForestRegressor(n_estimators=11, random_state=1)
    RF_regressor.fit(X_train, Y_train)

    cv = KFold(n_splits=5, random_state=204098784, shuffle=True)

    print("PART A Final Results: ")
    print_k_fold_results('DecisionTreeRegressor', DT_regressor, X_train, Y_train, cv)
    print_k_fold_results('RandomForestRegressor', RF_regressor, X_train, Y_train, cv)

    DT_test_set_res = DT_regressor.score(X_test, Y_test)
    DT_Y_pred = DT_regressor.predict(X_test)

    RF_test_set_res = RF_regressor.score(X_test, Y_test)
    RF_Y_pred = RF_regressor.predict(X_test)

    print(f"DecisionTreeRegressor Test Set Score : {DT_test_set_res}")
    print(f"RandomForestRegressor Test Set Score : {RF_test_set_res}")
    print('\n')

    print_y_test_vs_y_predict(Y_test, DT_Y_pred, RF_Y_pred)


#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################


def print_RFE_scores(regressor_scores, regressor_type):
    num_of_features_range = []
    for i in range(1, len(regressor_scores)+1):
        num_of_features_range.append(i)
    plt.plot(num_of_features_range, regressor_scores, color='b', linewidth=3, marker='8')
    plt.xlabel('Number of features selected', fontsize=15)
    plt.ylabel('% Score', fontsize=15)

    if regressor_type == "DecisionTreeRegressor":
        plt.title('DecisionTreeRegressor - Recursive Feature Elimination')
    if regressor_type == "RandomForestRegressor":
        plt.title('RandomForestRegressor - Recursive Feature Elimination')

    plt.show()


def get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, corona_df_no_pred_col, regressor_type, leaf_samples_range):
    best_score = 0
    best_index = 1
    best_features = []
    print('Regressor type: ', regressor_type)
    regressor = 'DTR_or_RFR'
    sel = 'RFE'
    DT_results_vec = []
    DT_scores_vec = []
    RF_results_vec = []
    RF_scores_vec = []
    for leaf_samples in range(1, leaf_samples_range+1):
        for index in range(1, len(corona_df_no_pred_col.columns)):
            print("Leafs Samples ", leaf_samples, "Index:", index)
            if regressor_type == "DecisionTreeRegressor":
                sel = RFE(DecisionTreeRegressor(random_state=1, min_samples_leaf=leaf_samples), n_features_to_select=index)
                sel.fit(X_train, Y_train)
                regressor = DecisionTreeRegressor(random_state=1, min_samples_leaf=leaf_samples)
            if regressor_type == "RandomForestRegressor":
                sel = RFE(RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=leaf_samples), n_features_to_select=index)
                sel.fit(X_train, Y_train)
                regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=leaf_samples)

            X_train_rfe = sel.transform(X_train)
            X_test_rfe = sel.transform(X_test)
            regressor.fit(X_train_rfe, Y_train)
            selected_features = [col for col in corona_df_no_pred_col]
            features = np.array(selected_features)[sel.get_support()]

            cv = KFold(n_splits=5, random_state=204098784, shuffle=True)
            # evaluate model
            scores = cross_val_score(regressor, X_train_rfe, Y_train, cv=cv, n_jobs=-1)
            mean_scores = np.mean(scores)
            # regressor_score = regressor.score(X_test_rfe, Y_test)

            if regressor_type == "DecisionTreeRegressor" and leaf_samples == 1:
                DT_scores_vec.append(mean_scores)
            if regressor_type == "RandomForestRegressor" and leaf_samples == 1:
                RF_scores_vec.append(mean_scores)

            if best_score < mean_scores:
                best_index = index
                best_score = mean_scores
                best_features = features

        print('\n')
        print('Regressor type: ', regressor_type)
        print("Samples In One Leaf: ", leaf_samples)
        print('Number of features: ', best_index)
        print('Score: ', best_score)
        print("Features were selected: ", best_features)
        print('\n')

        if regressor_type == "DecisionTreeRegressor":
            DT_results_vec.append((best_score, best_index, best_features, leaf_samples))
        if regressor_type == "RandomForestRegressor":
            RF_results_vec.append((best_score, best_index, best_features, leaf_samples))

        best_index = 0
        best_score = 0
        best_features = []

    if regressor_type == "DecisionTreeRegressor":
        if leaf_samples_range == 1:
            print_RFE_scores(DT_scores_vec, "DecisionTreeRegressor")
        return max(DT_results_vec)
    if regressor_type == "RandomForestRegressor":
        if leaf_samples_range == 1:
            print_RFE_scores(RF_scores_vec, "RandomForestRegressor")
        return max(RF_results_vec)


def print_best_RFE_results(regressor_best_features, regressor_type):
    print('\n')
    if regressor_type == "DecisionTreeRegressor":
        print('Regressor Type: DecisionTreeRegressor')
    if regressor_type == "RandomForestRegressor":
        print('Regressor Type: RandomForestRegressor')
    print('Best Score: ', regressor_best_features[0])
    print('Best Number Of Features: ', regressor_best_features[1])
    print("Features Were Selected: ", regressor_best_features[2])
    print("Min Samples in Leaf: ", regressor_best_features[3])


def print_features_importances(regressor_type):
    train_df, test_df = get_train_and_test_df()
    X_train, Y_train, X_test, Y_test = get_X_and_Y_tarin_test_sets(train_df, test_df)

    regressor = 'DTR_or_RFR'
    if regressor_type == "DecisionTreeRegressor":
        regressor = DecisionTreeRegressor(random_state=1)
        regressor.fit(X_train, Y_train)
        plt.title('DecisionTreeRegressor - Features importances')
    if regressor_type == "RandomForestRegressor":
        regressor = RandomForestRegressor(n_estimators=10, random_state=1)
        regressor.fit(X_train, Y_train)
        plt.title('RandomForestRegressor - Features importances')

    features = [col for col in train_df if col != 'today_verified_cases']
    f_i = list(zip(features, regressor.feature_importances_))
    f_i.sort(key=lambda x: x[1])
    plt.barh([x[0] for x in f_i], [x[1] for x in f_i])
    plt.show()


def get_RFE_best_features_and_best_min_samples_leaf():
    train_df, test_df = get_train_and_test_df()
    X_train, Y_train, X_test, Y_test = get_X_and_Y_tarin_test_sets(train_df, test_df)

    train_df_no_pred_col = train_df.drop(['today_verified_cases'], axis=1)

    # DT_best_results = get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, train_df_no_pred_col,
    #                                                  "DecisionTreeRegressor", 1)
    #
    # print_best_RFE_results(DT_best_results, "DecisionTreeRegressor")
    #
    # RF_best_results = get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, train_df_no_pred_col,
    #                                                  "RandomForestRegressor", 1)
    # print_best_RFE_results(RF_best_results, "RandomForestRegressor")

    DT_best_results = get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, train_df_no_pred_col,
                                                     "DecisionTreeRegressor", 10)
    print_best_RFE_results(DT_best_results, "DecisionTreeRegressor")

    RF_best_results = get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, train_df_no_pred_col,
                                                     "RandomForestRegressor", 10)
    print_best_RFE_results(RF_best_results, "RandomForestRegressor")

    return DT_best_results[2], DT_best_results[3], RF_best_results[2], RF_best_results[3]


def run_part_B():
    # print_features_importances("DecisionTreeRegressor")
    # print_features_importances("RandomForestRegressor")

    # DT_best_features, DT_best_min_samples_leaf, RF_best_features, RF_best_min_samples_leaf = get_RFE_best_features_and_best_min_samples_leaf()

    # # no pruning:
    # DT_best_features = ['City_Code', 'Cumulated_deaths', 'vaccinated_dose_1_total', 'dose_1_in_last_4_week',
    #                     'verified_cases_7_days_ago']
    #
    # RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    # with pruning:
    DT_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    train_df, test_df = get_train_and_test_df()

    for col in train_df:
        if col not in DT_best_features and col != 'today_verified_cases':
            train_df = train_df.drop([col], axis=1)
            test_df = test_df.drop([col], axis=1)

    X_train, Y_train, X_test, Y_test = get_X_and_Y_tarin_test_sets(train_df, test_df)

    DT_regressor = DecisionTreeRegressor(random_state=1, min_samples_leaf=4)
    DT_regressor.fit(X_train, Y_train)
    DT_res = DT_regressor.score(X_test, Y_test)

    train_df, test_df = get_train_and_test_df()

    for col in train_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            train_df = train_df.drop([col], axis=1)
            test_df = test_df.drop([col], axis=1)

    X_train, Y_train, X_test, Y_test = get_X_and_Y_tarin_test_sets(train_df, test_df)

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=1)
    RF_regressor.fit(X_train, Y_train)
    RF_res = RF_regressor.score(X_test, Y_test)

    print("PART B Final Results: ")
    print(f"DecisionTreeRegressor Score : {DT_res}")
    print(f"RandomForestRegressor Score : {RF_res}")
    print('\n')


#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################
#######################################################################################################################

def get_best_decision_tree_regressor():
    RF_best_min_samples_leaf = 1
    RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']
    train_df, test_df = get_train_and_test_df()

    for col in train_df:
        if col not in RF_best_features and col != 'today_verified_cases':
            train_df = train_df.drop([col], axis=1)

    X_train, Y_train, X_test, Y_test = get_X_and_Y_tarin_test_sets(train_df, test_df)

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=RF_best_min_samples_leaf)
    RF_regressor.fit(X_train, Y_train)

    return RF_regressor


def run_decision_tree_on_small_cities(population):
    test_df = pd.read_csv('../Preprocess/test_df.csv')

    population_df = pd.read_csv('../Resources/population_table.csv')
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

    print(f'desicion_tree_on_cities accuracy: {score}, population={population}')


def run_decision_tree_on_small_new_cases(new_cases):
    test_df = pd.read_csv('../Preprocess/test_df.csv')
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

    print(f'decision_tree_on_new_cases accuracy: {score}, new_cases {new_cases}')


def run_decision_tree_on_colour(colour):
    test_df = pd.read_csv('../Preprocess/test_df.csv')
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

    print(f'decision_tree_on_colours accuracy: {score}, colour {colour}')


def run_decision_tree_on_dates(start_date, end_date):
    test_df = pd.read_csv('../Preprocess/test_df.csv')

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


def sub_test_sets_experiments():
    run_decision_tree_on_small_cities(10000)
    run_decision_tree_on_small_cities(100000)
    run_decision_tree_on_small_new_cases(30)
    run_decision_tree_on_small_new_cases(400)
    run_decision_tree_on_colour(0)
    run_decision_tree_on_colour(1)
    run_decision_tree_on_colour(2)
    run_decision_tree_on_colour(3)

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
    # run_part_A()
    # run_part_B()
    sub_test_sets_experiments()







