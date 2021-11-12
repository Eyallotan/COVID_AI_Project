import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

font = {'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)


#######################################################################################################################
#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################


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
    train_df = pd.read_csv('../Preprocess/train_df.csv')
    train_df = train_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    test_df = pd.read_csv('../Preprocess/test_df.csv')
    test_df = test_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    DT_regressor = DecisionTreeRegressor(random_state=1)
    DT_regressor.fit(X_train, Y_train)
    DT_res = DT_regressor.score(X_test, Y_test)
    DT_Y_pred = DT_regressor.predict(X_test)

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1)
    RF_regressor.fit(X_train, Y_train)
    RF_res = RF_regressor.score(X_test, Y_test)
    RF_Y_pred = RF_regressor.predict(X_test)

    print("PART A Final Results: ")
    print(f"DecisionTreeRegressor Score : {DT_res}")
    print(f"RandomForestRegressor Score : {RF_res}")
    print('\n')

    # print_y_test_vs_y_predict(Y_test, DT_Y_pred, RF_Y_pred)


#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################


def create_regressor_and_return_score(X_train, Y_train, X_test, Y_test, regressor_type):
    if regressor_type == "DecisionTreeRegressor":
        DT_regressor = DecisionTreeRegressor(random_state=1)
        DT_regressor.fit(X_train, Y_train)
        return DT_regressor.score(X_test, Y_test)

    if regressor_type == "RandomForestRegressor":
        RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1)
        RF_regressor.fit(X_train, Y_train)
        return RF_regressor.score(X_test, Y_test)


def run_simulation(part, color_num, min_dose_2_vaccinated_percentage):

    train_df = pd.read_csv('../Preprocess/train_df.csv')
    train_df = train_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    test_df = pd.read_csv('../Preprocess/test_df.csv')
    test_df = test_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)

    old_train_df_size = len(train_df)

    if part == 'part_B1':
        train_df = train_df.loc[(train_df['colour'] == color_num)]
        test_df = test_df.loc[(test_df['colour'] == color_num)]
    if part == 'part_B2':
        train_df = train_df.loc[(train_df['vaccinated_dose_2_total'] >= min_dose_2_vaccinated_percentage) & (
                    train_df['vaccinated_dose_2_total'] < min_dose_2_vaccinated_percentage + 0.25)]
        test_df = test_df.loc[(test_df['vaccinated_dose_2_total'] >= min_dose_2_vaccinated_percentage) & (
                test_df['vaccinated_dose_2_total'] < min_dose_2_vaccinated_percentage + 0.25)]
    if part == 'part_B3':
        train_df = train_df.loc[(train_df['colour'] == color_num) & (
                    train_df['vaccinated_dose_2_total'] >= min_dose_2_vaccinated_percentage) & (
                                          train_df['vaccinated_dose_2_total'] < min_dose_2_vaccinated_percentage + 0.25)]
        test_df = test_df.loc[(test_df['colour'] == color_num) & (
                    test_df['vaccinated_dose_2_total'] >= min_dose_2_vaccinated_percentage) & (
                                          test_df['vaccinated_dose_2_total'] < min_dose_2_vaccinated_percentage + 0.25)]

    train_df.index = pd.RangeIndex(len(train_df.index))
    test_df.index = pd.RangeIndex(len(test_df.index))

    # train_df = train_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    # test_df = test_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)

    # corona_df_no_pred_col = corona_df.drop(['today_verified_cases'], axis=1)
    new_train_df_size = len(train_df)
    # X_train, X_test, Y_train, Y_test = train_test_split(corona_df_no_pred_col,corona_df['today_verified_cases'], test_size=0.2, random_state=1)

    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    DT_res = create_regressor_and_return_score(X_train, Y_train, X_test, Y_test, regressor_type="DecisionTreeRegressor")
    RF_res = create_regressor_and_return_score(X_train, Y_train, X_test, Y_test, regressor_type="RandomForestRegressor")

    return DT_res, RF_res, new_train_df_size/old_train_df_size


# partition by colors #
def run_part_B():
    # PART B1:
    # partition by colors #
    avg_DT_vec = []
    avg_RF_vec = []
    for color_num in range(4):
        DT_res, RF_res, regressor_size_percentage = run_simulation(part='part_B1', color_num=color_num, min_dose_2_vaccinated_percentage=None)
        avg_DT_vec.append((DT_res, regressor_size_percentage))
        avg_RF_vec.append((RF_res, regressor_size_percentage))

    DT_score = 0
    RF_score = 0
    for i in range(4):
        DT_score += avg_DT_vec[i][0]*avg_DT_vec[i][1]
        RF_score += avg_RF_vec[i][0]*avg_RF_vec[i][1]

    print("Partition By Colors Final Results: ")
    print(f"DecisionTreeRegressor Score : {DT_score}")
    print(f"RandomForestRegressor Score : {RF_score}")
    print('\n')

    # PART B2:
    # partition by vaccinated percentage #
    min_dose_2_vaccinated_percentage = 0
    avg_DT_vec = []
    avg_RF_vec = []
    while (min_dose_2_vaccinated_percentage <= 0.75):
        DT_res, RF_res, regressor_size_percentage = run_simulation(part='part_B2', color_num=None,
                                        min_dose_2_vaccinated_percentage=min_dose_2_vaccinated_percentage)
        avg_DT_vec.append((DT_res, regressor_size_percentage))
        avg_RF_vec.append((RF_res, regressor_size_percentage))
        min_dose_2_vaccinated_percentage += 0.25

    DT_score = 0
    RF_score = 0
    for i in range(4):
        DT_score += avg_DT_vec[i][0] * avg_DT_vec[i][1]
        RF_score += avg_RF_vec[i][0] * avg_RF_vec[i][1]

    print("Partition By Vaccinated Percentage Final Results: ")
    print(f"DecisionTreeRegressor Score : {DT_score}")
    print(f"RandomForestRegressor Score : {RF_score}")
    print('\n')

    # PART B3:
    # partition by color and vaccinated percentage #
    min_dose_2_vaccinated_percentage = 0
    first_avg_DT_vec = []
    first_avg_RF_vec = []
    final_avg_DT_vec = []
    final_avg_RF_vec = []
    for color_num in range(4):
        while (min_dose_2_vaccinated_percentage <= 0.75):
            DT_res, RF_res, regressor_size_percentage = run_simulation(part='part_B3', color_num=color_num,
                                            min_dose_2_vaccinated_percentage=min_dose_2_vaccinated_percentage)
            first_avg_DT_vec.append((DT_res, regressor_size_percentage))
            first_avg_RF_vec.append((RF_res, regressor_size_percentage))
            min_dose_2_vaccinated_percentage += 0.25

        final_avg_DT_vec.append(first_avg_DT_vec)
        final_avg_RF_vec.append(first_avg_RF_vec)
        min_dose_2_vaccinated_percentage = 0
        first_avg_RF_vec = []
        first_avg_DT_vec = []

    final_DT_score = 0
    final_RF_score = 0
    for i in range(4):
        for j in range(4):
            final_DT_score += final_avg_DT_vec[i][j][0]*final_avg_DT_vec[i][j][1]
            final_RF_score += final_avg_RF_vec[i][j][0]*final_avg_RF_vec[i][j][1]

    print("Partition By Color And Vaccinated Percentage Final Results: ")
    print(f"DecisionTreeRegressor Score : {final_DT_score}")
    print(f"RandomForestRegressor Score : {final_RF_score}")
    print('\n')


#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
#######################################################################################################################
################################################### PART C ############################################################
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


def best_features_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, train_df_no_pred_col, regressor_type, leaf_samples_range):
    best_score = 0
    best_index = 1
    best_features = []
    print('Regressor type: ', regressor_type)
    regressor = 'DTR_or_RFR'
    sel = 'RFE'
    DTR_results_vec = []
    DTR_scores_vec = []
    RFR_results_vec = []
    RFR_scores_vec = []
    for leaf_samples in range(1, leaf_samples_range+1):
        for index in range(1, len(train_df_no_pred_col.columns)):
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
            selected_features = [col for col in train_df_no_pred_col]
            features = np.array(selected_features)[sel.get_support()]
            regressor_score = regressor.score(X_test_rfe, Y_test)

            if regressor_type == "DecisionTreeRegressor" and leaf_samples == 1:
                DTR_scores_vec.append(regressor_score)
            if regressor_type == "RandomForestRegressor" and leaf_samples == 1:
                RFR_scores_vec.append(regressor_score)

            if best_score < regressor_score:
                best_index = index
                best_score = regressor_score
                best_features = features

        print('\n')
        print('Regressor type: ', regressor_type)
        print("Samples In One Leaf: ", leaf_samples)
        print('Number of features: ', best_index)
        print('Score: ', best_score)
        print("Features were selected: ", best_features)
        print('\n')

        if regressor_type == "DecisionTreeRegressor":
            DTR_results_vec.append((best_score, best_index, best_features, leaf_samples))
        if regressor_type == "RandomForestRegressor":
            RFR_results_vec.append((best_score, best_index, best_features, leaf_samples))

        best_index = 0
        best_score = 0
        best_features = []

    if regressor_type == "DecisionTreeRegressor":
        if leaf_samples_range == 1:
            print_RFE_scores(DTR_scores_vec, "DecisionTreeRegressor")
        return max(DTR_results_vec)
    if regressor_type == "RandomForestRegressor":
        if leaf_samples_range == 1:
            print_RFE_scores(RFR_scores_vec, "RandomForestRegressor")
        return max(RFR_results_vec)


def print_best_features_results(regressor_best_features, regressor_type):
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
    train_df = pd.read_csv('../Preprocess/train_df.csv')
    train_df = train_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    test_df = pd.read_csv('../Preprocess/test_df.csv')
    test_df = test_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    train_df_no_pred_col = train_df.drop(['today_verified_cases'], axis=1)

    regressor = 'DTR_or_RFR'
    if regressor_type == "DecisionTreeRegressor":
        regressor = DecisionTreeRegressor(random_state=1)
        regressor.fit(X_train, Y_train)
        plt.title('DecisionTreeRegressor - Features importances')
    if regressor_type == "RandomForestRegressor":
        regressor = RandomForestRegressor(n_estimators=10, random_state=1)
        regressor.fit(X_train, Y_train)
        plt.title('RandomForestRegressor - Features importances')

    features = [col for col in train_df_no_pred_col]
    f_i = list(zip(features, regressor.feature_importances_))
    f_i.sort(key=lambda x: x[1])
    plt.barh([x[0] for x in f_i], [x[1] for x in f_i])
    plt.show()


def print_best_results_PART_C():
    train_df = pd.read_csv('../Preprocess/train_df.csv')
    train_df = train_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    test_df = pd.read_csv('../Preprocess/test_df.csv')
    test_df = test_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    train_df_no_pred_col = train_df.drop(['today_verified_cases'], axis=1)

    DTR_best_features = best_features_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, train_df_no_pred_col,
                                                     "DecisionTreeRegressor", 1)
    print_best_features_results(DTR_best_features, "DecisionTreeRegressor")

    RFR_best_features = best_features_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, train_df_no_pred_col,
                                                     "RandomForestRegressor", 1)
    print_best_features_results(RFR_best_features, "RandomForestRegressor")

    DTR_best_features = best_features_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, train_df_no_pred_col,
                                                     "DecisionTreeRegressor", 10)
    print_best_features_results(DTR_best_features, "DecisionTreeRegressor")

    RFR_best_features = best_features_for_DTR_or_RFR(X_train, Y_train, X_test, Y_test, train_df_no_pred_col,
                                                     "RandomForestRegressor", 10)
    print_best_features_results(RFR_best_features, "RandomForestRegressor")

def run_part_C():
    print_features_importances("DecisionTreeRegressor")
    # print_features_importances("RandomForestRegressor")

    # print_best_results_PART_C()

    DTR_best_features = ['Cumulated_recovered', 'Cumulated_deaths', 'Cumulated_number_of_tests',
                         'Cumulated_number_of_diagnostic_tests',
                         'colour', 'vaccinated_dose_1_total', 'vaccinated_dose_2_total', 'dose_1_in_last_2_week',
                         'dose_1_in_last_4_week',
                         'verified_cases_7_days_ago', 'verified_cases_12_days_ago']

    RFR_best_features = ['Cumulated_recovered', 'Cumulated_deaths', 'final_score', 'vaccinated_dose_1_total',
                         'dose_1_in_last_4_week',
                         'verified_cases_7_days_ago']

    train_df = pd.read_csv('../Preprocess/train_df.csv')
    train_df = train_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    test_df = pd.read_csv('../Preprocess/test_df.csv')
    test_df = test_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)

    for col in train_df:
        if col not in DTR_best_features and col != 'today_verified_cases':
            train_df = train_df.drop([col], axis=1)
            test_df = test_df.drop([col], axis=1)

    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    DT_regressor = DecisionTreeRegressor(random_state=1, min_samples_leaf=4)
    DT_regressor.fit(X_train, Y_train)
    DT_res = DT_regressor.score(X_test, Y_test)


    train_df = pd.read_csv('../Preprocess/train_df.csv')
    train_df = train_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)
    test_df = pd.read_csv('../Preprocess/test_df.csv')
    test_df = test_df.drop(['City_Name', 'City_Code', 'Date'], axis=1)

    for col in train_df:
        if col not in RFR_best_features and col != 'today_verified_cases':
            train_df = train_df.drop([col], axis=1)
            test_df = test_df.drop([col], axis=1)

    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=1)
    RF_regressor.fit(X_train, Y_train)
    RF_res = RF_regressor.score(X_test, Y_test)

    print("PART C Final Results: ")
    print(f"DecisionTreeRegressor Score : {DT_res}")
    print(f"RandomForestRegressor Score : {RF_res}")
    print('\n')

    return DT_regressor, RF_regressor


#######################################################################################################################
################################################### PART C ############################################################
#######################################################################################################################
#######################################################################################################################


if __name__ == "__main__":
    run_part_A()
    # run_part_B()
    # run_part_C()





