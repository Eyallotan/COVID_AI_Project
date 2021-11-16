import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from Algorithms.DecisionTreeAlgo import DecisionTreeAuxiliaries
from Algorithms.DecisionTreeAlgo import DecisionTreePrinting


#######################################################################################################################
#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################


def get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, corona_df_no_pred_col, regressor_type, leaf_samples_range):
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
            regressor.fit(X_train_rfe, Y_train)
            selected_features = [col for col in corona_df_no_pred_col]
            features = np.array(selected_features)[sel.get_support()]

            cv = KFold(n_splits=5, random_state=204098784, shuffle=True)
            scores = cross_val_score(regressor, X_train_rfe, Y_train, cv=cv, n_jobs=-1)
            mean_scores = np.mean(scores)

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
            DecisionTreePrinting.print_RFE_scores(DT_scores_vec, "DecisionTreeRegressor")
        return max(DT_results_vec)
    if regressor_type == "RandomForestRegressor":
        if leaf_samples_range == 1:
            DecisionTreePrinting.print_RFE_scores(RF_scores_vec, "RandomForestRegressor")
        return max(RF_results_vec)


def get_RFE_best_features_and_best_min_samples_leaf(pruning_option):
    train_df, test_df = DecisionTreeAuxiliaries.get_train_and_test_df()
    X_train, Y_train, X_test, Y_test = DecisionTreeAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    train_df_no_pred_col = train_df.drop(['today_verified_cases'], axis=1)

    DT_best_results = []
    RF_best_results = []

    if pruning_option == "Without Pruning":
        DT_best_results = get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, train_df_no_pred_col,
                                                         "DecisionTreeRegressor", 1)

        DecisionTreePrinting.print_best_RFE_results(DT_best_results, "DecisionTreeRegressor")

        RF_best_results = get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, train_df_no_pred_col,
                                                         "RandomForestRegressor", 1)
        DecisionTreePrinting.print_best_RFE_results(RF_best_results, "RandomForestRegressor")

    if pruning_option == "With Pruning":
        DT_best_results = get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, train_df_no_pred_col,
                                                         "DecisionTreeRegressor", 10)
        DecisionTreePrinting.print_best_RFE_results(DT_best_results, "DecisionTreeRegressor")

        RF_best_results = get_best_RFE_results_for_DTR_or_RFR(X_train, Y_train, train_df_no_pred_col,
                                                         "RandomForestRegressor", 10)
        DecisionTreePrinting.print_best_RFE_results(RF_best_results, "RandomForestRegressor")

    return DT_best_results[2], DT_best_results[3], RF_best_results[2], RF_best_results[3]


def run_part_B(pruning_option):
    # DecisionTreePrinting.print_features_importances("DecisionTreeRegressor")
    # DecisionTreePrinting.print_features_importances("RandomForestRegressor")

    # DT_best_features, DT_best_min_samples_leaf, RF_best_features, RF_best_min_samples_leaf = get_RFE_best_features_and_best_min_samples_leaf(pruning_option)

    DT_best_features = []
    RF_best_features = []
    DT_best_min_samples_leaf = 1
    RF_best_min_samples_leaf = 1

    if pruning_option == "Without Pruning":
        DT_best_features = ['City_Code', 'Cumulated_deaths', 'vaccinated_dose_1_total', 'dose_1_in_last_4_week',
                            'verified_cases_7_days_ago']

        RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    if pruning_option == "With Pruning":
        DT_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']
        DT_best_min_samples_leaf = 4

        RF_best_features = ['City_Code', 'Cumulated_deaths', 'verified_cases_7_days_ago']

    train_df, test_df = DecisionTreeAuxiliaries.get_train_and_test_df()

    train_df = train_df[[col for col in train_df.columns if col in DT_best_features or col == 'today_verified_cases']]
    test_df = test_df[[col for col in test_df.columns if col in DT_best_features or col == 'today_verified_cases']]

    X_train, Y_train, X_test, Y_test = DecisionTreeAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    DT_regressor = DecisionTreeRegressor(random_state=1, min_samples_leaf=DT_best_min_samples_leaf)
    DT_regressor.fit(X_train, Y_train)
    DT_test_set_res = DT_regressor.score(X_test, Y_test)

    cv = KFold(n_splits=5, random_state=204098784, shuffle=True)

    if pruning_option == "Without Pruning":
        print("PART B Final Results (Without Pruning): ")
    if pruning_option == "With Pruning":
        print("PART B Final Results (With Pruning): ")

    DecisionTreePrinting.print_k_fold_results('DecisionTreeRegressor', DT_regressor, X_train, Y_train, cv)
    print(f"DecisionTreeRegressor Test Set Score : {DT_test_set_res}")

    train_df, test_df = DecisionTreeAuxiliaries.get_train_and_test_df()

    train_df = train_df[[col for col in train_df.columns if col in RF_best_features or col == 'today_verified_cases']]
    test_df = test_df[[col for col in test_df.columns if col in RF_best_features or col == 'today_verified_cases']]

    X_train, Y_train, X_test, Y_test = DecisionTreeAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1, min_samples_leaf=RF_best_min_samples_leaf)
    RF_regressor.fit(X_train, Y_train)
    RF_test_set_res = RF_regressor.score(X_test, Y_test)

    DecisionTreePrinting.print_k_fold_results('RandomForestRegressor', RF_regressor, X_train, Y_train, cv)
    print(f"RandomForestRegressor Test Set Score : {RF_test_set_res}")


#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################
#######################################################################################################################


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


if __name__ == "__main__":
    run_part_B("Without Pruning")
    run_part_B("With Pruning")







