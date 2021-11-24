import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from Algorithms.DT import DecisionTreeAuxiliaries


# font = {'weight' : 'bold',
#         'size'   : 16}
#
# plt.rc('font', **font)


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
    train_df, test_df = DecisionTreeAuxiliaries.get_train_and_test_df()
    X_train, Y_train, X_test, Y_test = DecisionTreeAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

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


#######################################################################################################################
################################################### PART B ############################################################
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
#######################################################################################################################
################################################### PART C ############################################################
#######################################################################################################################


def print_r2_score_vs_MAE_score(experiment_func, experiment_type, score_results):
    x_ax = 0
    if experiment_type == 'City Population':
        x_ax = [50000, 60000, 70000, 80000, 90000, 100000]
    if experiment_type == 'Colour':
        x_ax = [0, 1, 2, 3]
    if experiment_func == 'R^2':
        plt.plot(x_ax, score_results, linewidth=1.5, label="R^2 Score Results")
    if experiment_func == 'MAE':
        plt.plot(x_ax, score_results, linewidth=1.5, label="MAE Score Results")
    plt.title(f'{experiment_type} Scores Comparison')
    plt.ylabel('Score')
    if experiment_type == 'City Population':
        plt.xlabel('City Population Amount')
    if experiment_type == 'Colour':
        plt.xlabel('Colour Number')

    for x, y in zip(x_ax, score_results):
        label = float("{:.3f}".format(y))

        plt.annotate(label, (x, y), textcoords="offset points", xytext=(-15, 8), ha='center')
    for x, y in zip(x_ax, score_results):
        label = float("{:.3f}".format(y))

        plt.annotate(label, (x, y), textcoords="offset points", xytext=(-15, 8), ha='center')

    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


#######################################################################################################################
################################################### PART C ############################################################
#######################################################################################################################
#######################################################################################################################







