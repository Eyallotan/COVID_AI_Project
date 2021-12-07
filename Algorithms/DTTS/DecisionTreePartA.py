from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from Algorithms.DTTS import DecisionTreeAuxiliaries
from Algorithms.DTTS import DecisionTreePrinting
import pandas as pd
from Preprocess import utils


#######################################################################################################################
#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################


# decision tree regressor vs. random forest regressor #
def run_part_A():
    params = utils.DataParams()
    corona_df = DecisionTreeAuxiliaries.get_corona_df()
    train_df, test_df = DecisionTreeAuxiliaries.get_train_and_test_df(corona_df, params)
    X_train, Y_train, X_test, Y_test = DecisionTreeAuxiliaries.get_X_and_Y_tarin_test_sets(train_df, test_df)

    DT_regressor = DecisionTreeRegressor(random_state=1)
    DT_regressor.fit(X_train, Y_train)
    DT_test_set_res = DT_regressor.score(X_test, Y_test)
    DT_Y_pred = DT_regressor.predict(X_test)

    RF_regressor = RandomForestRegressor(n_estimators=10, random_state=1)
    RF_regressor.fit(X_train, Y_train)
    RF_test_set_res = RF_regressor.score(X_test, Y_test)
    RF_Y_pred = RF_regressor.predict(X_test)

    print("PART A Final Results: ")
    print(f"DecisionTreeRegressor Test Set Score : {DT_test_set_res}")
    print(f"RandomForestRegressor Test Set Score : {RF_test_set_res}")

    # DecisionTreePrinting.print_y_test_vs_y_predict(Y_test, DT_Y_pred, RF_Y_pred)


#######################################################################################################################
################################################### PART A ############################################################
#######################################################################################################################
#######################################################################################################################


if __name__ == "__main__":
    run_part_A()








