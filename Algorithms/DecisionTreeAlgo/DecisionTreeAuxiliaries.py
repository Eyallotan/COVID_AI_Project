import pandas as pd


def get_train_and_test_df():
    train_df = pd.read_csv('../../Preprocess/train_df.csv')
    train_df = train_df.drop(['City_Name', 'Date'], axis=1)
    test_df = pd.read_csv('../../Preprocess/test_df.csv')
    test_df = test_df.drop(['City_Name', 'Date'], axis=1)

    return train_df, test_df


def get_X_and_Y_tarin_test_sets(train_df, test_df):
    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    return X_train, Y_train, X_test, Y_test