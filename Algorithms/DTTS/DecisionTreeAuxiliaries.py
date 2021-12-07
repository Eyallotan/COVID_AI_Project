import pandas as pd


def get_corona_df():
    corona_df = pd.read_csv('../../Preprocess/corona_df.csv')
    corona_df['Date'] = pd.to_datetime(corona_df['Date'])
    corona_df = corona_df.sort_values(ascending=False, by=['Date'])

    return corona_df

def get_train_and_test_df(df, params):
    train_df = df[df['Date'] < params.split_date]
    test_df = df[df['Date'] >= params.split_date]
    train_df = train_df.drop(['today_verified_cases_smoothed', 'City_Name', 'Date'], axis=1)
    test_df = test_df.drop(['today_verified_cases_smoothed', 'City_Name', 'Date'], axis=1)

    return train_df, test_df

def get_X_and_Y_tarin_test_sets(train_df, test_df):
    X_train = train_df[[col for col in train_df if col != 'today_verified_cases']].values
    Y_train = train_df['today_verified_cases'].values
    X_test = test_df[[col for col in test_df if col != 'today_verified_cases']].values
    Y_test = test_df['today_verified_cases'].values

    return X_train, Y_train, X_test, Y_test




