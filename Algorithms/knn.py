from datetime import datetime
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from Preprocess.utils import DataParams


def play_knn():
    data = pd.read_csv('../Preprocess/output.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    haifa_tlv_data = data[data['City_Code'].isin([5000, 4000])]

    start_date = datetime(2021, 7, 20)
    end_date = datetime(2021, 7, 28)

    # we work in specified date range
    haifa_tlv_data = haifa_tlv_data[(haifa_tlv_data['Date'] > start_date) & (haifa_tlv_data['Date'] < end_date)]
    with_answers = haifa_tlv_data.copy()

    neigh = KNeighborsClassifier(n_neighbors=3)
    Y = haifa_tlv_data['today_verified_cases']
    haifa_tlv_data.drop(['today_verified_cases', 'City_Name', 'Date'], axis=1, inplace=True)
    X = haifa_tlv_data

    neigh.fit(X, Y)

    x_hat = [4000, 16917, 31, 20, 18, 6, 15, 16, 16, 9, 11, 14, 2, 9, 2, 8]
    y_pred = neigh.predict([x_hat])
    print(y_pred)


def preprocess(data):
    params = DataParams()
    base_columns = ['City_Code', params.Y]
    N = params.number_of_days_for_infected_stats
    daily_new_cases_columns = []
    vaccination_columns = []#['vaccinated_dose_1_total', 'vaccinated_dose_2_total', 'dose_1_in_last_1_week', 'dose_1_in_last_2_week', 'dose_2_in_last_3_week']  # TODO: use them later
    for i in range(2, N + 2):
        daily_new_cases_columns.append(f'verified_cases_{i - 1}_days_ago')
    data = data[base_columns + daily_new_cases_columns + vaccination_columns]
    return data

def init_model(data):
    neigh = KNeighborsClassifier(n_neighbors=3)
    Y = data['today_verified_cases']
    data.drop(['today_verified_cases'], axis=1, inplace=True)
    X = data
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=0)
    neigh.fit(X_train, y_train)

    return neigh, X_test, y_test

def evaluate_model(model, X_test, y_test):
    acc = model.score(X_test, y_test)
    print(f'knn accuracy: {acc}')


if __name__ == "__main__":
    # play_knn()
    data = pd.read_csv('../Preprocess/corona_df.csv')
    data = preprocess(data)

    model, X_test, y_test = init_model(data)
    evaluate_model(model, X_test, y_test)
