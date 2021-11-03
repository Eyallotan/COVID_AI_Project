from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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
    neigh.fit(X_train.values, y_train.values)

    return neigh, X_test, y_test


def check_score_function(model, X_test, y_test):
    row159 = [X_test.iloc[159]]
    y159 = y_test.iloc[159]
    ypred159 = model.predict(row159)
    print(f'y: {y159}, y_hat: {ypred159}')
    acc = model.score(row159, [133])
    print(f'159 accuracy: {acc}')  # Conclustion: score gives 1 only if same result


def evaluate_model(model, X_test, y_test):
    acc = model.score(X_test.values, y_test.values)
    print(f'knn accuracy: {acc}')

    print('exact eval:')
    exact = 0
    for index, row in X_test.iterrows():
        ypred = model.predict([row.values])
        if ypred == y_test[index]:
            exact+=1

    print(f'exact acc: {exact/len(X_test)}. exact: {exact}, len_test: {len(X_test)}')

    print('second accuracy:')
    diffs = []
    for index, row in X_test.iterrows():
        ypred = model.predict([row.values])
        # if ypred - y_test[index] == 0:
        #     continue
        if y_test[index] == 0:
            if ypred == 0:
                diffs.append(0)
            else:
                diffs.append(1)  # 1 for max error. abs(ypred) is not good
        else:
            acc = abs(ypred - y_test[index]) / y_test[index]
            diffs.append(acc[0])

    print(f'second accuracy: {1 - np.mean(diffs)}')


def experiment(examples):
    kf = KFold(n_splits=5, shuffle=True, random_state=307916502)
    K = range(3, 50)
    accuracies = []
    for k in K:
        print(f'k:{k}')
        sum = 0
        for train_index, test_index in kf.split(examples):
            neigh = KNeighborsClassifier(n_neighbors=k)
            train_examples, test_examples = examples.iloc[train_index], examples.iloc[test_index]

            y_train = train_examples['today_verified_cases'].values
            x_train = train_examples.loc[:, train_examples.columns != 'today_verified_cases']
            neigh.fit(x_train.values, y_train)

            y_test = test_examples['today_verified_cases'].values
            x_test = test_examples.loc[:, test_examples.columns != 'today_verified_cases']
            acc = neigh.score(x_test.values, y_test)
            # print(f'k={k}, knn accuracy: {acc}')

            sum += acc
        avg_accuracy = sum / kf.n_splits
        accuracies.append(avg_accuracy)
        print(f'K:{k}, average accuracy: {avg_accuracy}')

    plt.plot(K, accuracies)
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.show()

if __name__ == "__main__":
    # play_knn()
    data = pd.read_csv('train_df.csv')
    data = preprocess(data)

    experiment(data)
    # model, X_test, y_test = init_model(data)
    # evaluate_model(model, X_test, y_test)
