from random import sample
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from Preprocess.utils import DataParams

cols = ['City_Code','Cumulative_verified_cases',
        'Cumulated_recovered','Cumulated_deaths','Cumulated_number_of_tests',
        'Cumulated_number_of_diagnostic_tests','colour','final_score',
        'vaccinated_dose_1_total','vaccinated_dose_2_total','vaccinated_dose_3_total',
        'dose_1_in_last_1_week','dose_1_in_last_2_week','dose_1_in_last_3_week',
        'dose_1_in_last_4_week','dose_2_in_last_1_week','dose_2_in_last_2_week',
        'dose_2_in_last_3_week','dose_2_in_last_4_week','dose_3_in_last_1_week',
        'dose_3_in_last_2_week','dose_3_in_last_3_week','dose_3_in_last_4_week',
        'verified_cases_1_days_ago','verified_cases_2_days_ago',
        'verified_cases_3_days_ago','verified_cases_4_days_ago','verified_cases_5_days_ago',
        'verified_cases_6_days_ago','verified_cases_7_days_ago','verified_cases_8_days_ago',
        'verified_cases_9_days_ago','verified_cases_10_days_ago',
        'verified_cases_11_days_ago','verified_cases_12_days_ago','verified_cases_13_days_ago',
        'verified_cases_14_days_ago'] # 'Date','today_verified_cases','City_Name',
tlv_code = 5000
haifa_code = 4000
def play_knn():
    data = pd.read_csv('../Preprocess/output.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    haifa_tlv_data = data[data['City_Code'].isin([5000, 4000])]

    start_date = datetime(2021, 7, 20)
    end_date = datetime(2021, 7, 28)

    # we work in specified date range
    haifa_tlv_data = haifa_tlv_data[(haifa_tlv_data['Date'] > start_date) & (haifa_tlv_data['Date'] < end_date)]
    with_answers = haifa_tlv_data.copy()

    neigh = KNeighborsRegressor(n_neighbors=3)
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
    # for i in range(2, N + 2):
    for i in range(2, 7):
        daily_new_cases_columns.append(f'verified_cases_{i - 1}_days_ago')
    # prod_cols = ['colour','final_score',
    #              'vaccinated_dose_1_total','vaccinated_dose_2_total','vaccinated_dose_3_total',
    #              'dose_1_in_last_1_week','dose_1_in_last_2_week','dose_1_in_last_3_week',
    #              'dose_1_in_last_4_week','dose_2_in_last_1_week','dose_2_in_last_2_week',
    #              'dose_2_in_last_3_week','dose_2_in_last_4_week','dose_3_in_last_1_week',
    #              'dose_3_in_last_2_week','dose_3_in_last_3_week','dose_3_in_last_4_week',
    #              'verified_cases_1_days_ago','verified_cases_2_days_ago',
    #              'verified_cases_3_days_ago','verified_cases_4_days_ago','verified_cases_5_days_ago',
    #              'verified_cases_6_days_ago','verified_cases_7_days_ago','verified_cases_8_days_ago',
    #              'verified_cases_9_days_ago','verified_cases_10_days_ago',
    #              'verified_cases_11_days_ago','verified_cases_12_days_ago','verified_cases_13_days_ago',
    #              'verified_cases_14_days_ago']
    # data = data[base_columns + prod_cols]  #+ daily_new_cases_columns + vaccination_columns]
    data = data[base_columns + daily_new_cases_columns + vaccination_columns]
    return data


def run_knn(k, train, test):
    neigh = KNeighborsRegressor(n_neighbors=k)

    y_train = train['today_verified_cases'].values
    x_train = train.loc[:, train.columns != 'today_verified_cases']
    neigh.fit(x_train.values, y_train)

    y_test = test['today_verified_cases'].values
    x_test = test.loc[:, test.columns != 'today_verified_cases']
    acc = neigh.score(x_test.values, y_test)

    return acc


def experiment_features(examples, columns):
    for m in range(5, 15):
        print(f'm={m}')
        experiment_m_features(m, examples, columns)

def experiment_m_features(m, examples, columns):
    kf = KFold(n_splits=5, shuffle=True, random_state=307916502)
    for i in range(10):
        test_columns = sample(columns, m) + ['today_verified_cases']
        experiment_examples = examples[test_columns]
        sum = 0
        for train_index, test_index in kf.split(examples):
            train_examples, test_examples = experiment_examples.iloc[train_index], \
                                            experiment_examples.iloc[test_index]
            acc = run_knn(5, train_examples, test_examples)
            sum += acc

        avg_accuracy = sum / kf.n_splits
        print(f'accuracy: {avg_accuracy}, columns={test_columns}')

def experiment_k(examples):
    kf = KFold(n_splits=5, shuffle=True, random_state=307916502)
    K = range(3, 15)
    accuracies = []
    for k in K:
        print(f'k:{k}')
        sum = 0
        for train_index, test_index in kf.split(examples):
            train_examples, test_examples = examples.iloc[train_index], examples.iloc[test_index]

            acc = run_knn(k, train_examples,test_examples)
            sum += acc
        avg_accuracy = sum / kf.n_splits
        accuracies.append(avg_accuracy)
        print(f'K:{k}, average accuracy: {avg_accuracy}')

    plt.plot(K, accuracies)
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.show()

def run_knn_on_small_cities(population, k, train_df, test_df):
    population_df = pd.read_csv('../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])

    test_df = test_df[test_df['population'] <= population]
    test_df.drop(['population'], axis=1, inplace=True)

    acc = run_knn(k, train_df, test_df)
    print(f'knn_on_small_cities accuracy: {acc}, population={population}')

def run_knn_on_big_cities(population, k, train_df, test_df):
    population_df = pd.read_csv('../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])

    test_df = test_df[test_df['population'] >= population]
    test_df.drop(['population'], axis=1, inplace=True)

    acc = run_knn(k, train_df, test_df)
    print(f'knn_on_big_cities accuracy: {acc}, population={population}')

def run_knn_on_small_new_cases(new_cases, k, train_df, test_df):
    test_df = test_df[test_df['today_verified_cases'] <= new_cases]
    acc = run_knn(k, train_df, test_df)
    print(f'knn_on_small_new_cases accuracy: {acc}, new_cases {new_cases}')


def run_knn_on_big_new_cases(new_cases, k, train_df, test_df):
    test_df = test_df[test_df['today_verified_cases'] >= new_cases]

    acc = run_knn(k, train_df, test_df)
    print(f'knn_on_big_new_cases accuracy: {acc}, new_cases {new_cases}')


def run_knn_on_colour(colour, k, train_df, test_df):
    test_df = test_df[test_df['colour'] == colour]

    acc = run_knn(k, train_df, test_df)
    print(f'knn_on_big_new_cases accuracy: {acc}, colour {colour}')


def experiment_subset_data(k, train_df, test_df):
    run_knn_on_small_cities(10000, k, train_df, test_df)
    run_knn_on_small_cities(100000, k, train_df, test_df)
    run_knn_on_small_new_cases(30, k, train_df, test_df)
    run_knn_on_big_new_cases(400, k, train_df, test_df)
    run_knn_on_colour(0, k, train_df, test_df)
    run_knn_on_colour(1, k, train_df, test_df)
    run_knn_on_colour(2, k, train_df, test_df)
    run_knn_on_colour(3, k, train_df, test_df)


if __name__ == "__main__":
    # play_knn()
    train_df = pd.read_csv('../Preprocess/train_df.csv')
    test_df = pd.read_csv('../Preprocess/test_df.csv')

    best_columns = ['vaccinated_dose_3_total', 'dose_3_in_last_2_week', 'verified_cases_7_days_ago', 'City_Code',
                    'verified_cases_14_days_ago', 'today_verified_cases', 'colour']  # 'colour' is not realy part of the best columns

    # data = preprocess(data)
    # model, X_test, y_test = init_model(data)
    train_df = train_df[best_columns]
    test_df = test_df[best_columns]
    # acc = run_knn(5, train_df, test_df)
    # print(f'knn accuracy: {acc}')
    # experiment_k(train_df[best_columns])
    # experiment_features(train_df, cols)
    experiment_subset_data(5, train_df, test_df)