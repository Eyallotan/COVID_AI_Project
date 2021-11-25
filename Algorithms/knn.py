from random import sample
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math

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
min = '2021-01-20'
max = '2021-09-11'
params = DataParams()


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


def run_knn(k, train, test, weights='uniform'):
    neigh = KNeighborsRegressor(n_neighbors=k, weights=weights)

    y_train = train[params.Y].values
    x_train = train.loc[:, train.columns != params.Y]
    neigh.fit(x_train.values, y_train)

    y_test = test[params.Y].values
    x_test = test.loc[:, test.columns != params.Y]
    acc = neigh.score(x_test.values, y_test)
    y_predicted = neigh.predict(x_test.values)
    mse = math.sqrt(mean_squared_error(y_test, y_predicted))

    my_acc = 1 - ((y_test - y_predicted)** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()

    return {'acc': acc, 'mse': mse, 'test_mean': y_test.mean(), 'my_acc': my_acc,
            'y_test': y_test, 'y_predicted': y_predicted}


def plot_graphs(k, train, test, weights='uniform'):
    knn_output = run_knn(k, train, test, weights)
    y_test = knn_output['y_test']
    y_predicted = knn_output['y_predicted']
    plt.plot(y_test[:100], '.', label="y_true")
    plt.plot(y_predicted[:100], '.', label="y_predicted")
    plt.axhline(y_predicted[:100].mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(y_predicted[:100].std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('first 100 examples')
    plt.show()

    diff = abs(y_test - y_predicted)
    plt.errorbar(range(100), y_test[:100], diff[:100], linestyle='None', marker='^')
    plt.title('errorbar')
    plt.show()

    plt.plot(diff[:100], label="diff")
    plt.axhline(y_test[:100].mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(y_test[:100].std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('diff first 100 examples')
    plt.show()

    y_no_patients = y_test[y_test == 0]
    y_pred_no_patients = y_predicted[y_test == 0]
    plt.plot(y_no_patients[:100], '.', label="y_true")
    plt.plot(y_pred_no_patients[:100], '.', label="y_predicted")
    plt.axhline(y_no_patients[:100].mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(y_no_patients[:100].std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('y_true equals 0')
    plt.show()

    y_spec_patients = y_test[(y_test >= 50) & (y_test <= 200)]
    y_pred_spec_patients = y_predicted[(y_test >= 50) & (y_test <= 200)]

    # diff = abs(y_spec_patients - y_pred_spec_patients)
    # plt.errorbar(range(100), y_spec_patients[:100], diff[:100], linestyle='None', marker='^')
    # plt.axhline(y_spec_patients[:100].mean(), color='r', alpha=0.2, linestyle='--')
    # plt.title('errorbar 50 to 200')
    # plt.show()

    plt.plot(y_spec_patients[:100], '.', label="y_true")
    plt.plot(y_pred_spec_patients[:100], '.', label="y_predicted")
    plt.axhline(y_spec_patients[:100].mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(y_spec_patients[:100].std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('y_true 50 to 200')
    plt.show()


def experiment_features(examples, columns):
    for m in range(5, 15):
        print(f'm={m}')
        experiment_m_features(m, examples, columns)


def experiment_m_features(m, examples, columns):
    examples['Date'] = pd.to_datetime(examples['Date'])
    for i in range(30):
        test_columns = sample(columns, m) + [params.Y]

        split_date = datetime(2021, 7, 23)
        train_examples = examples[examples['Date'] < split_date][test_columns]
        test_examples = examples[examples['Date'] >= split_date][test_columns]
        knn_output = run_knn(5, train_examples, test_examples)
        acc = knn_output['acc']
        mse = knn_output['mse']
        test_mean = knn_output['test_mean']

        print(f'accuracy:{acc}, mse:{mse}, test_mean:{test_mean}, columns={test_columns}')


def experiment_k(examples):
    kf = KFold(n_splits=5, shuffle=True, random_state=307916502)
    K = range(3, 25)
    accuracies = []
    for k in K:
        print(f'k:{k}')
        sum = 0
        for train_index, test_index in kf.split(examples):
            train_examples, test_examples = examples.iloc[train_index], examples.iloc[test_index]

            acc = run_knn(k, train_examples, test_examples)
            sum += acc['acc']
        avg_accuracy = sum / kf.n_splits
        accuracies.append(avg_accuracy)
        print(f'K:{k}, average accuracy: {avg_accuracy}')

    plt.plot(K, accuracies)
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.show()


def experiment_param(examples):
    kf = KFold(n_splits=5, shuffle=True, random_state=307916502)

    uniform_acc_sum = 0
    distance_acc_sum = 0
    for train_index, test_index in kf.split(examples):
        train_examples, test_examples = examples.iloc[train_index], examples.iloc[test_index]

        uniform_acc = run_knn(6, train_examples, test_examples, 'uniform')
        distance_acc = run_knn(6, train_examples, test_examples, 'distance')
        # print(f'uniform_accuracy: {uniform_acc}, distance_accuracy: {distance_acc}')
        uniform_acc_sum += uniform_acc
        distance_acc_sum += distance_acc

    avg_uniform_accuracy = uniform_acc_sum / kf.n_splits
    avg_distance_accuracy = distance_acc_sum / kf.n_splits
    print(f'uniform_accuracy: {avg_uniform_accuracy}, distance_accuracy: {avg_distance_accuracy}')


def run_knn_on_small_cities(population, k, train_df, test_df):
    population_df = pd.read_csv('../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])

    test_df = test_df[test_df['population'] <= population]
    test_df.drop(['population'], axis=1, inplace=True)

    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mse = knn_output['mse']
    test_mean = knn_output['test_mean']
    print(f'knn_on_small_cities accuracy: {acc}, mse: {mse}, y_test_mean={test_mean}, population={population}')


def run_knn_on_big_cities(population, k, train_df, test_df):
    population_df = pd.read_csv('../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])

    test_df = test_df[test_df['population'] >= population]
    test_df.drop(['population'], axis=1, inplace=True)

    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mse = knn_output['mse']
    test_mean = knn_output['test_mean']
    print(f'knn_on_big_cities accuracy: {acc}, mse: {mse}, y_test_mean={test_mean}, population={population}')


def run_knn_on_small_new_cases(new_cases, k, train_df, test_df):
    test_df = test_df[test_df[params.Y] <= new_cases]
    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mse = knn_output['mse']
    test_mean = knn_output['test_mean']
    print(f'knn_on_small_new_cases accuracy: {acc}, mse: {mse}, y_test_mean={test_mean}, new_cases {new_cases}')


def run_knn_on_big_new_cases(new_cases, k, train_df, test_df):
    test_df = test_df[test_df[params.Y] >= new_cases]

    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mse = knn_output['mse']
    test_mean = knn_output['test_mean']
    print(f'knn_on_big_new_cases accuracy: {acc}, mse: {mse}, y_test_mean={test_mean}, new_cases {new_cases}')


def run_knn_on_colour(colour, k, train_df, test_df):
    test_df = test_df[test_df['colour'] == colour]

    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mse = knn_output['mse']
    test_mean = knn_output['test_mean']
    print(f'knn_on_colour accuracy: {acc}, mse: {mse}, y_test_mean={test_mean}, colour {colour}')


def run_knn_on_dates(k, train_df, test_df, start_date, end_date, best_columns):
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df = test_df[(test_df['Date'] >= start_date) & (test_df['Date'] <= end_date)]
    test_df = test_df[best_columns]
    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mse = knn_output['mse']
    test_mean = knn_output['test_mean']
    my_acc = knn_output['my_acc']
    print(f'knn_on_big_new_cases accuracy: {acc}, mse: {mse}, y_test_mean={test_mean}, my_acc={my_acc}, start_date {start_date}, , end_date {end_date}')


def experiment_subset_data(k, train_df, test_df, full_test_df, best_columns):
    run_knn_on_small_cities(10000, k, train_df, test_df)
    run_knn_on_small_cities(100000, k, train_df, test_df)
    run_knn_on_small_new_cases(30, k, train_df, test_df)
    run_knn_on_big_new_cases(400, k, train_df, test_df)
    run_knn_on_colour(0, k, train_df, test_df)
    run_knn_on_colour(1, k, train_df, test_df)
    run_knn_on_colour(2, k, train_df, test_df)
    run_knn_on_colour(3, k, train_df, test_df)

    # start_date = datetime(2021, 1, 20)
    # end_date = datetime(2021, 3, 20)
    # run_knn_on_dates(k, train_df, full_test_df, start_date, end_date, best_columns)
    #
    # start_date = datetime(2021, 3, 20)
    # end_date = datetime(2021, 5, 20)
    # run_knn_on_dates(k, train_df, full_test_df, start_date, end_date, best_columns)
    #
    # start_date = datetime(2021, 5, 20)
    # end_date = datetime(2021, 7, 20)
    # run_knn_on_dates(k, train_df, full_test_df, start_date, end_date, best_columns)
    #
    # start_date = datetime(2021, 7, 20)
    # end_date = datetime(2021, 9, 11)
    # run_knn_on_dates(k, train_df, full_test_df, start_date, end_date, best_columns)


def investigate_corona_df(corona_df):
    plt.plot(corona_df[corona_df['City_Code'] == 5000]['Date'].values, corona_df[corona_df['City_Code'] == 5000]['today_verified_cases'].values, label="TLV")
    plt.plot(corona_df[corona_df['City_Code'] == 4000]['Date'].values, corona_df[corona_df['City_Code'] == 4000]['today_verified_cases'].values, label="HAIFA")
    plt.plot(corona_df[corona_df['City_Code'] == 6100]['Date'].values, corona_df[corona_df['City_Code'] == 6100]['today_verified_cases'].values, label="BNEI BRAK")
    plt.axhline(corona_df[corona_df['City_Code'] == 5000]['today_verified_cases'].values.mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(corona_df[corona_df['City_Code'] == 5000]['today_verified_cases'].values.std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('New Cases')
    plt.show()

    plt.plot(corona_df[corona_df['City_Code'] == 5000]['Date'].values, corona_df[corona_df['City_Code'] == 5000]['today_verified_cases_smoothed'].values, label="TLV")
    plt.plot(corona_df[corona_df['City_Code'] == 4000]['Date'].values, corona_df[corona_df['City_Code'] == 4000]['today_verified_cases_smoothed'].values, label="HAIFA")
    plt.plot(corona_df[corona_df['City_Code'] == 6100]['Date'].values, corona_df[corona_df['City_Code'] == 6100]['today_verified_cases_smoothed'].values, label="BNEI BRAK")
    plt.xlabel('#Sample')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('New Cases smooth')
    plt.show()

    params = DataParams()

    best_columns = ['vaccinated_dose_3_total', 'dose_3_in_last_2_week', 'verified_cases_7_days_ago', 'City_Code',
                    'verified_cases_14_days_ago', 'today_verified_cases', 'colour']

    corona_df2 = corona_df[best_columns]
    train_df, test_df = train_test_split(corona_df2, test_size=params.split_test_size, random_state=params.split_random_state)
    knn_output = run_knn(6, train_df, test_df)
    print(f'without normalization: {knn_output}')

    corona_df['today_verified_cases'] = corona_df['today_verified_cases_smoothed']
    corona_df = corona_df[best_columns]
    train_df, test_df = train_test_split(corona_df, test_size=params.split_test_size, random_state=params.split_random_state)
    knn_output = run_knn(6, train_df, test_df)
    print(f'with normalization: {knn_output}')


if __name__ == "__main__":
    # play_knn()
    # data = pd.read_csv('train_df.csv')
    # data.sort_values(by=['City_Code', 'Date'], inplace=True)
    corona_df = pd.read_csv('../Preprocess/corona_df.csv')
    # investigate_corona_df(corona_df)

    train_df = pd.read_csv('../Preprocess/train_df.csv')
    full_test_df = pd.read_csv('../Preprocess/test_df.csv')
    # investigate_corona_df(train_df)
    # investigate_corona_df(full_test_df)

    best_columns = ['vaccinated_dose_3_total', 'dose_3_in_last_2_week', 'verified_cases_7_days_ago', 'City_Code',
                    'verified_cases_14_days_ago', 'colour'] + [params.Y] # 'colour' is not realy part of the best columns

    # best_columns = ['Cumulative_verified_cases', 'verified_cases_14_days_ago', 'final_score',
    #                 'verified_cases_8_days_ago', 'dose_1_in_last_2_week', 'Cumulated_deaths',
    #                 'verified_cases_9_days_ago', 'verified_cases_3_days_ago', 'dose_3_in_last_2_week',
    #                 'verified_cases_13_days_ago', 'City_Code', 'verified_cases_10_days_ago'] + [params.Y]

    # data = preprocess(data)
    # model, X_test, y_test = init_model(data)
    train_df = train_df[best_columns]
    test_df = full_test_df[best_columns]
    # acc = run_knn(6, train_df, test_df)
    # print(f'knn accuracy: {acc}')
    # exit(0)
    # plot_graphs(6, train_df, test_df)
    # experiment_k(train_df)
    # experiment_param(train_df)
    # experiment_features(train_df, cols)
    experiment_subset_data(6, train_df, test_df, full_test_df, best_columns)
