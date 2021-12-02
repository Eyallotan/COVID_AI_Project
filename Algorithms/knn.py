from random import sample
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import math

from Preprocess.utils import DataParams

K=6
params = DataParams()
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

best_columns = ['vaccinated_dose_3_total', 'dose_3_in_last_2_week', 'verified_cases_7_days_ago', 'City_Code',
                'verified_cases_14_days_ago', 'colour'] + [params.Y]

tlv_code = 5000
haifa_code = 4000
min = '2021-01-20'
max = '2021-09-11'


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


def run_knn_with_split(k, examples, weights='uniform'):
    tscv = TimeSeriesSplit(n_splits=20)
    sum = 0
    divider = 0
    for train_index, test_index in tscv.split(examples):
        train_examples, test_examples = examples.iloc[train_index], examples.iloc[test_index]
        acc = run_knn(k, train_examples, test_examples, weights=weights)
        if acc['acc'] > 0:
            sum += acc['acc']
            divider = divider + 1
        print(f'accuracy: {acc}')

    avg_accuracy = sum / divider # tscv.n_splits
    print(f'avg accuracy: {avg_accuracy}')

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
    mae = mean_absolute_error(y_test, y_predicted)
    mape = mean_absolute_percentage_error(y_test, y_predicted)

    # my_acc = 1 - ((y_test - y_predicted)** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()

    return {'acc': acc, 'mae': mae, 'test_mean': y_test.mean()} # , 'y_test': y_test, 'y_predicted': y_predicted, 'mse': None


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


def experiment_k_kfold(examples):
    kf = KFold(n_splits=5)#, shuffle=True, random_state=307916502)
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


def experiment_k(train, test):
    train = train[best_columns]
    test = test[best_columns]
    K = range(3, 25)
    accuracies = []
    for k in K:
        knn_output = run_knn(k, train, test)
        accuracies.append(knn_output['acc'])

    plt.plot(K, accuracies)
    plt.xlabel('K')
    plt.ylabel('accuracy')
    plt.show()


def experiment_weights(train, test):
    print('testing best weights')
    print('uniform:')
    run_knn_best_columns(K, train, test, 'uniform')
    print('distance:')
    run_knn_best_columns(K, train, test, 'distance')


def run_knn_on_small_cities(population, k, train_df, test_df):
    population_df = pd.read_csv('../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])

    test_df = test_df[test_df['population'] <= population]
    test_df.drop(['population'], axis=1, inplace=True)

    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mae = knn_output['mae']
    test_mean = knn_output['test_mean']
    print(f'knn_on_small_cities accuracy: {acc}, mae: {mae}, y_test_mean={test_mean}, population={population}')


def run_knn_on_big_cities(population, k, train_df, test_df):
    population_df = pd.read_csv('../Resources/population_table.csv')
    population_df = population_df[['City_Code', 'population']].drop_duplicates()

    test_df = test_df.merge(population_df, on=["City_Code"])

    test_df = test_df[test_df['population'] >= population]
    test_df.drop(['population'], axis=1, inplace=True)

    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mae = knn_output['mae']
    test_mean = knn_output['test_mean']
    print(f'knn_on_big_cities accuracy: {acc}, mae: {mae}, y_test_mean={test_mean}, population={population}')


def run_knn_on_small_new_cases(new_cases, k, train_df, test_df):
    test_df = test_df[test_df[params.Y] <= new_cases]
    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mae = knn_output['mae']
    test_mean = knn_output['test_mean']
    print(f'knn_on_small_new_cases accuracy: {acc}, mae: {mae}, y_test_mean={test_mean}, new_cases {new_cases}')


def run_knn_on_big_new_cases(new_cases, k, train_df, test_df):
    test_df = test_df[test_df[params.Y] >= new_cases]

    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mae = knn_output['mae']
    test_mean = knn_output['test_mean']
    print(f'knn_on_big_new_cases accuracy: {acc}, mae: {mae}, y_test_mean={test_mean}, new_cases {new_cases}')


def run_knn_on_colour(colour, k, train_df, test_df):
    test_df = test_df[test_df['colour'] == colour]

    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mae = knn_output['mae']
    test_mean = knn_output['test_mean']
    print(f'knn_on_colour accuracy: {acc}, mae: {mae}, y_test_mean={test_mean}, colour {colour}')


def run_knn_on_dates(k, train_df, test_df, start_date, end_date, best_columns):
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df = test_df[(test_df['Date'] >= start_date) & (test_df['Date'] <= end_date)]
    test_df = test_df[best_columns]
    knn_output = run_knn(k, train_df, test_df)
    acc = knn_output['acc']
    mae = knn_output['mae']
    test_mean = knn_output['test_mean']
    my_acc = knn_output['my_acc']
    print(f'knn_on_big_new_cases accuracy: {acc}, mae: {mae}, y_test_mean={test_mean}, my_acc={my_acc}, start_date {start_date}, , end_date {end_date}')


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


def investigate_corona_df(corona_df, train_df, test_df):
    plt.plot(corona_df[corona_df['City_Code'] == 5000]['Date'].values, corona_df[corona_df['City_Code'] == 5000]['today_verified_cases'].values, label="TLV")
    plt.plot(corona_df[corona_df['City_Code'] == 4000]['Date'].values, corona_df[corona_df['City_Code'] == 4000]['today_verified_cases'].values, label="HAIFA")
    plt.plot(corona_df[corona_df['City_Code'] == 6100]['Date'].values, corona_df[corona_df['City_Code'] == 6100]['today_verified_cases'].values, label="BNEI BRAK")
    plt.plot(corona_df[corona_df['City_Code'] == 3730]['Date'].values, corona_df[corona_df['City_Code'] == 3730]['today_verified_cases'].values, label="Givat Zeev")
    plt.plot(corona_df[corona_df['City_Code'] == 53]['Date'].values, corona_df[corona_df['City_Code'] == 53]['today_verified_cases'].values, label="Atlit")
    plt.axhline(corona_df[corona_df['City_Code'] == 5000]['today_verified_cases'].values.mean(), color='r', alpha=0.2, linestyle='--')
    plt.axhline(corona_df[corona_df['City_Code'] == 5000]['today_verified_cases'].values.std(), color='b', alpha=0.2, linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('New Cases')
    plt.show()

    plt.plot(corona_df[corona_df['City_Code'] == 5000]['Date'].values, corona_df[corona_df['City_Code'] == 5000]['today_verified_cases_smoothed'].values, label="TLV")
    plt.plot(corona_df[corona_df['City_Code'] == 4000]['Date'].values, corona_df[corona_df['City_Code'] == 4000]['today_verified_cases_smoothed'].values, label="HAIFA")
    plt.plot(corona_df[corona_df['City_Code'] == 6100]['Date'].values, corona_df[corona_df['City_Code'] == 6100]['today_verified_cases_smoothed'].values, label="BNEI BRAK")
    plt.plot(corona_df[corona_df['City_Code'] == 3730]['Date'].values, corona_df[corona_df['City_Code'] == 3730]['today_verified_cases_smoothed'].values, label="Givat Zeev")
    plt.plot(corona_df[corona_df['City_Code'] == 53]['Date'].values, corona_df[corona_df['City_Code'] == 53]['today_verified_cases_smoothed'].values, label="Atlit")
    plt.xlabel('Date')
    plt.ylabel('daily new cases')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.title('New Cases smooth')
    plt.show()

    print('not smoothed')
    run_knn_best_columns(K, train_df, test_df)
    # corona_df2 = corona_df[best_columns]
    # train_df, test_df = train_test_split(corona_df2, test_size=params.split_test_size, random_state=params.split_random_state)
    # knn_output = run_knn(6, train_df, test_df)
    # print(f'without normalization: {knn_output}')

    train_df['today_verified_cases'] = train_df['today_verified_cases_smoothed']
    test_df['today_verified_cases'] = test_df['today_verified_cases_smoothed']
    print('smoothed')
    run_knn_best_columns(K, train_df, test_df)
    exit(0)

    # corona_df = corona_df[best_columns]
    # train_df, test_df = train_test_split(corona_df, test_size=params.split_test_size, random_state=params.split_random_state)
    # knn_output = run_knn(6, train_df, test_df)
    # print(f'with normalization: {knn_output}')


def run_knn_best_columns(k, train, test, weights='uniform'):
    train_df = train[best_columns]
    test_df = test[best_columns]
    knn_output = run_knn(k, train_df, test_df, weights)
    print_results(knn_output)


def print_results(knn_output):
    acc = knn_output['acc']
    mae = knn_output['mae']
    test_mean = knn_output['test_mean']
    print(f'knn_on_small_cities accuracy: R2:{acc}, mae: {mae}, y_test_mean={test_mean}')


if __name__ == "__main__":
    # Load data
    corona_df = pd.read_csv('../Preprocess/corona_df.csv')
    train_df = pd.read_csv('../Preprocess/train_df.csv')
    test_df = pd.read_csv('../Preprocess/test_df.csv')

    corona_df['Date'] = pd.to_datetime(corona_df['Date'])
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    test_df['Date'] = pd.to_datetime(test_df['Date'])

    corona_df.sort_values(by=['Date'], inplace=True)
    train_df.sort_values(by=['Date'], inplace=True)
    test_df.sort_values(by=['Date'], inplace=True)

    ################
    # run_knn_best_columns(K, train_df, full_test_df)
    ################
    # experiment_k(train_df, test_df)
    # exit(0)
    ################
    experiment_weights(train_df, test_df)
    exit(0)
    ################
    # investigate_corona_df(corona_df, train_df, test_df)
    ################
    corona_df = corona_df[best_columns]
    run_knn_with_split(6, corona_df)
    ################

    # plot_graphs(6, train_df, test_df)
    experiment_features(train_df, cols)

    # train_df = train_df.drop(['City_Name', 'Date'], axis=1)
    # test_df = full_test_df.drop(['City_Name', 'Date'], axis=1)


    experiment_subset_data(6, train_df, test_df, test_df, best_columns)
