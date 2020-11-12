import pandas as pd


def read_data(number):
    s = str(number)
    if len(s) == 1:
        s = '0' + s
    train = pd.read_csv(f'lab05_dt/resources/{s}_train.csv')
    test = pd.read_csv(f'lab05_dt/resources/{s}_test.csv')
    x_train, y_train = train.drop(['y'], axis=1), train['y']
    x_test, y_test = test.drop(['y'], axis=1), test['y']
    return x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy()