import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import re
from urllib.parse import urlparse
from web.cfg import cfg


def ip_exist_two(one_url):
    compile_rule = re.compile(r'(?<![\.\d])(?:\d{1,3}\.){3}\d{1,3}(?![\.\d])')
    match_list = re.findall(compile_rule, one_url)
    if match_list:
        return 1
    else:
        return 0


def url_startwith_https(param):
    if param == 'https':
        return 0
    else:
        return 1


def preprocess_url(csv_file):
    df = pd.read_csv(csv_file, header=None)
    labels = df.iloc[:][1]
    X = list()

    for idx in range(len(labels)):
        ft = list()
        url = df.iloc[idx, 3]
        parse_result = urlparse(url)
        # print(parse_result)

        ft.append(ip_exist_two(url))  # domain为IP
        ft.append(0 if parse_result[0] == 'https' else 1)  # 端口非80、443
        ft.append(0 if len(parse_result[1].split('.')) <= 4 else 1)  # domain超过4个
        ft.append(len(parse_result[2].split('/')))  # path深度

        ft.append(1 if 'login' in url or 'account' in url else 0)  # url中含有account/login
        ft.append(0 if len(url) <= 23 else 1)  # url长度超过23个

        if df.iloc[idx, 1] == 'n':
            df.iloc[idx, 1] = 0
        elif df.iloc[idx, 1] == 'd':
            df.iloc[idx, 1] = 1
        elif df.iloc[idx, 1] == 'p':
            df.iloc[idx, 1] = 2

        X.append(ft)

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(np.array(X))
    y = df[:][1]

    return X, y


if __name__ == '__main__':
    X, y = preprocess_url(cfg['filelist_csv'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    lr = LogisticRegression()
    print('start training Logistic Regression...')
    lr.fit(X_train, y_train)
    print('finish training Logistic Regression...')
    acc = lr.score(X_test, y_test)
    print('Accuracy on test set is %f' % acc)

    print('confusion matrix:')
    print(confusion_matrix(y_test, lr.predict(X_test)))

    f1 = f1_score(y_test, lr.predict(X_test), average='weighted')
    print('F1 score is %f' % f1)
