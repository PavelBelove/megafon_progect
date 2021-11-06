import pandas as pd
import numpy as np
import luigi

from functions import reduce_mem_usage


class PredictProbability(luigi.Task):

    data_test = luigi.Parameter()
    data_features = luigi.Parameter()


    def run(self):

        # load data
        test = pd.read_csv(self.data_test)
        data_features = dd.read_csv(self.data_features, sep ='\t')

        # merge data
        ids = np.unique(test['id'])
        data_features = data_features[data_features['id'].isin(ids)]
        data_features = data_features.compute()
        # удалим признаки с единственным значением
        df_nunique = data_features.apply(lambda x: x.nunique(dropna=False))
        const = df_nunique[df_nunique ==1].index.tolist()
        data_features = data_features.drop(columns = const)
        # функция сжатия данных
        data_features = reduce_mem_usage(data_features)

        # сортируем и мерджим
        data_test = data_test.sort_values(by="buy_time")
        forward = data_features.sort_values(by="buy_time")

        valid = pd.merge_asof(data_features, data_test,  on='buy_time', by='id', direction ='nearest')

        features = [f for f in valid.columns if f not in ['id']]

        # load models

        with open('fs_pipe.pkl', 'rb') as model_file:
            fs_pipe = pickle.load(model_file)

        with open('model.pkl', 'rb') as f:
            model = pickle.load(model_file)

            # transform data
        X_valid = fs_pipe.transform(X_valid)

        # predict
        answers_test = test

        answers_test['target'] = model.predict_proba(X_valid)[:, 1]

        # save result

        answers_test.to_csv('answers_test.csv', float_format='%20f', index=False, encoding='utf8',sep=',')


    def output(self):
        return luigi.LocalTarget('answers_test.csv')


if __name__ == '__main__':
    luigi.build([PredictProbability("data/data_test.csv", "data/features.csv")])
