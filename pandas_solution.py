import pandas as pd
import numpy as np
import re

class Preprocess:
    def __init__(self):
        self._df_mean=None
        self._df_std=None
        

    def read_tsv(self,path='D:/Job/CV/summer2019/jooble/data/test.tsv'):
        df_chunk = pd.read_csv(path, sep='\t', chunksize=100000)
        chunk_list = [] 
        for chunk in df_chunk:  
                chunk_list.append(chunk)
        df = pd.concat(chunk_list)
        df.features = df.features.str.split(',')
        df.features = df.features.apply(lambda x: list(map(int, x)))
        return df
    
    def _get_codes(self, features):
        codes_parameters = dict()
        for item in features:
            codes_parameters[item[0]] = len(item[1:])
        return codes_parameters
    
    def _code(self, x):
        return x[0]

    def _create_col_names(self, code):
        return ['feature_{}_{}'.format(code[0], index) for index in range(1, code[1]+1)]
    
    def _create_columns(self, df):
        codes_parameters = self._get_codes(df.features.values)
        df['features_code'] = df.features.map(self._code)
        df.features = df.features.apply(lambda x: x[1:])
        for code in codes_parameters.items():
            columns = self._create_col_names(code)
            for index in range(0, code[1]):
#             df[columns] = df[df['features_code']==code[0]].features.apply(pd.Series, index=columns)
                df[columns[index]] = df[df['features_code']==code[0]].features.map(lambda x : x[index])

        df = df.drop(['features','features_code'], axis=1)

        return df
    
    def _max_feature_abs_mean_diff(self, df, mean):
        df['max_feature_index'] = df.T.iloc[1:,:].idxmax(axis=0, skipna=True)
        df['max_feature_abs_mean_diff'] = df.apply(lambda x: (x[x['max_feature_index']] - mean[x['max_feature_index']]), axis=1).abs()
        df['max_feature_index'] = df['max_feature_index'].map(lambda x: re.findall(r'\d+', x)[1])
        return df
    
    def _normalizer(self,norm):
        if norm=="z_score":
            return self._z_score
        else:
            print('{} normalization was not implemented'.format(norm))
            return None
    
    def _z_score(self, df, df_mean, df_std):
        return (df-df_mean)/df_std
    
    def _mean(self, df):
        return df.iloc[:, 1:].mean()
    
    def _std(self, df):
        return df.iloc[:, 1:].std(skipna=True)
    
    def start_preprocessing(self, df):
        df = self._create_columns(df.copy())    
        self._df_mean = self._mean(df.copy())#.mean()
        self._df_std = self._std(df.copy())#df.iloc[:, 1:].std(skipna=True)
        df = self._max_feature_abs_mean_diff(df.copy(), self._df_mean.copy())
        return df    
            
    def normalization(self, df, norm):
        normalizer = self._normalizer(norm) 
        if normalizer!=None:
            df.iloc[:, 1:-2] = normalizer(df.iloc[:, 1:-2], df_mean=self._df_mean, df_std=self._df_std)     
            print('{} normalization finished'.format(norm))
            return df
        return df
    
    def to_tsv(self, df, path):
        df.to_csv(path, sep='\t',index=False, mode='a')