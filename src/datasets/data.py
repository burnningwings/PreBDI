from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets import utils

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class BridgeData(BaseData):
    """
    Dataset class for Bridge datasets
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        #self.set_num_processes(n_proc=n_proc)

        self.config = config

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        # self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        # if limit_size is not None:
        #     if limit_size > 1:
        #         limit_size = int(limit_size)
        #     else:  # interpret as proportion if in (0, 1]
        #         limit_size = int(limit_size * len(self.all_IDs))
        #     self.all_IDs = self.all_IDs[:limit_size]
        #     self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        # self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.pkl')]
        if len(input_paths) == 0:
            raise Exception("No .pkl files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):

        # Every row of the returned df corresponds to a sample;
        # every column is a pd.Series indexed by timestamp and corresponds to a different dimension (feature)
        time_len = 128
        data = utils.load_pkl(filepath)
        if self.config['task'] == 'regression':
            df = data["data"]
            labels_df = data["label"]
        elif self.config['task'] == 'classification' and 'clabel' in data:
            df = data["data"]
            labels_df = data["clabel"]
        elif self.config['task'] == 'classification' and 'domain' in data:
            df = data["data"]
            labels_df = data["label"]
            domain = data["domain"]
            self.domain = domain
        elif self.config['task'] == 'classification':
            df = data["data"]
            labels_df = data["label"]
        else:  # e.g. imputation
            df = data["data"]
            labels_df = data["label"]

        # 扩大这个长度为L
        self.max_seq_len = time_len
        # df[..., [0]] *= 1e7
        # return df * 1e7, labels_df, tempo_df

        if self.config['stiffness_informed']:
            df[..., 0] = df[..., 0] * 15 / df[..., 1]
        return df, labels_df

class BridgeDomainData(BaseData):

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.config = config
        # self.data_mode = "eval"
        data_list = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df = data_list[0]
        self.labels_df = data_list[1]
        if len(data_list) == 4:
            self.data_mode = "train"
            self.real_all_df = data_list[2]
            self.real_labels_df = data_list[3]
        self.feature_df = self.all_df

    def load_all(self, root_dir, file_list=None, pattern=None):

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.pkl')]
        if len(input_paths) == 0:
            raise Exception("No .pkl files found using pattern: '{}'".format(pattern))

        data_list = self.load_single(input_paths[0])  # a single file contains dataset

        return data_list

    def load_single(self, filepath):
        time_len = 128
        data = utils.load_pkl(filepath)
        df = data["data"]
        labels_df = data["label"]
        domain = data["domain"]
        self.domain = domain
        self.max_seq_len = time_len
        if "real_data" in data:
            real_df = data["real_data"]
            real_label_df = data["real_label"]
            real_domain = data["real_domain"]
            self.real_domain = real_domain
            return df, labels_df, real_df, real_label_df
        else:
            return df, labels_df



data_factory = {
                'bridge': BridgeData,
                'domain': BridgeDomainData}
