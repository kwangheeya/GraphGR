import os

import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

import pandas as pd

class GroupDataset():

    def __init__(self, dataset):
        self.dataset = dataset
        self.data_dir = os.path.join('data/', dataset)
        self.df_ui = self._load_user_data()
        self.df_gi, self.df_gu, self.df_gu_train = self._load_group_data()
        self.graphdata = self._build_graph_data()

    def get_test_data(self):
        self.graphdata['group', '', 'user'].edge_index = torch.t(torch.from_numpy(self.df_gu.values.astype("int64")))
        
        self.graphdata = T.ToUndirected()(self.graphdata)
        self.graphdata = T.AddSelfLoops()(self.graphdata)
        return self.graphdata

    def _build_graph_data(self):
        
        data = HeteroData()

        data['user'].x = torch.tensor(range(self.n_users))
        data['item'].x = torch.tensor(range(self.n_items))
        data['group'].x = torch.tensor(range(self.n_groups))

        data['user', '', 'item'].edge_index = torch.t(torch.from_numpy(self.df_ui.values.astype("int64")))
        data['group', '', 'item'].edge_index = torch.t(torch.from_numpy(self.df_gi.values.astype("int64")))
        data['group', '', 'user'].edge_index = torch.t(torch.from_numpy(self.df_gu_train.values.astype("int64")))

        #train label
        data['user'].y = torch.zeros(self.n_users, self.n_items)
        data['group'].y = torch.zeros(self.n_groups, self.n_items)

        data['user'].y[torch.t(data['user', '', 'item'].edge_index[0,:]), torch.t(data['user', '', 'item'].edge_index[1,:])] =1
        data['group'].y[torch.t(data['group', '', 'item'].edge_index[0,:]), torch.t(data['group', '', 'item'].edge_index[1,:])] =1

        #validation label
        data['group'].val_y = torch.zeros(self.n_groups, self.n_items)
        val_gi_tensor = torch.from_numpy(self.val_df_gi.values.astype("int64"))
        data['group'].val_y[val_gi_tensor[:,0], val_gi_tensor[:,1]] = 1

        #test label
        data['group'].test_y = torch.zeros(self.n_groups, self.n_items)
        test_gi_tensor = torch.from_numpy(self.test_df_gi.values.astype("int64"))
        data['group'].test_y[test_gi_tensor[:,0], test_gi_tensor[:,1]] = 1

        data = T.ToUndirected()(data)
        data = T.AddSelfLoops()(data)

        return data

    def _load_group_data(self):
        """ load training group-item interactions as a sparse matrix and user-group memberships """
        path_gu = os.path.join(self.data_dir, 'group_users.csv')
        df_gu = pd.read_csv(path_gu).astype(int)  # load user-group memberships.
        self.n_groups = len(df_gu['group'].unique())
        print("\t# total groups: ", self.n_groups)

        path_gi = os.path.join(self.data_dir, 'train_gi.csv')
        df_gi = pd.read_csv(path_gi)  # load training group-item interactions.
        self.n_train_groups = len(df_gi['group'].unique())        

        val_path_gi = os.path.join(self.data_dir, 'val_gi.csv')
        self.val_df_gi = pd.read_csv(val_path_gi)  # load valid group-item interactions.

        test_path_gi = os.path.join(self.data_dir, 'test_gi.csv')
        self.test_df_gi = pd.read_csv(test_path_gi)  # load valid group-item interactions.  


        df_gu_train = df_gu[df_gu.group.isin(df_gi['group'].unique())]
        self.max_group_size = df_gu_train.groupby('group').size().max()  # max group size denoted by G

        assert len(df_gu_train['group'].unique()) == self.n_train_groups

        print("\t# training groups: {}, # max train group size: {}".format(self.n_train_groups, self.max_group_size))

        return df_gi, df_gu, df_gu_train

    def _load_user_data(self):
        """ load user-item interactions of all users that appear in training groups, as a sparse matrix """
        train_path_ui = os.path.join(self.data_dir, 'train_ui.csv')
        df_train_ui = pd.read_csv(train_path_ui)

        # include users from the (fold-in item set) of validation and test sets of user-item data.
        val_path_ui = os.path.join(self.data_dir, 'val_ui_tr.csv')
        df_val_ui = pd.read_csv(val_path_ui)

        test_path_ui = os.path.join(self.data_dir, 'test_ui_tr.csv')
        df_test_ui = pd.read_csv(test_path_ui)
        df_ui = pd.concat([df_train_ui, df_val_ui, df_test_ui])

        self.n_users, self.n_items = df_ui['user'].max() + 1, df_ui['item'].max() + 1
        print('\t# users: ', self.n_users, '\n\t# items: ', self.n_items)
        return df_ui

