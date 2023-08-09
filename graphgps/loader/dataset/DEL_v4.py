import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from chainer_chemistry.dataset.splitters.random_splitter import RandomSplitter
from ogb.utils import smiles2graph

class DEL_Dataset(InMemoryDataset):
    def __init__(self, root, smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'DEL')

        super(DEL_Dataset, self).__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __len__(self):
        return len(self.data)

    def process(self):
        df = pd.read_csv(osp.join(self.original_root, 'DEL_v4.csv'))
        train_df, valid_df, test_df = self.split(df, data_column='SMILES', split_type='random')

        train_data = self.process_data(train_df)

        valid_data = self.process_data(valid_df)

        test_data = self.process_data(test_df)

        processed_data = self.collate(train_data + valid_data + test_data)
        print('Saving...')
        torch.save(processed_data, self.processed_paths[0])

    def process_data(self, data_df):
        smiles_list = data_df['SMILES']

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            graph = self.smiles2graph(smiles)

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
            data.edge_attr = torch.tensor(graph['edge_feat'], dtype=torch.long)
            data.x = torch.tensor(graph['node_feat'], dtype=torch.long)
            data.y = torch.Tensor([data_df['Activity'].iloc[i]])

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        return data_list

    def split(self, df: pd.DataFrame, data_column: str, split_type: str):
        if split_type == 'random':
            splitter = RandomSplitter()
        else:
            raise NotImplemented(f'Split type {split_type} is not allowed. Only random splitting is supported!')
        
        train_idx, valid_idx, test_idx = splitter.train_valid_test_split(df, smiles_list=df[data_column],
                                                                         frac_train=0.7, frac_valid=0.1, frac_test=0.2,
                                                                         seed=0, include_chirality=True,
                                                                         return_index=True)
        df['split'] = None
        df.loc[train_idx, 'split'] = 'Train'
        df.loc[valid_idx, 'split'] = 'Valid'
        df.loc[test_idx, 'split'] = 'Test'

        return df[df['split'] == 'Train'], df[df['split'] == 'Valid'], df[df['split'] == 'Test']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def get(self, idx):
        return self.data_list[idx]


root = './DEL'
dataset = DEL_Dataset(root)
print(len(dataset))  
print(dataset[0])
print(dataset.get(0)) 
