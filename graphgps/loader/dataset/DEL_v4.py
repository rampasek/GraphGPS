import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from chainer_chemistry.dataset.splitters.random_splitter import RandomSplitter
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from torch_geometric.graphgym.loader import set_dataset_attr

class DEL_Dataset(InMemoryDataset):
    def __init__(self, root, smiles2graph=smiles2graph, transform=None, pre_transform=None):
        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'DEL')
        self.train_graph_index = []
        self.val_graph_index = []
        self.test_graph_index = []
        self.data_list = []

        super(DEL_Dataset, self).__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    

    def process(self):
        df = pd.read_csv(osp.join(self.original_root, 'DEL_v4.csv'))
        
        train_df = df[df['Split'] == 'train']
        valid_df = df[df['Split'] == 'valid']
        test_df = df[df['Split'] == 'test']
        
        self.data, self.slices = self.process_data(df)
        
        shuffle_split = {'train': train_df.index.tolist(), 'val': valid_df.index.tolist(), 'test': test_df.index.tolist()}
       
        torch.save(shuffle_split,
                   osp.join(self.folder, f'DEL_shuffle_split_dict.pt'))
        
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        splits = [train_df.index.tolist(), valid_df.index.tolist(), test_df.index.tolist()]
        for split_name, split_index in zip(split_names, splits):
            set_dataset_attr(self, split_name, split_index, len(split_index))
        #print(self.data)
        print("DEL_v4.py", self.data.val_graph_index[:10])
        
        

        processed_data = (self.data, self.slices)
        torch.save(processed_data, self.processed_paths[0])

        #train_data = self.process_data(train_df)

        #valid_data = self.process_data(valid_df)

        #test_data = self.process_data(test_df)

        print('Saving...')
        print(self.processed_paths)

    def process_data(self, data_df):
        
        smiles_list = data_df['SMILES'].tolist()

        print('Converting SMILES strings into graphs...')
        
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
            data.y = torch.tensor([data_df['Activity'].iloc[i]])

            self.data_list.append(data)

        if self.pre_transform is not None:
            self.data_list = [self.pre_transform(data) for data in self.data_list]

        return self.collate(self.data_list)
    
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def get(self, idx):
        return self.data_list[idx]
                   
    def get_idx_split(self):
        """ Get dataset splits.

        Args:
            split_name: Split type: 'shuffle', 'num-atoms'

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(
            self.folder,
            f'DEL_shuffle_split_dict.pt'
        )
        
        split_dict = replace_numpy_with_torchtensor(torch.load(split_file))
        return split_dict
if __name__=='__main__':
    
    root = './DEL'
    dataset = DEL_Dataset(root)

    #for split_name in 'train_graph_index', 'val_graph_index', 'test_graph_index':
        #if not hasattr(dataset.data, split_name):
            #raise ValueError(f"Missing '{split_name}' for standard split")

    print(len(dataset))  
    print(dataset[0])
    print(dataset.get(0))
    train_idx, val_idx, test_idx = dataset.get_idx_split()

    print("Train Index:", train_idx)
    print("Validation Index:", val_idx)
    print("Test Index:", test_idx)