from typing import Optional, Callable, List

import os
import glob
import os.path as osp

import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar, extract_zip)
from torch_geometric.utils import remove_isolated_nodes

"""
This is a local copy of MalNetTiny class from PyG
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/malnet_tiny.py

TODO: Delete and use PyG's version once it is part of a released version.
At the time of writing this class is in the main PyG github branch but is not
 included in the current latest released version 2.0.2.
"""

class MalNetTiny(InMemoryDataset):
    r"""The MalNet Tiny dataset from the
    `"A Large-Scale Database for Graph Representation Learning"
    <https://openreview.net/pdf?id=1xDTDk3XPW>`_ paper.
    :class:`MalNetTiny` contains 5,000 malicious and benign software function
    call graphs across 5 different types. Each graph contains at most 5k nodes.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'http://malnet.cc.gatech.edu/graph-data/malnet-graphs-tiny.tar.gz'
    # 70/10/20 train, val, test split by type
    split_url = 'http://malnet.cc.gatech.edu/split-info/split_info_tiny.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        folders = ['addisplay', 'adware', 'benign', 'downloader', 'trojan']
        return [osp.join('malnet-graphs-tiny', folder) for folder in folders]

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt', 'split_dict.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_tar(path, self.raw_dir)
        os.unlink(path)
        path = download_url(self.split_url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        data_list = []
        split_dict = {'train': [], 'valid': [], 'test': []}

        parse = lambda f: set([x.split('/')[-1]
                               for x in f.read().split('\n')[:-1]])  # -1 for empty line at EOF
        split_dir = osp.join(self.raw_dir, 'split_info_tiny', 'type')
        with open(osp.join(split_dir, 'train.txt'), 'r') as f:
            train_names = parse(f)
            assert len(train_names) == 3500
        with open(osp.join(split_dir, 'val.txt'), 'r') as f:
            val_names = parse(f)
            assert len(val_names) == 500
        with open(osp.join(split_dir, 'test.txt'), 'r') as f:
            test_names = parse(f)
            assert len(test_names) == 1000

        for y, raw_path in enumerate(self.raw_paths):
            raw_path = osp.join(raw_path, os.listdir(raw_path)[0])
            filenames = glob.glob(osp.join(raw_path, '*.edgelist'))

            for filename in filenames:
                with open(filename, 'r') as f:
                    edges = f.read().split('\n')[5:-1]
                edge_index = [[int(s) for s in edge.split()] for edge in edges]
                edge_index = torch.tensor(edge_index).t().contiguous()
                # Remove isolated nodes, including those with only a self-loop
                edge_index = remove_isolated_nodes(edge_index)[0]
                num_nodes = int(edge_index.max()) + 1
                data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)
                data_list.append(data)

                ind = len(data_list) - 1
                graph_id = osp.splitext(osp.basename(filename))[0]
                if graph_id in train_names:
                    split_dict['train'].append(ind)
                elif graph_id in val_names:
                    split_dict['valid'].append(ind)
                elif graph_id in test_names:
                    split_dict['test'].append(ind)
                else:
                    raise ValueError(f'No split assignment for "{graph_id}".')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])

    def get_idx_split(self):
        return torch.load(self.processed_paths[1])
