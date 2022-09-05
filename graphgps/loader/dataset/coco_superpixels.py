import os
import os.path as osp
import shutil
import pickle

import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)


class COCOSuperpixels(InMemoryDataset):
    r"""The COCOSuperpixels dataset which contains image superpixels and a semantic segmentation label
    for each node superpixel.
    
    Construction and Preparation:
    - The superpixels are extracted in a similar fashion as the MNIST and CIFAR10 superpixels. 
    - In COCOSuperpixels, the number of superpixel nodes <=500. (Note that it was <=75 for MNIST and
    <=150 for CIFAR10.)
    - The labeling of each superpixel node is done with the same value of the original pixel ground
    truth  that is on the mean coord of the superpixel node
    
    - Based on the COCO 2017 dataset. Original
    source `here<https://cocodataset.org>`_.
    
    num_classes = 81

    COCO categories: 
    person bicycle car motorcycle airplane bus train truck boat traffic light fire hydrant stop
    sign parking meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack
    umbrella handbag tie suitcase frisbee skis snowboard sports ball kite baseball bat baseball
    glove skateboard surfboard tennis racket bottle wine glass cup fork knife spoon bowl banana
    apple sandwich orange broccoli carrot hot dog pizza donut cake chair couch potted plant bed
    dining table toilet tv laptop mouse remote keyboard cell phone microwave oven toaster sink
    refrigerator book clock vase scissors teddy bear hair drier toothbrush
    
    Splitting:
    - In the original image dataset there are only train and val splitting.
    - For COCOSuperpixels, we maintain the original val split as the new test split, and divide the
    original train split into new val split and train split. The resultant train, val and test split 
    have 113286, 5000, 5000 superpixel graphs.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): Option to select the graph construction format.
            If :obj: `"edge_wt_only_coord"`, the graphs are 8-nn graphs with the edge weights computed based on
            only spatial coordinates of superpixel nodes.
            If :obj: `"edge_wt_coord_feat"`, the graphs are 8-nn graphs with the edge weights computed based on
            combination of spatial coordinates and feature values of superpixel nodes.
            If :obj: `"edge_wt_region_boundary"`, the graphs region boundary graphs where two regions (i.e. 
            superpixel nodes) have an edge between them if they share a boundary in the original image.
            (default: :obj:`"edge_wt_region_boundary"`)
        slic_compactness (int, optional): Option to select compactness of slic that was used for superpixels
            (:obj:`10`, :obj:`30`). (default: :obj:`30`)
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
    
    url = {
        10: {
        'edge_wt_only_coord': 'https://www.dropbox.com/s/prqizdep8gk0ndk/coco_superpixels_edge_wt_only_coord.zip?dl=1',
        'edge_wt_coord_feat': 'https://www.dropbox.com/s/zftoyln1pkcshcg/coco_superpixels_edge_wt_coord_feat.zip?dl=1',
        'edge_wt_region_boundary': 'https://www.dropbox.com/s/fhihfcyx2y978u8/coco_superpixels_edge_wt_region_boundary.zip?dl=1'
        },
        30: {
        'edge_wt_only_coord': 'https://www.dropbox.com/s/hrbfkxmc5z9lsaz/coco_superpixels_edge_wt_only_coord.zip?dl=1',
        'edge_wt_coord_feat': 'https://www.dropbox.com/s/4rfa2d5ij1gfu9b/coco_superpixels_edge_wt_coord_feat.zip?dl=1',
        'edge_wt_region_boundary': 'https://www.dropbox.com/s/r6ihg1f4pmyjjy0/coco_superpixels_edge_wt_region_boundary.zip?dl=1'
        }
    }

    def __init__(self, root, name='edge_wt_region_boundary', slic_compactness=30, split='train',
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.slic_compactness = slic_compactness
        assert split in ['train', 'val', 'test']
        assert name in ['edge_wt_only_coord', 'edge_wt_coord_feat', 'edge_wt_region_boundary']
        assert slic_compactness in [10, 30]
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    
    @property
    def raw_file_names(self):
        return ['train.pickle', 'val.pickle', 'test.pickle']

    @property
    def raw_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness),
                        self.name,
                        'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root,
                        'slic_compactness_' + str(self.slic_compactness),
                        self.name,
                        'processed')
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url[self.slic_compactness][self.name], self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'coco_superpixels_' + self.name), self.raw_dir)
        os.unlink(path)
        
    def label_remap(self):
        # Util function to remap the labels as the original label idxs are not contiguous
        
        original_label_ix = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 
                             11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                             23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                             37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48,
                             49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 
                             60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74,
                             75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86,
                             87, 88, 89, 90]
        label_map = {}
        for i, key in enumerate(original_label_ix):
            label_map[key] = i
        
        return label_map
        
    def process(self):
        label_map = self.label_remap()
        for split in ['train', 'val', 'test']:
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                graphs = pickle.load(f)
            
            indices = range(len(graphs))

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                graph = graphs[idx] 
                
                """
                Each `graph` is a tuple (x, edge_attr, edge_index, y)
                    Shape of x : [num_nodes, 14]
                    Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
                    Shape of edge_index : [2, num_edges]
                    Shape of y : [num_nodes]
                """
                
                x = graph[0].to(torch.float)
                edge_attr = graph[1].to(torch.float)
                edge_index = graph[2]
                y = torch.LongTensor(graph[3])
                
                # Label remapping. See self.label_remap() func
                for i, label in enumerate(y):
                    y[i] = label_map[label.item()]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
