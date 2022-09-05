import hashlib
import os.path as osp
import shutil

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url as ogb_download_url
from rdkit.Chem.AllChem import MolFromSmiles
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.graphgym.models.transform import create_link_label
from torch_geometric.utils import to_undirected, negative_sampling
from tqdm import tqdm

from graphgps.utils import negate_edge_index


def cxsmiles_to_mol_with_contact(cxsmiles):
    mol = MolFromSmiles(cxsmiles, sanitize=False)
    num_atoms = mol.GetNumAtoms()
    list_of_contacts = [[]] * num_atoms
    for ii, atom in enumerate(mol.GetAtoms()):
        try:
            this_list = [int(val) for val in atom.GetProp("contact").split(";")]
            list_of_contacts[ii] = this_list
        except:
            pass
    max_count = max(sum(list_of_contacts, []) + [-1]) + 1

    if max_count == 0:
        contact_idx = np.zeros(shape=(0, 2), dtype=int)
    else:
        contact_idx = []
        for this_count in range(max_count):
            this_count_found = []
            for ii in range(num_atoms):
                if this_count in list_of_contacts[ii]:
                    this_count_found.append(ii)
            contact_idx.append(this_count_found)
        contact_idx = np.array(contact_idx, dtype=int)

    return mol, contact_idx


def mol2graph(mol):
    """
    Slightly modified from ogb `smiles2graph`. Takes mol instead of smiles.

    Converts rdkit.Mol string to graph Data object
    :input: rdkit.Mol
    :return: graph object
    """

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def cxsmiles_to_graph_with_contact(cxsmiles):
    mol, contact_idx = cxsmiles_to_mol_with_contact(cxsmiles)
    graph = mol2graph(mol)
    graph["contact_idx"] = contact_idx
    return graph


def custom_structured_negative_sampling(edge_index, num_nodes: int,
                                        num_neg_per_pos: int,
                                        contains_neg_self_loops: bool = True,
                                        return_ik_only: bool = False):
    r"""Customized `torch_geometric.utils.structured_negative_sampling`.

    Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        num_neg_per_pos (int): Number of negative edges to sample from a head
            (source) of each positive edge
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)
        return_ik_only: Instead of :obj:`(i,j,k)` return just :obj:`(i,k)`
            leaving out the original tails of the positive edges.

    :rtype: (LongTensor, LongTensor, LongTensor) or (LongTensor, LongTensor)
    """

    def get_redo_indices(neg_idx, pos_idx):
        """
        Compute indices in `neg_idx` that are invalid because they:
        a) overlap with `neg_idx`, i.e. these are in fact positive edges
        b) are duplicates of the same edge in `neg_idx`
        Args:
            neg_idx (LongTensor): Candidate negative edges encodes as indices in
                a serialized adjacency matrix.
            pos_idx (LongTensor): Positive edges encodes as indices in
                a serialized adjacency matrix.

        Returns:
            LongTensor
        """
        _, unique_ind = np.unique(neg_idx, return_index=True)
        duplicate_mask = np.ones(len(neg_idx), dtype=bool)
        duplicate_mask[unique_ind] = False
        mask = torch.from_numpy(np.logical_or(np.isin(neg_idx, pos_idx),
                                              duplicate_mask)).to(torch.bool)
        return mask.nonzero(as_tuple=False).view(-1)

    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col  # Encodes as the index in a serialized adjacency matrix
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)

    heads = row.unsqueeze(1).repeat(1, num_neg_per_pos).flatten()
    if not return_ik_only:
        tails = col.unsqueeze(1).repeat(1, num_neg_per_pos).flatten()
    rand = torch.randint(num_nodes, (num_neg_per_pos * row.size(0),),
                         dtype=torch.long)
    neg_idx = heads * num_nodes + rand

    # Resample duplicates or sampled negative edges that are actually positive.
    tries_left = 10
    redo = get_redo_indices(neg_idx, pos_idx)
    while redo.numel() > 0 and tries_left > 0:  # pragma: no cover
        tries_left -= 1
        tmp = torch.randint(num_nodes, (redo.size(0),), dtype=torch.long)
        rand[redo] = tmp
        neg_idx = heads * num_nodes + rand
        redo = get_redo_indices(neg_idx, pos_idx)

    # Remove left-over invalid edges.
    if redo.numel() > 0:
        # print(f"> FORCED TO REMOVE {redo.numel()} edges.")
        del_mask = torch.ones(heads.numel(), dtype=torch.bool)
        del_mask[redo] = False
        heads = heads[del_mask]
        rand = rand[del_mask]
        if not return_ik_only:
            tails = tails[del_mask]

    if not return_ik_only:
        return heads, tails, rand
    else:
        return heads, rand


def structured_neg_sampling_transform(data):
    """ Structured negative sampling for link prediction tasks as a transform.

    Sample `num_neg_per_pos` negative edges for each head node of a positive
    edge.

    Args:
        data (torch_geometric.data.Data): Input data object

    Returns: Transformed data object with negative edges + link pred labels
    """
    id_pos = data.edge_index_labeled[:, data.edge_label == 1]  # Positive edge_index
    sampling_out = custom_structured_negative_sampling(
        edge_index=id_pos,
        num_nodes=data.num_nodes,
        num_neg_per_pos=2,
        contains_neg_self_loops=True,
        return_ik_only=True)
    id_neg = torch.stack(sampling_out)

    data.edge_index_labeled = torch.cat([id_pos, id_neg], dim=-1)
    data.edge_label = create_link_label(id_pos, id_neg).int()
    return data


def neg_sampling_transform(data):
    """ Negative sampling for link prediction tasks as a transform.

    Sample `num_neg_samples` random negative edges using PyG method.

    Args:
        data (torch_geometric.data.Data): Input data object

    Returns: Transformed data object with negative edges + link pred labels
    """
    id_pos = data.edge_index_labeled[:, data.edge_label == 1]  # Positive edge_index
    id_neg = negative_sampling(
        edge_index=torch.cat([id_pos, data.edge_index], dim=-1),
        num_nodes=data.num_nodes,
        num_neg_samples=2 * id_pos.shape[1],
        force_undirected=True).long()
    data.edge_index_labeled = torch.cat([id_pos, id_neg], dim=-1)
    data.edge_label = create_link_label(id_pos, id_neg).int()
    return data


def complete_neg_transform(data):
    """ Compute all negative edges for link prediction tasks as a transform.

    Mark all possible edges that are not positive as negative. This will result
    in total (V**2 - V) number of labeled links.

    Args:
        data (torch_geometric.data.Data): Input data object

    Returns: Transformed data object with negative edges + link pred labels
    """
    id_pos = data.edge_index_labeled[:, data.edge_label == 1]  # Positive edge_index
    id_neg = negate_edge_index(
        edge_index=id_pos,
        batch=torch.zeros(data.num_nodes, dtype=torch.long)
    )
    data.edge_index_labeled = torch.cat([id_pos, id_neg], dim=-1)
    data.edge_label = create_link_label(id_pos, id_neg).int()
    # assert len(data.edge_label) == data.edge_index_labeled.shape[1]
    # assert len(data.edge_label) == data.num_nodes ** 2 - data.num_nodes
    # print("POS: ", id_pos, id_pos.shape)
    # print("NEG: ", id_neg, id_neg.shape)
    # print('-' * 80)
    return data


class PygPCQM4Mv2ContactDataset(InMemoryDataset):
    SEED = 42
    VAL_RATIO = 0.05
    TEST_RATIO = 0.05

    def __init__(self, root='dataset', subset='530k',
                 smiles2graph=cxsmiles_to_graph_with_contact,
                 transform=None, pre_transform=None):
        """
        PyG dataset of Contact Map prediction of the PCQM4Mv2 3D conformations.

        This is a link prediction task, with 98% of the links being in the
        negative class (no contact).

        The contacts are determined as any 2 atoms with distance <3.5 Angstrom,
        and graph distance >=5. So the network must learn both the 3D distance
        and the 2D distance.

        Args:
            root (string): Root directory where the dataset should be saved.
            subset (string): Subset specifier: '530k' (default), 'full'
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object.
                * The default cxsmiles_to_graph_with_contact requires rdkit! *
        """

        self.original_root = root
        self.subset = subset
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2-contact', subset)

        self.url = 'https://datasets-public-research.s3.us-east-2.amazonaws.com/PCQM4M/pcqm4m-contact.tsv.gz'
        self.version = 'f7ffb27942145a2e72f6f5f51716d3bc'  # MD5 hash of the intended dataset file

        if subset == 'full':
            self.url_shuffle_split = 'https://www.dropbox.com/s/r3mjzqyulslkyz4/full_shuffle_split_dict.pt?dl=1'
            self.md5_shuffle_split = 'b480bb25a93a9267509eaaa0c9ef76fc'
            self.url_numatoms_split = 'https://www.dropbox.com/s/exzur542cw13buc/full_num-atoms_split_dict.pt?dl=1'
            self.md5_numatoms_split = '173db7902bd9b1963c28899845669c9f'
        elif subset == '530k':
            self.url_shuffle_split = 'https://www.dropbox.com/s/p8fzkt2ff3zrpo7/530k_shuffle_split_dict.pt?dl=1'
            self.md5_shuffle_split = 'e7951276cf80b4d011c59d5efe9b70cd'
            self.url_numatoms_split = 'https://www.dropbox.com/s/vjilcw352lvl8kl/530k_num-atoms_split_dict.pt?dl=1'
            self.md5_numatoms_split = 'ee6385dec83f608d0cc796ccb4f40e8a'
        else:
            raise f"Unexpected dataset subset name: {self.subset}"
        self.generate_splits = False

        # Check version and update if necessary.
        release_tag = osp.join(self.folder, self.version)
        if osp.isdir(self.folder) and (not osp.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == 'y':
                shutil.rmtree(self.folder)

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'pcqm4m-contact.tsv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self.url):
            path = ogb_download_url(self.url, self.raw_dir)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(osp.join(self.root, hash), 'w').close()
            # Download train/val/test splits.
            try:
                path_split1 = download_url(self.url_shuffle_split, self.folder)
                assert self._md5sum(path_split1) == self.md5_shuffle_split
                path_split2 = download_url(self.url_numatoms_split, self.folder)
                assert self._md5sum(path_split2) == self.md5_numatoms_split
            except Exception as e:
                print(f"Exception while downloading dataset splits: {e}")
                print(f"...splits will be regenerated.")
                self.generate_splits = True
        else:
            print('Stop download.')
            exit(-1)

    def _process_smiles(self, smiles):
        """ Construct PyG graph data object with contact edges from a CXSMILES.

        Args:
            smiles (str): Chemaxon Extended SMILES

        Returns:
            torch_geometric.data.Data
        """
        data = Data()

        graph = self.smiles2graph(smiles)
        if len(graph['contact_idx']) == 0:
            return None

        assert len(graph['edge_feat']) == graph['edge_index'].shape[1]
        assert len(graph['node_feat']) == graph['num_nodes']

        data.__num_nodes__ = int(graph['num_nodes'])
        data.edge_index = torch.from_numpy(graph['edge_index']).long()
        data.edge_attr = torch.from_numpy(graph['edge_feat']).long()
        data.x = torch.from_numpy(graph['node_feat']).long()
        data.y = None

        # Format edge labels.
        id_pos = to_undirected(torch.from_numpy(graph['contact_idx'].T))
        data.edge_index_labeled = id_pos
        data.edge_label = torch.ones(id_pos.shape[1], dtype=torch.int)

        # Note: Call a negative edge sampling transform to save precomputed
        # negative edges, otherwise rely on on-the-fly sampling by setting
        # one of these transforms as the Dataset's transform function.

        ## Sample negative edges for each head node of a positive edge.
        # data = structured_neg_sampling_transform(data)

        ## Sample random negative edges using PyG method.
        # data = neg_sampling_transform(data)

        ## All edges that are "not in contact" are negative edges.
        # data = complete_neg_transform(data)

        return data

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'pcqm4m-contact.tsv.gz'),
                              sep="\t")
        # Chemaxon Extended SMILES
        if self.subset == 'full':
            smiles_list = data_df['cxsmiles']
        elif self.subset == '530k':
            smiles_list = [s for i, s in enumerate(data_df['cxsmiles'])
                           if i % 6 == 0]  # Subset.
        else:
            raise f"Unexpected dataset subset name: {self.subset}"
        del data_df

        print('Converting CXSMILES strings into graphs...')
        data_list = Parallel(n_jobs=-1, batch_size='auto')(
            delayed(self._process_smiles)(s) for s in tqdm(smiles_list)
        )
        data_list = [g for g in data_list if g is not None]

        NG = len(data_list)
        num_skipped = len(smiles_list) - NG
        size_stats = [0] * 3
        for d in data_list:
            size_stats[0] += d.num_nodes
            size_stats[1] += (d.edge_label == 1).long().sum()
            size_stats[2] += (d.edge_label == 0).long().sum()
        print(f"Processing done: "
              f"num. kept mols={NG}, num. skipped={num_skipped}")
        print(f"      avg stats: |G|={size_stats[0] / NG}, "
              f"|pos_e|={size_stats[1] / NG}, |neg_e|={size_stats[2] / NG}")

        if self.generate_splits:
            # Random shuffle split of the molecules by 90/5/5 ratio.
            self.create_shuffle_split(len(data_list),
                                      self.VAL_RATIO, self.TEST_RATIO)

            # Create 90/5/5 split by the size of molecules.
            num_atoms_list = [d.num_nodes for d in data_list]
            self.create_numatoms_split(num_atoms_list,
                                       self.VAL_RATIO, self.TEST_RATIO)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def create_shuffle_split(self, N, val_ratio, test_ratio):
        """ Create a random shuffle split and saves it to disk.
        Args:
            N: Total size of the dataset to split.
        """
        rng = np.random.default_rng(seed=self.SEED)
        all_ind = rng.permutation(N)
        train_ratio = 1 - val_ratio - test_ratio
        val_ratio_rem = val_ratio / (val_ratio + test_ratio)

        # Random shuffle split into 90/5/5.
        train_ind = all_ind[:int(train_ratio * N)]
        tmp_ind = all_ind[int(train_ratio * N):]
        val_ind = tmp_ind[:int(val_ratio_rem * len(tmp_ind))]
        test_ind = tmp_ind[int((1 - val_ratio_rem) * len(tmp_ind)):]
        assert self._check_splits(N, [train_ind, val_ind, test_ind],
                                  [train_ratio, val_ratio, test_ratio])

        shuffle_split = {'train': train_ind, 'val': val_ind, 'test': test_ind}
        torch.save(shuffle_split,
                   osp.join(self.folder, f'{self.subset}_shuffle_split_dict.pt'))

    def create_numatoms_split(self, num_atoms_list, val_ratio, test_ratio):
        """ Create split by the size of molecules, testing on the largest ones.
        Args:
            num_atoms_list: List with molecule size per each graph.
        """
        rng = np.random.default_rng(seed=self.SEED)
        all_ind = np.argsort(np.array(num_atoms_list))
        train_ratio = 1 - val_ratio - test_ratio
        val_ratio_rem = val_ratio / (val_ratio + test_ratio)

        # Split based on mol size into 90/5/5, but shuffle the top 10% randomly
        # before splitting to validation and test set.
        N = len(num_atoms_list)
        train_ind = all_ind[:int(train_ratio * N)]
        tmp_ind = all_ind[int(train_ratio * N):]
        rng.shuffle(tmp_ind)
        val_ind = tmp_ind[:int(val_ratio_rem * len(tmp_ind))]
        test_ind = tmp_ind[int((1 - val_ratio_rem) * len(tmp_ind)):]
        assert len(train_ind) + len(val_ind) + len(test_ind) == N
        assert self._check_splits(N, [train_ind, val_ind, test_ind],
                                  [train_ratio, val_ratio, test_ratio])

        size_split = {'train': train_ind, 'val': val_ind, 'test': test_ind}
        torch.save(size_split,
                   osp.join(self.folder, f'{self.subset}_num-atoms_split_dict.pt'))

    def _check_splits(self, N, splits, ratios):
        """ Check whether splits intersect and raise error if so.
        """
        assert sum([len(split) for split in splits]) == N
        for ii, split in enumerate(splits):
            true_ratio = len(split) / N
            assert abs(true_ratio - ratios[ii]) < 3 / N
        for i in range(len(splits) - 1):
            for j in range(i + 1, len(splits)):
                n_intersect = len(set(splits[i]) & set(splits[j]))
                if n_intersect != 0:
                    raise ValueError(
                        f"Splits must not have intersecting indices: "
                        f"split #{i} (n = {len(splits[i])}) and "
                        f"split #{j} (n = {len(splits[j])}) have "
                        f"{n_intersect} intersecting indices"
                    )
        return True

    def get_idx_split(self, split_name):
        """ Get dataset splits.

        Args:
            split_name: Split type: 'shuffle', 'num-atoms'

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = osp.join(
            self.folder,
            f"{self.subset}_{split_name.replace('-', '_')}_split_dict.pt"
        )
        split_dict = replace_numpy_with_torchtensor(torch.load(split_file))
        return split_dict


if __name__ == '__main__':
    dataset = PygPCQM4Mv2ContactDataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset.get_idx_split('shuffle'))
