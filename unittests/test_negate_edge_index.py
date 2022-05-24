import shutil
import tempfile
import unittest

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx, to_networkx

from graphgps.utils import negate_edge_index


class TestNegateEdgeIndex(unittest.TestCase):

    def test_simple(self):
        """
        Simple path graph 0 <-> 1 <-> 2 <-> 3
        """
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        x = torch.tensor([[-1], [-2], [-3], [-4]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        answer = torch.tensor([[0, 0, 1, 2, 3, 3],
                               [2, 3, 3, 0, 0, 1]], dtype=torch.long)
        np.testing.assert_array_equal(negate_edge_index(data.edge_index),
                                      answer)

    def test_binomial_tree(self):
        G = nx.binomial_tree(6)
        Gneg = nx.algorithms.operators.unary.complement(G)

        np.testing.assert_array_equal(
            negate_edge_index(from_networkx(G).edge_index),
            from_networkx(Gneg).edge_index
        )

    def test_barbell_graph(self):
        G = nx.barbell_graph(4, 8)
        Gneg = nx.algorithms.operators.unary.complement(G)

        np.testing.assert_array_equal(
            negate_edge_index(from_networkx(G).edge_index),
            from_networkx(Gneg).edge_index
        )

    def test_erdos_renyi(self):
        G = nx.erdos_renyi_graph(123, 0.5)
        Gneg = nx.algorithms.operators.unary.complement(G)

        np.testing.assert_array_equal(
            negate_edge_index(from_networkx(G).edge_index),
            from_networkx(Gneg).edge_index
        )

    def test_watts_strogatz(self):
        G = nx.watts_strogatz_graph(n=500, k=6, p=0.1)
        Gneg = nx.algorithms.operators.unary.complement(G)

        np.testing.assert_array_equal(
            negate_edge_index(from_networkx(G).edge_index),
            from_networkx(Gneg).edge_index
        )

    def test_path_batch(self):
        """
        Two path graphs in a batch
        """
        data_list = []

        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]], dtype=torch.long)
        x = torch.tensor([[-1], [0], [1], [3]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

        batch = Batch.from_data_list(data_list)

        answer = torch.tensor([[0, 2, 3, 3, 4, 5, 6, 6],
                               [2, 0, 5, 6, 6, 3, 3, 4]], dtype=torch.long)
        np.testing.assert_array_equal(
            negate_edge_index(batch.edge_index, batch.batch),
            answer
        )

    def test_random_batch(self):
        orig_data_list = []
        neg_data_list = []
        for i in range(1, 10):
            N = 13 * i + i
            M = 2 * i + 5
            G = nx.barabasi_albert_graph(n=N, m=M)
            data = Data(edge_index=from_networkx(G).edge_index, num_nodes=N)
            orig_data_list.append(data)

            NG = nx.algorithms.operators.unary.complement(G)
            data = Data(edge_index=from_networkx(NG).edge_index, num_nodes=N)
            neg_data_list.append(data)

        orig_batch = Batch.from_data_list(orig_data_list)
        neg_batch = Batch.from_data_list(neg_data_list)
        np.testing.assert_array_equal(
            negate_edge_index(orig_batch.edge_index, orig_batch.batch),
            neg_batch.edge_index
        )


# @unittest.skip("Run this after the first unittest succeeded.")
class TestNegateEdgeIndexNCI1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_batches(self):
        dataset = TUDataset(root=self.test_dir, name='NCI1', use_node_attr=True)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        # Test on first several batches.
        for batch_id, batch in zip(range(5), loader):
            with self.subTest(i=batch_id):
                # print(f'>> testing batch {batch_id}')
                neg_data_list = []
                prev_graphs_size = 0
                for i in range(batch.batch.max() + 1):
                    # Mask for edges that belong to i-th graph in the batch.
                    edge_mask = batch.batch[batch.edge_index[0]] == i
                    # Check the graph is undirected.
                    np.testing.assert_array_equal(
                        edge_mask,
                        batch.batch[batch.edge_index[1]] == i
                    )

                    # Create correct negated graph.
                    edges = batch.edge_index[:, edge_mask] - prev_graphs_size
                    num_nodes = (batch.batch == i).sum().item()
                    G = to_networkx(Data(edge_index=edges,
                                         num_nodes=num_nodes))
                    NG = nx.algorithms.operators.unary.complement(G)
                    data = Data(edge_index=from_networkx(NG).edge_index,
                                num_nodes=num_nodes)
                    neg_data_list.append(data)
                    prev_graphs_size += num_nodes

                # Finally, test the batch negation.
                neg_batch = Batch.from_data_list(neg_data_list)
                np.testing.assert_array_equal(
                    negate_edge_index(batch.edge_index, batch.batch),
                    neg_batch.edge_index
                )


if __name__ == '__main__':
    unittest.main()
