import unittest as ut

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian, \
    from_networkx

from graphgps.transform.posenc_stats import (eigvec_normalizer,
                                             get_heat_kernels,
                                             get_heat_kernels_diag,
                                             get_rw_landing_probs,
                                             get_electrostatic_function_encoding)


def _get_linear_graph_edges(N):
    edges = torch.stack((torch.arange(1, N), torch.arange(0, N - 1)), dim=0)
    edges = torch.cat((edges, edges.flipud()), dim=1)
    return edges


def _get_eigvec_eigval(N):
    edges = _get_linear_graph_edges(N)
    L = to_scipy_sparse_matrix(*get_laplacian(edges, normalization=None))

    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals = torch.from_numpy(np.real(EigVals)).clamp_min(0)
    EigVecs = torch.from_numpy(EigVecs).float()

    return EigVecs, EigVals


def _get_ergraph_eigvec_eigval(N, p):
    G = nx.erdos_renyi_graph(N, p)
    L = to_scipy_sparse_matrix(*get_laplacian(from_networkx(G).edge_index,
                                              normalization=None))

    EigVals, EigVecs = np.linalg.eigh(L.toarray())
    EigVals = torch.from_numpy(np.real(EigVals)).clamp_min(0)
    EigVecs = torch.from_numpy(EigVecs).float()

    return EigVecs, EigVals


class TestEigvecsNormalization(ut.TestCase):

    def test_L1(self):

        for N in range(4, 10):
            EigVecs, EigVals = _get_eigvec_eigval(N)
            # Testing normalization L1
            normed_eigvec = eigvec_normalizer(EigVecs, EigVals,
                                              normalization="L1")
            self.assertListEqual(list(EigVecs.shape), list(normed_eigvec.shape))
            np.testing.assert_array_almost_equal(
                normed_eigvec.abs().sum(dim=0).numpy(),
                np.ones(N), decimal=6)

    def test_L2(self):
        for N in range(4, 10):
            EigVecs, EigVals = _get_eigvec_eigval(N)

            # Testing normalization L2
            normed_eigvec = eigvec_normalizer(EigVecs, EigVals,
                                              normalization="L2")
            self.assertListEqual(list(EigVecs.shape), list(normed_eigvec.shape))
            np.testing.assert_array_almost_equal(
                (normed_eigvec ** 2).sum(dim=0).numpy(),
                np.ones(N), decimal=6)

    def test_abs_max(self):
        for N in range(4, 10):
            EigVecs, EigVals = _get_eigvec_eigval(N)

            # Testing normalization abs-max
            normed_eigvec = eigvec_normalizer(EigVecs, EigVals,
                                              normalization="abs-max")
            self.assertListEqual(list(EigVecs.shape), list(normed_eigvec.shape))
            np.testing.assert_array_almost_equal(
                normed_eigvec.abs().max(dim=0).values.numpy(),
                np.ones(N), decimal=6)

    def test_wavelength(self):
        for N in range(4, 10):
            EigVecs, EigVals = _get_eigvec_eigval(N)

            # Testing normalization wavelength
            normed_eigvec = eigvec_normalizer(EigVecs, EigVals,
                                              normalization="wavelength")
            self.assertListEqual(list(EigVecs.shape), list(normed_eigvec.shape))
            self.assertAlmostEqual(
                normed_eigvec[0, 1].abs().numpy(),
                np.array(N / 2), delta=0.1)

    def test_wavelenght_asin(self):
        for N in range(4, 10):
            EigVecs, EigVals = _get_eigvec_eigval(N)

            # Testing normalization wavelength-asin
            normed_eigvec = eigvec_normalizer(EigVecs, EigVals,
                                              normalization="wavelength-asin")
            self.assertListEqual(list(EigVecs.shape), list(normed_eigvec.shape))
            self.assertAlmostEqual(
                normed_eigvec[0, 1].abs().numpy(),
                np.array(N / 2), delta=0.1)

    def test_wavelength_soft(self):
        for N in range(4, 10):
            EigVecs, EigVals = _get_eigvec_eigval(N)

            # Testing normalization wavelength-soft
            normed_eigvec = eigvec_normalizer(EigVecs, EigVals,
                                              normalization="wavelength-soft")
            self.assertListEqual(list(EigVecs.shape), list(normed_eigvec.shape))
            self.assertAlmostEqual(
                (normed_eigvec[0, 1].abs()).numpy(),
                N / 2, delta=0.5)


class TestHeatKernel(ut.TestCase):

    def test_heat_kernels(self):
        for N in [5, 10, 15, 20]:
            EigVecs, EigVals = _get_eigvec_eigval(N)
            no_kernel, _ = get_heat_kernels(EigVecs, EigVals,
                                            kernel_times=[])
            self.assertListEqual(no_kernel, [])

            heat_kernel_times = [1, 2, 3, 5]
            heat_kernels, heat_kernels_diag = get_heat_kernels(EigVecs, EigVals,
                                                               kernel_times=heat_kernel_times)

            self.assertListEqual(list(heat_kernels_diag.shape),
                                 [N, len(heat_kernel_times)])
            self.assertTrue(torch.all(heat_kernels_diag > 0))

            for kernel in heat_kernels:
                self.assertListEqual([N, N], list(kernel.shape))
                np.testing.assert_almost_equal(kernel.numpy(), kernel.T.numpy())
                self.assertAlmostEqual(0, kernel.sum(dim=0).abs().max(),
                                       delta=0.01)

    def test_heat_kernels_diag(self):
        for N in [10, 15, 20, 23]:
            EigVecs, EigVals = _get_eigvec_eigval(N)
            no_kernel = get_heat_kernels_diag(EigVecs, EigVals, kernel_times=[])
            self.assertListEqual(no_kernel, [])

            heat_kernel_times = [1, 2, 3, 5, 12, 24]
            heat_kernels_diag = get_heat_kernels_diag(EigVecs, EigVals, kernel_times=heat_kernel_times)

            self.assertListEqual(list(heat_kernels_diag.shape),
                                 [N, len(heat_kernel_times)])

            with self.assertRaises(AssertionError):
                np.testing.assert_array_almost_equal(heat_kernels_diag[:, 0], heat_kernels_diag[:, 1], decimal=1)
                np.testing.assert_array_almost_equal(heat_kernels_diag[:, 0], heat_kernels_diag[:, 2], decimal=1)
                np.testing.assert_array_almost_equal(heat_kernels_diag[:, 0], heat_kernels_diag[:, 3], decimal=1)

            heat_kernels_diag = get_heat_kernels_diag(EigVecs, EigVals, kernel_times=heat_kernel_times, space_dim=1)

            np.testing.assert_array_almost_equal(heat_kernels_diag[:, 0], heat_kernels_diag[:, 1], decimal=1)
            np.testing.assert_array_almost_equal(heat_kernels_diag[:, 0], heat_kernels_diag[:, 2], decimal=1)
            np.testing.assert_array_almost_equal(heat_kernels_diag[:, 0], heat_kernels_diag[:, 3], decimal=1)
            self.assertTrue(torch.all(heat_kernels_diag > 0))

    def test_heat_kernels_diag_vs_full(self):
        """
        Test if Heat kernel diagonals are the same computed from the full
        diffusion matrices or when specifically computing just the diagonal.
        """
        for N in [10, 15, 20, 23]:
            EigVecs, EigVals = _get_ergraph_eigvec_eigval(N, p=0.5)

            heat_kernel_times = [1, 2, 3, 5, 12, 24]
            hk_only_diag = get_heat_kernels_diag(EigVecs, EigVals, kernel_times=heat_kernel_times)
            _, hk_full_diag = get_heat_kernels(EigVecs, EigVals, kernel_times=heat_kernel_times)

            np.testing.assert_array_almost_equal(hk_only_diag, hk_full_diag)

    def test_rw_landing_probs(self):
        for N in [10, 15, 20, 23]:
            edges = _get_linear_graph_edges(N)

            rw_landing = get_rw_landing_probs(ksteps=[1], edge_index=edges)
            self.assertTrue(torch.all(rw_landing == 0))

            ksteps = [4, 6, 12, 24]
            rw_landing = get_rw_landing_probs(ksteps=ksteps, edge_index=edges)

            self.assertListEqual(list(rw_landing.shape), [N, len(ksteps)])

            with self.assertRaises(AssertionError):
                np.testing.assert_allclose(rw_landing[:, 0], rw_landing[:, 1], rtol=0.3)
                np.testing.assert_allclose(rw_landing[:, 0], rw_landing[:, 2], rtol=0.3)
                np.testing.assert_allclose(rw_landing[:, 0], rw_landing[:, 3], rtol=0.3)

            rw_landing = get_rw_landing_probs(ksteps=ksteps, edge_index=edges, space_dim=1)

            np.testing.assert_allclose(rw_landing[:, 0], rw_landing[:, 1], rtol=0.3)
            np.testing.assert_allclose(rw_landing[:, 0], rw_landing[:, 2], rtol=0.3)

            self.assertTrue(torch.all(rw_landing > 0))

    def test_get_electrostatic_function_encoding(self):
        for N in [10, 15, 20, 23]:
            edges = _get_linear_graph_edges(N)
            elstatic_encoding = get_electrostatic_function_encoding(edges, N)
            self.assertListEqual(list(elstatic_encoding.shape), [N, 10])


if __name__ == '__main__':
    ut.main()
