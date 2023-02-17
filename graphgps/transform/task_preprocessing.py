import torch


def shuffle(tensor):
    idx = torch.randperm(len(tensor))
    return tensor[idx]


def task_specific_preprocessing(data, cfg):
    """Task-specific preprocessing before the dataset is logged and finalized.

    Args:
        data: PyG graph
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    if cfg.gnn.head == "infer_links":
        N = data.x.size(0)
        idx = torch.arange(N, dtype=torch.long)
        complete_index = torch.stack([idx.repeat_interleave(N), idx.repeat(N)], 0)

        data.edge_attr = None
        
        if cfg.dataset.infer_link_label == "edge":
            labels = torch.empty(N, N, dtype=torch.long)
            non_edge_index = (complete_index.T.unsqueeze(1) != data.edge_index.T).any(2).all(1).nonzero()[:, 0]
            non_edge_index = shuffle(non_edge_index)[:data.edge_index.size(1)]
            edge_index = (complete_index.T.unsqueeze(1) == data.edge_index.T).all(2).any(1).nonzero()[:, 0]

            final_index = shuffle(torch.cat([edge_index, non_edge_index]))
            data.complete_edge_index = complete_index[:, final_index]

            labels.fill_(0)
            labels[data.edge_index[0], data.edge_index[1]] = 1

            assert labels.flatten()[final_index].mean(dtype=torch.float) == 0.5
        else:
            raise ValueError(f"Infer-link task {cfg.dataset.infer_link_label} not available.")

        data.y = labels.flatten()[final_index]

    supported_encoding_available = (
        cfg.posenc_LapPE.enable or
        cfg.posenc_RWSE.enable or
        cfg.posenc_GraphormerBias.enable
    )

    if cfg.dataset.name == "TRIANGLES":

        # If encodings are present they can append to the empty data.x
        if not supported_encoding_available:
            data.x = torch.zeros((data.x.size(0), 1))
        data.y = data.y.sub(1).to(torch.long)

    if cfg.dataset.name == "CSL":

        # If encodings are present they can append to the empty data.x
        if not supported_encoding_available:
            data.x = torch.zeros((data.num_nodes, 1))
        else:
            data.x = torch.zeros((data.num_nodes, 0))

    return data
