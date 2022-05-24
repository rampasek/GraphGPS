# GraphGPS: General Powerful Scalable Graph Transformers

How to build a graph Transformer? We provide a 3-part recipe on how to build graph Transformers with linear complexity. Our GPS recipe consists of choosing 3 main ingredients:
1. positional/structural encoding,
2. local message-passing mechanism,
3. global attention mechanism.

In this *GraphGPS* package we provide several positional/structural encodings and model choices, implementing the GPS recipe. GraphGPS is built using [PyG](https://www.pyg.org/) and [GraphGym from PyG2](https://pytorch-geometric.readthedocs.io/en/2.0.0/notes/graphgym.html).
Specifically PyG v2.0.2 is required.


### Python environment setup with Conda

```bash
conda create -n graphgps python=3.9
conda activate graphgps

conda install pytorch=1.9 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.2 -c pyg -c conda-forge
conda install pandas scikit-learn

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install performer-pytorch
pip install torchmetrics==0.7.2
pip install ogb
pip install wandb

conda clean --all
```


### Running GraphGPS
```bash
conda activate graphgps

# Running GPS with RWSE and tuned hyperparameters for ZINC.
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml  wandb.use False

# Running config with tuned SAN hyperparams for ZINC.
python main.py --cfg configs/SAN/zinc-SAN.yaml  wandb.use False

# Running a debug/dev config for ZINC.
python main.py --cfg tests/configs/graph/zinc.yaml  wandb.use False
```


### Benchmarking GPS on 11 datasets
See `run/run_experiments.sh` script to run multiple random seeds per each of the 11 datasets. We rely on Slurm job scheduling system.

Alternatively, you can run them in terminal following the example below. Configs for all 11 datasets are in `configs/GPS/`.
```bash
conda activate graphgps
# Run 10 repeats with 10 different random seeds (0..9):
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml  --repeat 10  wandb.use False
# Run a particular random seed:
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml  --repeat 1  seed 42  wandb.use False
```


### W&B logging
To use W&B logging, set `wandb.use True` and have a `gtransformers` entity set-up in your W&B account (or change it to whatever else you like by setting `wandb.entity`).



## Unit tests

To run all unit tests, execute from the project root directory:

```bash
python -m unittest -v
```

Or specify a particular test module, e.g.:

```bash
python -m unittest -v unittests.test_eigvecs
```
