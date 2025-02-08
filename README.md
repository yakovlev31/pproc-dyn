# Learning Spatiotemporal Dynamical Systems from Point Process Observations

Official implementation of [Learning Spatiotemporal Dynamical Systems from Point Process Observations](https://arxiv.org/abs/2406.00368).

## Installation

```bash
git clone https://github.com/yakovlev31/pproc-dyn.git; 
cd pproc-dyn; 
conda create -n pproc_env python=3.11; 
conda activate pproc_env; 
pip install -e .; 
```

## Getting Started

### Data

Archive with datasets can be downloaded [here](https://drive.google.com/file/d/16i75fwtJ06A2j-CnZvEmf1wJcDtgdyFC/view?usp=sharing). The datasets should be extracted to `pproc-dyn/`.

If you want to use your own dataset, follow the scripts in `./stpp/utils/`.

### Training and testing

```bash
python experiments/_burgers1d/train.py --name mymodel --device cuda --visualize 1

python experiments/_burgers1d/test.py --name mymodel --device cuda
```

See `./stpp/utils/_burgers1d.py` for all command line arguments.
