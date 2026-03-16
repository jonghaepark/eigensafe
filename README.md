# EigenSafe: A Spectral Framework for Learning-Based Probabilistic Safety Assessment

**Authors:** [Inkyu Jang*](https://janginkyu.github.io/), [Jonghae Park*](https://jonghaepark.github.io/), [Sihyun Cho](https://www.linkedin.com/in/sihyun-cho-2198863b7/), [Chams E. Mballo](https://www.linkedin.com/in/chams-eddine-m-6bba4688/), [Claire J. Tomlin](https://people.eecs.berkeley.edu/~tomlin/), [H. Jin Kim](https://larr.snu.ac.kr/)
<br>
<sub>(* Equal contribution)</sub>



This repository provides the official implementation of **EigenSafe** for Safe Reinforcement Learning.
For more details, please refer to the [paper](https://arxiv.org/abs/2509.17750) and the [project page](https://eigen-safe.github.io).
 
## Setup
```bash
conda env create -f environment.yml
conda activate eigensafe
```

## Train

Run the training scripts from the repository root:

```bash
./scripts/train_{cheetahlow,hopperhigh,antball,lunarlanderhard}.sh
```
Pass extra args to override defaults, e.g. `./train_cheetahlow.sh --seed 717`.

## Evaluate

Select an experiment in `results/`, choose the desired episode checkpoint, and run:

```bash
python evaluate.py
```

## Repo map
- `envs/`: task definitions and assets
- `algo/`: SAC + EigenSafe safety critic 
- `results/`: logs, videos, checkpoints
- `scripts/`: training launch scripts
- `train.py`: training implementation
- `evaluate.py`: evaluation script


## Citation
```bibtex
@article{jang2026eigensafe,
  title={EigenSafe: A Spectral Framework for Learning-Based Probabilistic Safety Assessment},
  author={Jang, Inkyu and Park, Jonghae and Cho, Sihyun and Mballo, Chams E and Tomlin, Claire J and Kim, H Jin},
  journal={arXiv preprint arXiv:2509.17750},
  year={2026}
}
```
