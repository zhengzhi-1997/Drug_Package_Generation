# Interaction-aware Drug Package Recommendation via Policy Gradient
This repository contains the source code for the paper `Interaction-aware Drug Package Recommendation via Policy Gradient`, presented at [TOIS 2022](nips.cc).
```
@article{10.1145/3511020,
author = {Zheng, Zhi and Wang, Chao and Xu, Tong and Shen, Dazhong and Qin, Penggang and Zhao, Xiangyu and Huai, Baoxing and Wu, Xian and Chen, Enhong},
title = {Interaction-Aware Drug Package Recommendation via Policy Gradient},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1046-8188},
url = {https://doi.org/10.1145/3511020},
doi = {10.1145/3511020},
note = {Just Accepted},
journal = {ACM Trans. Inf. Syst.},
month = {jan},
keywords = {package recommendation, drug recommendation, reinforcement learning, graph neural network}
}
}
```


## Instruction
We use both a private dataset (APH) and a public dataset (MIMIC-III) in our paper, and the code in this repository is specific for MIMIC-III. However, the drug-drug interaction data in our paper is specific for our private dataset, and the overlapping part with MIMIC-III is ralatively small. Therefore, use `run_simple.py` is enough for MIMIC-III. You can also add your own drug-drug interaction data for MIMIC-III and use `run.py` to get better results.

## How to run the code
### Preprocessing
You should first prepare the MIMIC-III data and modify the file path in the code. Then run `preprocessing.py` to preprocess the MIMIC-III data.
### Build the Cython code
We utilize Cython to calculate the F-value as reward, and we need to build the Cython code before we use it.
```
sh build.sh
```
### Train the model
Pretrain the model with MSELoss:
```
python run_simple.py --reinforce no
```
Fintune the model with reinforcement learning:
```
python run_simple.py --reinforce yes
```
## Reproducibility
According to [rusty1s](https://github.com/pyg-team/pytorch_geometric/issues/92), PyG is non-deterministic by nature on GPU, so `run.py` is non-deterministic, and `run_simple.py` is deterministic since there this is no PyG code.
