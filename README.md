# Expressive Higher-Order Link Prediction through Hypergraph Symmetry Breaking

Implementation of the Symmetry Finder Algorithm, GWL-1 and hyperedge prediction baselines.

## Installation:
Requirements:
```commandline
alembic==1.12.0
certifi==2023.7.22
charset-normalizer==3.3.0
cmaes==0.10.0
colorlog==6.7.0
contourpy==1.1.1
cycler==0.12.1
filelock==3.12.4
fonttools==4.43.1
fsspec==2023.9.2
greenlet==3.0.0
idna==3.4
igraph==0.11.2
Jinja2==3.1.2
joblib==1.3.2
kiwisolver==1.4.5
Mako==1.2.4
MarkupSafe==2.1.3
matplotlib==3.8.0
mpmath==1.3.0
networkx==3.1
numpy==1.26.1
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.18.1
nvidia-nvjitlink-cu12==12.2.140
nvidia-nvtx-cu12==12.1.105
optuna==3.3.0
packaging==23.2
pandas==2.1.1
Pillow==10.1.0
psutil==5.9.6
pyparsing==3.1.1
python-dateutil==2.8.2
pytz==2023.3.post1
PyYAML==6.0.1
requests==2.31.0
scikit-learn==1.3.1
scipy==1.11.3
six==1.16.0
SQLAlchemy==2.0.22
sympy==1.12
texttable==1.7.0
threadpoolctl==3.2.0
torch==2.1.0
torch_geometric==2.4.0
torchaudio==2.1.0
torchvision==0.16.0
tqdm==4.66.1
triton==2.1.0
typing_extensions==4.8.0
tzdata==2023.3
urllib3==2.0.6
```
For installation:
```
pip install -r requirements.txt
```
## Running Experiments:
```commandline
python symmetrybreaker_hypergnn.py --dataset <dataset> --hgnn <hypergnn>
```
```
usage: symmetrybreaker_hypergnn.py [-h]
                                [-hgnn {HGNN,HGNNP,HNHN,HyperGCN,UniGCN,UniGIN,UniSAGE,UniGAT}]
                                [-dataset {congress-bills,email-Enron,email-Eu,contact-high-school,cat-edge-DAWN,cat-edge-Brain,cat-edge-Cooking,cat-edge-geometry-questions,cat-edge-madison-restaurant-reviews,cat-edge-music-blues-reviews,cat-edge-vegas-bars-reviews,NDC-classes,contact-primary-school,rand-regular,preferential-attachment,FB15k-237,penn94,reed98,amherst41,cornell5,johnshopkins55,genius,AIFB,MUTAG}]
                                [-pos_embedding {laplacian_emap,id}]
                                [-epochs EPOCHS] 
                                [-max_iter MAX_ITER][-device DEVICE]
                                [-embdim EMBDIM] [-p_sym P_SYM]
```
## Baselines:
```commandline
python hypergnn_baseline.py --dataset <dataset> --hgnn <hypergnn>
```
```
usage: hypergnn_baseline.py     [-h]
                                [-hgnn {HGNN,HGNNP,HNHN,HyperGCN,UniGCN,UniGIN,UniSAGE,UniGAT}]
                                [-dataset {congress-bills,email-Enron,email-Eu,contact-high-school,cat-edge-DAWN,cat-edge-Brain,cat-edge-Cooking,cat-edge-geometry-questions,cat-edge-madison-restaurant-reviews,cat-edge-music-blues-reviews,cat-edge-vegas-bars-reviews,NDC-classes,contact-primary-school,rand-regular,preferential-attachment,FB15k-237,penn94,reed98,amherst41,cornell5,johnshopkins55,genius,AIFB,MUTAG}]
                                [-pos_embedding {laplacian_emap,id}]
                                [-epochs EPOCHS] 
                                [-max_iter MAX_ITER][-device DEVICE]
                                [-embdim EMBDIM]
```
```commandline
python linkpred_hypergnn.py --dataset <dataset> --hgnn <hypergnn>
```
```
usage: linkpred_hypergnn.py     [-h]
                                [-hgnn {HGNN,HGNNP,HNHN,HyperGCN,UniGCN,UniGIN,UniSAGE,UniGAT}]
                                [-dataset {FB15k-237,penn94,reed98,amherst41,cornell5,johnshopkins55,genius,AIFB,MUTAG}]
                                [-pos_embedding {laplacian_emap,id}]
                                [-epochs EPOCHS][-max_iter MAX_ITER][-embdim EMBDIM]            
```
```commandline
python linkpred_gnn.py --dataset <dataset> --gnn <gnn>
```
```
usage: linkpred_gnn.py          [-h]
                                [-gnn {GCN,GraphSAGE,GIN,GAT,APPNP,GCN2}]
                                [-dataset {FB15k-237,penn94,reed98,amherst41,cornell5,johnshopkins55,genius,AIFB,MUTAG}]
                                [-pos_embedding {laplacian_emap,id}]
                                [-epochs EPOCHS][-max_iter MAX_ITER][-embdim EMBDIM]            
```

