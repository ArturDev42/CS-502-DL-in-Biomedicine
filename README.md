# CS-502-DL-in-Biomedicine

## Structure

### How do clean cache:
conda clean --all
pip cache purge

### How do check vm:
htop
nvidia-smi --loop=5         (5 is for 5 sec)


### How to make comet run (with EnFCNet)
in /home/timwiebelhaus/CS-502-DL-in-Biomedicine/fewshotbench_v2

Do these changes:
- conf/dataset/tabula_muris.yaml
    - uncomment _target_: backbones.fcnet.EnFCNet
    - comment out _target_: backbones.fcnet.FCNet
    - comment out layer_dim: [ 64, 64 ]

- conf/dataset/swissprot.yaml
    - uncomment _target_: backbones.fcnet.EnFCNet
    - comment out _target_: backbones.fcnet.FCNet
    - comment out layer_dim: [ 64, 64 ]

- conf/main.yaml
    - uncomment model: EnFCNet
    - comment out model: FCNet

- datasets/cell/tabula_muris.py
    - uncomment: go2gene = get_go2gene(adata=adata, GO_min_genes=32,GO_max_genes=None, GO_min_level=6, GO_max_level=1)
    - uncomment: go_mask = create_go_mask(adata, go2gene)

    - change dataloader functions

--> implemented in script comet_changes_script.py
