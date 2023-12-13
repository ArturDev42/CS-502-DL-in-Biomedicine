# CS-502-DL-in-Biomedicine

## How we proceeded
1. Create instance & install neccessary dependencies
2. Clone our own repo with fewshotbench_v2 and comet on instance
3. Include comet into fewshotbench and make neccessary changes to run it as a method on tabula_muris
4. Implement mapcell as a method (implemented from scratch) and run it on tabula_muris and swissprot

## How we produced our results

1. Run "benchmarks_script.py" to run all benchmarks on original mail.yaml configurations (5 shot, 5 way)
2. Change 5 shot to 1 shot in main.yaml
3. Run "benchmarks_script.py" to run all benchmarks for 1 shot 5 way
4. Run "new_methods_runs.py" script to hypertune parameters for new methods (comet & mapcell) (5 shot, 5 way)
5. Run "new_methods_runs.py" script to hypertune parameters for new methods (comet & mapcell) (1 shot, 5 way)
6. Choose best models (hyperparameters) for both methods
7. For comet: Change euclidean distance to manhatten distance and run again on same hyperparameters
8. For comet: Change backbone from EnFCNet to EnFCNet_4 and run again on same hyperparameters
9. For mapcell: Change euclidean distance to manhatten distance and run again on same hyperparameters
10. For mapcell: Change margin for contrastive loss and run again on same hyperparameters


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

--> automated in script comet_changes_script.py
