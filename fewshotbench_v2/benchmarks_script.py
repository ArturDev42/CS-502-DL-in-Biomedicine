import yaml
import subprocess

yaml_path = 'conf/main.yaml'


#add more hypers
datasets = ["tabula_muris", "swissprot"]
methods = ["maml" , "protonet", "matchingnet", "baseline"]

#make sure original parameters are in:
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

config['iter_num'] = 600
config['lr'] = 0.001


# from the output we can see that by default these values are used
# although not explicitly set in the config (???)
# n_way: 5
# n_shot: 5
# n_query: 15

# as a comparison
#config['n_way'] = 1
#config['n_shot'] = 1
#config['n_query'] = 3

with open(yaml_path, 'w') as file:
    yaml.dump(config, file)


#make sure params are not for comet
command = f'sudo /home/tim.wiebelhaus18/miniconda3/envs/fewshotbench/bin/python comet_changes_script.py comet=false'
subprocess.run(command, shell=True)

#Run all baselines
for method in methods:
    for dataset in datasets:
        # Construct the command to run the experiment
        exp_name = f'baserun_tim_{method}_{dataset}'
        command = f'sudo /home/tim.wiebelhaus18/miniconda3/envs/fewshotbench/bin/python run.py exp.name={exp_name} method={method} dataset={dataset}'

        # Run the command
        subprocess.run(command, shell=True)
