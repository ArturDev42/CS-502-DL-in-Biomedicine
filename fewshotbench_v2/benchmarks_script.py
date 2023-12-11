import yaml
import subprocess

iter_nums = [30, 60, 90]
learning_rates = [0.001, 0.01, 0.1]
#add more hypers
datasets = ["tabula_muris", "swissprot"]
methods = ["maml" , "protonet", "matchingnet", "baseline"]
yaml_path = 'conf/main.yaml'

#make sure original parameters are in:
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)

config['iter_num'] = 600
config['lr'] = 0.001

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


#make sure params are for comet
command = f'sudo /home/tim.wiebelhaus18/miniconda3/envs/fewshotbench/bin/python comet_changes_script.py comet=true'
subprocess.run(command, shell=True)

run_num = 0
for dataset in datasets:
    for iter_num in iter_nums:
        for lr in learning_rates:
            run_num += 1
            print(run_num)
            # Read the current yaml file
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)

            # Update the iter_num, lr, and exp.name values
            config['iter_num'] = iter_num
            config['lr'] = lr
            exp_name = f'comet_iter{iter_num}_lr{lr}'.replace('.', 'p')  # Replace '.' with 'p' for naming
            config['exp']['name'] = exp_name

            # Write the updated configuration back to the yaml file
            with open(yaml_path, 'w') as file:
                yaml.dump(config, file)

            # Construct the command to run the experiment
            command = f'sudo /home/tim.wiebelhaus18/miniconda3/envs/fewshotbench/bin/python run.py exp.name={exp_name} method=comet dataset={dataset}'

            # Run the command
            subprocess.run(command, shell=True)
