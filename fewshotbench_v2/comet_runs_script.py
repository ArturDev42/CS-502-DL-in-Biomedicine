import yaml
import subprocess

yaml_path = 'conf/main.yaml'


iter_nums = [30, 60, 90]
learning_rates = [0.001, 0.01, 0.1]
#add more hypers
methods = ["comet", "mapcell_v2"]
datasets = ["tabula_muris", "swissprot"]

#make sure params are for comet
command = f'sudo /home/tim.wiebelhaus18/miniconda3/envs/fewshotbench/bin/python comet_changes_script.py comet=true'
subprocess.run(command, shell=True)

run_num = 0
for method in methods:
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
                command = f'sudo /home/tim.wiebelhaus18/miniconda3/envs/fewshotbench/bin/python run.py exp.name={exp_name} method={method} dataset={dataset}'

                # Run the command
                subprocess.run(command, shell=True)
