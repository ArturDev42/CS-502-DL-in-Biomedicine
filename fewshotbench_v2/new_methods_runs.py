import yaml
import subprocess

yaml_path = 'conf/main.yaml'

#We used these parameters for a grid search hyperparameter tuning
#iter_nums = [30, 90, 600]
#learning_rates = [0.001, 0.01, 0.1]
#methods = ["comet","mapcell"]
#datasets = ["tabula_muris","swissprot"]

iter_nums = [30, 90, 600]
learning_rates = [0.001, 0.01, 0.1]
methods = ["comet", "mapcell"]
datasets = ["tabula_muris","swissprot"]

run_num = 0
for method in methods:
    if method == "comet":
        #runs script to make necessary comet changes
        command = f'sudo /home/tim.wiebelhaus18/miniconda3/envs/fewshotbench/bin/python comet_changes_script.py comet=true'
        subprocess.run(command, shell=True)
    for dataset in datasets:
        if method == "comet" and dataset == "swissprot":
            continue
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

                # Write the updated configuration back to the yaml file
                with open(yaml_path, 'w') as file:
                    yaml.dump(config, file)

                print(config)

                # Construct the command to run the experiment
                command = f'sudo /home/tim.wiebelhaus18/miniconda3/envs/fewshotbench/bin/python run.py exp.name={exp_name} method={method} dataset={dataset}'                

                # Run the command
                subprocess.run(command, shell=True)
