import sys
import re

def modify_file(file_path, modifications):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            for pattern, replacement in modifications:
                if re.search(pattern, line):
                    line = re.sub(pattern, replacement, line)
                    break
            file.write(line)

def modify_configurations(comet):
    # Define the modifications for each file
    tabula_muris_modifications = [
        (r'^( *)(_target_: backbones.fcnet.EnFCNet)$', r'\1\2') if comet else (r'^( *)#(_target_: backbones.fcnet.EnFCNet)$', r'\1#\2')
        (r'^( *)(_target_: backbones.fcnet.FCNet)$', r'\1\2') if not comet else (r'^( *)#(_target_: backbones.fcnet.FCNet)$', r'\1#\2')
        (r'^( *)(layer_dim: [ 64, 64 ])$', r'\1\2') if not comet else (r'^( *)#(layer_dim: [ 64, 64 ])$', r'\1#\2')

    ]

    main_modifications = [
        (r'^( *)(model: EnFCNet)$', r'\1\2') if comet else (r'^( *)#(model: EnFCNet)$', r'\1#\2')
        (r'^( *)(model: FCNet)$', r'\1\2') if not comet else (r'^( *)#(model: FCNet)$', r'\1#\2')

    ]

    # Modify the files
    modify_file('conf/dataset/tabula_muris.yaml', tabula_muris_modifications)
    modify_file('conf/main.yaml', main_modifications)

#comet = true when you want script to make changes for comet run, false for all other runs
if __name__ == "__main__":
    comet_state = sys.argv[1] == "comet=true"
    modify_configurations(comet_state)
