import os
import shutil
import subprocess
from glob import glob
from omegaconf import OmegaConf
from itertools import product


def clean(experiment_name):
    if os.path.exists(experiment_name):
        shutil.rmtree(experiment_name, ignore_errors=True)
    os.mkdir(experiment_name)


def create_rename_file(experiment_name):
    file = f"""#!/home/ychen878/miniconda3/bin/python
#
# rename.py
from glob import glob
import subprocess

subprocess.run(["mv", "results.parquet", f"{experiment_name}.parquet"])

for file in glob("extracts_*.err"):
    subprocess.run(["rm", "-rf", file])
for file in glob("extracts_*.log"):
    subprocess.run(["rm", "-rf", file])
for file in glob("extracts_*.out"):
    subprocess.run(["rm", "-rf", file])

# END"""
    with open(os.path.join(experiment_name, "rename.py"), "w") as fp:
        fp.write(file)


def create_agg_items(experiment_name):
    file = f"""#!/home/ychen878/miniconda3/bin/python
import os
from glob import glob
import shutil
import tarfile
import subprocess

os.mkdir("items")
for file in glob("item_*.item"):
    shutil.move(file, "items")

tarball = tarfile.open("{experiment_name}_items.tar.gz", "w:gz")
tarball.add("./items")
tarball.close()

shutil.move("{experiment_name}_items.tar.gz", "/staging/ychen878/polytope-enumeration/")
subprocess.run(["rm", "-rf", "items"])

for file in glob("item_*.item"):
    subprocess.run(["rm", "-rf", file])
for file in glob("items_*.err"):
    subprocess.run(["rm", "-rf", file])
for file in glob("items_*.out"):
    subprocess.run(["rm", "-rf", file])
for file in glob("items_*.log"):
    subprocess.run(["rm", "-rf", file])
"""
    with open(os.path.join(experiment_name, "agg_items.py"), "w") as fp:
        fp.write(file)


def create_agg_users(experiment_name):
    file = f"""#!/home/ychen878/miniconda3/bin/python
import os
from glob import glob
import shutil
import subprocess
import tarfile

for file in glob("users_*.tar.gz"):
    subprocess.run(["tar", "-zxf", file, "--strip-components", "1"])

tarball = tarfile.open("{experiment_name}_users.tar.gz", "w:gz")
tarball.add("./users")
tarball.close()

shutil.move("{experiment_name}_users.tar.gz", f"/staging/ychen878/polytope-enumeration/")
subprocess.run(["rm", "-rf", "users"])

for file in glob("users_*.tar.gz"):
    subprocess.run(["rm", "-rf", file])
for file in glob("users_*.err"):
    subprocess.run(["rm", "-rf", file])
for file in glob("users_*.out"):
    subprocess.run(["rm", "-rf", file])
for file in glob("users_*.log"):
    subprocess.run(["rm", "-rf", file])
"""
    with open(os.path.join(experiment_name, "agg_users.py"), "w") as fp:
        fp.write(file)

        
def create_agg_results(experiment_name):
    file = f"""#!/home/ychen878/miniconda3/bin/python
import os
from glob import glob
import subprocess
import tarfile
import shutil

for file in glob("results_*.tar.gz"):
    subprocess.run(["tar", "-zxf", file, "--strip-components", "1"])

tarball = tarfile.open("{experiment_name}_results.tar.gz", "w:gz")
tarball.add("./results")
tarball.close()

shutil.move("{experiment_name}_results.tar.gz", f"/staging/ychen878/polytope-enumeration/")
subprocess.run(["rm", "-rf", "results"])

for file in glob("results_*.tar.gz"):
    subprocess.run(["rm", "-rf", file])
for file in glob("*results_*.err"):
    subprocess.run(["rm", "-rf", file])
for file in glob("*results_*.out"):
    subprocess.run(["rm", "-rf", file])
for file in glob("*results_*.log"):
    subprocess.run(["rm", "-rf", file])
"""
    with open(os.path.join(experiment_name, "agg_results.py"), "w") as fp:
        fp.write(file)

        
def create_dag_file(experiment_name):
    file = """JOB items items.sub
JOB users users.sub
JOB our_results our_results.sub
JOB lu_results lu_results.sub
FINAL extracts extracts.sub

SCRIPT POST items ./agg_items.py
SCRIPT POST users ./agg_users.py
SCRIPT PRE extracts ./agg_results.py
SCRIPT POST extracts ./rename.sh

PARENT items CHILD users
PARENT users CHILD our_results lu_results
"""
    with open(os.path.join(experiment_name, f"{experiment_name}.dag"), "w") as fp:
        fp.write(file)

        
def create_submit_file(experiment_name, name, queue_str, arguments_str, cpu, ram, disk):
    submit_file = f'''# generate_{name}.sub

max_retries = 5
batch_name = {name}
universe = container

container_image = osdf:///chtc/staging/ychen878/sif/polytope-enumeration-v2.sif
transfer_input_files = polytope-enumeration-v2.sif

{arguments_str}
executable = {name}.sh

log = {name}_$(Process).log
error = {name}_$(Process).err
output = {name}_$(Process).out

request_cpus = {cpu}
request_memory = {ram}
request_disk = {disk}
max_idle = 2000

+WantFlocking = true
+WantGlideIn = true

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

periodic_hold = (JobStatus == 2) && (time() - EnteredCurrentStatus) > (2 * 3600)
periodic_hold_reason = "Job ran for more than two hours"
periodic_hold_subcode = 42
periodic_release = (HoldReasonSubCode == 42)

Requirements = (Target.HasCHTCStaging == true)

{queue_str}
# END'''
    with open(os.path.join(experiment_name, f"{name}.sub"), "w") as fp:
        fp.write(submit_file)

        
def create_executable(experiment_name, name, run_str, fro):
    if fro is None:
        executable_file = f"""#!/bin/bash
#
# {name}.sh

cp /staging/ychen878/code/polytope-enumeration.tar.gz ./
tar -xzf polytope-enumeration.tar.gz

{run_str}

rm -rf polytope-enumeration.tar.gz
rm -rf {name}
rm -rf src
rm -rf lu
rm *.py

# END"""
    else:
        executable_file = f"""#!/bin/bash
#
# {name}.sh

cp /staging/ychen878/code/polytope-enumeration.tar.gz ./
tar -xzf polytope-enumeration.tar.gz

cp /staging/ychen878/polytope-enumeration/{experiment_name}_{fro}.tar.gz ./
tar -xzf {experiment_name}_{fro}.tar.gz

{run_str}

rm -rf polytope-enumeration.tar.gz
rm -rf {experiment_name}_{fro}.tar.gz
rm -rf {fro}
rm -rf {name}
rm -rf src
rm -rf lu
rm *.py

# END"""
    with open(os.path.join(experiment_name, f"{name}.sh"), "w") as fp:
        fp.write(executable_file)

        
def generate_strs(args, name, flatten):
    arguments_str = "arguments = "
    for arg in args:
        arguments_str += f"$({arg}) "
    if flatten:
        arguments_str += "$(idx)"

    run_str = f"python generate_{name}.py "
    for i, arg in enumerate(args):
        run_str += f"--{arg} ${i + 1} "
    if flatten:
        run_str += f"--idx ${i + 2} "

    queue_str = "queue "
    for i, arg in enumerate(args):
        if i == len(args) - 1:
            queue_str += f"{arg}"
        else:
            queue_str += f"{arg}, "
    if flatten:
        queue_str += ", idx"
    queue_str += f" from {name}.inputs"
    return arguments_str, run_str, queue_str


def generate_inputs(experiment_name):
    config = OmegaConf.load(f'configs/{experiment_name}.yaml')

    num_parameters = {None: 1}
    for input_name in config.keys():
        args = config[input_name].parameters.keys()
        size = 0
        with open(os.path.join(experiment_name, f"{input_name}.inputs"), 'w') as f:
            if config[input_name]['flatten']:
                parameter_combinations = product(*config[input_name].parameters.values(), range(num_parameters[config[input_name]['from']]))
            else:
                parameter_combinations = product(*config[input_name].parameters.values())
            _num_parameters = 0
            for parameter_values in parameter_combinations:
                f.write(", ".join(map(str, parameter_values)) + "\n")
                size += 1
                _num_parameters += 1
            num_parameters[input_name] = _num_parameters
        print(f"Generated {size} inputs for {input_name}")
        arguments_str, run_str, queue_str = generate_strs(args, input_name, config[input_name]['flatten'])

        resources = config[input_name].resources
        cpu, ram, disk = resources.cpu, resources.ram, resources.disk
        create_submit_file(experiment_name, input_name, queue_str, arguments_str, cpu=cpu, ram=ram, disk=disk)
        create_executable(experiment_name, input_name, run_str, fro=config[input_name]['from'])


def generate_extracts(experiment_name):
    file = '''# extracts.sub
batch_name = extracts
universe = container

container_image = osdf:///chtc/staging/ychen878/sif/polytope-enumeration-v2.sif
transfer_input_files = polytope-enumeration-v2.sif

executable = extracts.sh

log = extracts_$(Process).log
error = extracts_$(Process).err
output = extracts_$(Process).out

request_cpus = 1
request_memory = 512MB
request_disk = 1GB

+WantFlocking = true
+WantGlideIn = true

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

Requirements = (Target.HasCHTCStaging == true)

queue 1
# END'''
    with open(os.path.join(experiment_name, "extracts.sub"), "w") as fp:
        fp.write(file)
    file = f'''#!/bin/bash
#
# extracts.sh

cp /staging/ychen878/code/polytope-enumeration.tar.gz ./
cp /staging/ychen878/polytope-enumeration/{experiment_name}_results.tar.gz ./

tar -xzf polytope-enumeration.tar.gz
tar -xzf {experiment_name}_results.tar.gz

python generate_parquet.py

rm -rf polytope-enumeration.tar.gz
rm -rf {experiment_name}_results.tar.gz
rm -rf results
rm -rf src
rm -rf lu
rm *.py

# END'''
    with open(os.path.join(experiment_name, "extracts.sh"), "w") as fp:
        fp.write(file)


def to_chtc(experiment_name):
    for script in glob(os.path.join(experiment_name, "*.py")) + glob(os.path.join(experiment_name, "*.sh")):
        subprocess.run(["chmod", "+x", script])
    subprocess.run(["scp", "-r", experiment_name, "ychen878@submit1.chtc.wisc.edu:/home/ychen878"])


def clean_up(experiment_name):
    if os.path.exists(experiment_name):
        shutil.rmtree(experiment_name, ignore_errors=True)


def create_experiment(experiment_name):
    clean(experiment_name)
    generate_inputs(experiment_name)
    create_rename_file(experiment_name)
    create_agg_items(experiment_name)
    create_agg_users(experiment_name)
    create_agg_results(experiment_name)
    create_dag_file(experiment_name)
    generate_extracts(experiment_name)
    to_chtc(experiment_name)
    clean_up(experiment_name)