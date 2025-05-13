"""
This module provides functionality for training machine learning models using PyTorch Lightning.

It includes methods for training with different logging tools (e.g., Weights & Biases, Neptune), 
generating SLURM scripts for HPC environments, and managing hyperparameter optimization trials.

Classes:
    - baseparser: A custom argument parser for hyperparameter optimization.
    - parser: Extends baseparser to handle Neptune and Weights & Biases trials.

Functions:
    - train: Trains a model using PyTorch Lightning.
    - wandbtrain: Wrapper for training with Weights & Biases logging.
    - neptunetrain: Wrapper for training with Neptune logging.
    - SlurmRun: Generates a SLURM script for running experiments on HPC systems.
    - __get_hopt_params: Converts hyperparameter optimization trials into script parameters.
    - __should_escape: Determines if a value should be escaped in the command line.
"""

from test_tube import HyperOptArgumentParser
import wandb
import torch
from tqdm import tqdm
import datetime
import pytorch_lightning
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping
import os
import sys
from model import Pangu
from DataModule import DatasetFromMinioBucket

YOURPROJECTNAME = "TestDeploy"
WANDBUSER = "st7ma784"
NEPTUNEUSER = "st7ma784"
torch.set_float32_matmul_precision('medium')


def train(config={
        "batch_size": 16,  # ADD MODEL ARGS HERE
        "codeversion": "-1",
    }, dir=None, devices=None, accelerator=None, Dataset=None, logtool=None, EvalOnLaunch=False):
    """
    Trains a model using PyTorch Lightning.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        dir (str): Directory to save model checkpoints.
        devices (int or str): Number of devices to use for training.
        accelerator (str): Accelerator type (e.g., 'gpu', 'cpu').
        Dataset (Dataset): Dataset object for training.
        logtool (Logger): Logger for tracking training metrics.
        EvalOnLaunch (bool): Whether to evaluate the model immediately after training.

    Returns:
        None
    """
    minioClient = {
        "host": config["MINIOHost"],
        "port": config["MINIOPort"],
        "access_key": config["MINIOAccesskey"],
        "secret_key": config["MINIOSecret"]
    }
    config.update({"minioClient": minioClient})
    model = Pangu(**config)
    dataModule = DatasetFromMinioBucket(**config)

    print("Building model")
    if devices is None:
        devices = config.get("devices", "auto")
    if accelerator is None:
        accelerator = config.get("accelerator", "auto")

    filename = "model-{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    callbacks = [
        TQDMProgressBar(),
        EarlyStopping(monitor="train_loss", mode="min", patience=10, check_finite=True, stopping_threshold=0.00001),
        pytorch_lightning.callbacks.ModelCheckpoint(
            monitor='train_loss',
            dirpath=dir,
            filename=filename,
            save_top_k=1,
            mode='min',
            save_last=True
        ),
    ]

    trainer = pytorch_lightning.Trainer(
        devices=devices,
        accelerator=accelerator,
        max_epochs=200,
        logger=logtool,
        callbacks=callbacks,
        gradient_clip_val=0.25,
        fast_dev_run=config.get("fast_dev_run", False),
        precision=config.get('precision', 16)
    )

    trainer.fit(model, dataModule)


def wandbtrain(config=None, dir=None, devices=None, accelerator=None, Dataset=None):
    """
    Wrapper for training with Weights & Biases logging.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        dir (str): Directory to save model checkpoints.
        devices (int or str): Number of devices to use for training.
        accelerator (str): Accelerator type (e.g., 'gpu', 'cpu').
        Dataset (Dataset): Dataset object for training.

    Returns:
        None
    """
    if config is not None:
        config = config.__dict__
        dir = config.get("dir", dir)
        logtool = pytorch_lightning.loggers.WandbLogger(
            project="TestDeploy",
            entity=WANDBUSER,
            save_dir=os.getenv("global_scratch", dir),
            name="TestDeploy"
        )
    else:
        run = wandb.init(project="TestDeploy", entity=WANDBUSER, name="TestDeploy", config=config)
        logtool = pytorch_lightning.loggers.WandbLogger(
            project="TestDeploy",
            entity=WANDBUSER,
            experiment=run,
            save_dir=os.getenv("global_scratch", dir)
        )
        config = run.config.as_dict()

    train(config, dir, devices, accelerator, Dataset, logtool)
    wandb.finish()


def neptunetrain(config=None, dir=None, devices=None, accelerator=None, Dataset=None):
    """
    Wrapper for training with Neptune logging.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
        dir (str): Directory to save model checkpoints.
        devices (int or str): Number of devices to use for training.
        accelerator (str): Accelerator type (e.g., 'gpu', 'cpu').
        Dataset (Dataset): Dataset object for training.

    Returns:
        None
    """
    import neptune
    from neptunecontrib.api import search_runs

    if config is not None:
        config = config.__dict__
        dir = config.get("dir", dir)
        logtool = pytorch_lightning.loggers.NeptuneLogger(
            project="TestDeploy",
            entity=NEPTUNEUSER,
            save_dir=dir
        )
    else:
        run = neptune.init(project="TestDeploy", entity=NEPTUNEUSER, name="TestDeploy", config=config)
        logtool = pytorch_lightning.loggers.NeptuneLogger(
            project="TestDeploy",
            entity=NEPTUNEUSER,
            experiment=run,
            save_dir=dir
        )
        config = run.config.as_dict()

    train(config, dir, devices, accelerator, Dataset, logtool)


def SlurmRun(trialconfig):
    """
    Generates a SLURM script for running experiments on HPC systems.

    Args:
        trialconfig (dict): Configuration dictionary for the trial.

    Returns:
        str: SLURM script as a string.
    """
    job_with_version = '{}v{}'.format("SINGLEGPUTESTLAUNCH", 0)

    sub_commands = ['#!/bin/bash',
                    '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',
                    '#SBATCH --time={}'.format('24:00:00'),  # Max run time
                    '#SBATCH --job-name={}'.format(job_with_version),
                    '#SBATCH --nodes=1',  # Nodes per experiment
                    '#SBATCH --ntasks-per-node=1',  # Set this to GPUs per node.
                    '#SBATCH --gres=gpu:1',  # {}'.format(per_experiment_nb_gpus),
                    f'#SBATCH --signal=USR1@{5 * 60}',
                    '#SBATCH --mail-type={}'.format(','.join(['END', 'FAIL'])),
                    '#SBATCH --mail-user={}'.format('st7ma784@gmail.com'),
                    ]
    comm = "python"
    slurm_commands = {}

    if str(os.getenv("HOSTNAME", "localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
            '#SBATCH --account MYACOCUNT',
            '''
                arch=$(uname -i) # Get the CPU architecture
                if [[ $arch == "aarch64" ]]; then
                   # Set variables and source scripts for aarch64
                   export CONDADIR=/nobackup/projects/<project>/$USER/ # Update this with your <project> code.
                   source $CONDADIR/aarchminiconda/etc/profile.d/conda.sh
                fi
                ''',
            'export CONDADIR=/nobackup/projects/<BEDEPROJECT>/$USER/miniconda',
            'export NCCL_SOCKET_IFNAME=ib0'])
        comm = "python3"
    else:

        sub_commands.extend(['#SBATCH -p gpu-medium',
                             '#SBATCH --mem=96G',
                             '#SBATCH --cpus-per-task=16',
                             'export CONDADIR=/storage/hpc/46/manders3/conda4/open-ce',
                             'export NCCL_SOCKET_IFNAME=enp0s31f6', ])
    sub_commands.extend(['#SBATCH --{}={}\n'.format(cmd, value) for (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        'export WANDB_API_KEY=9cf7e97e2460c18a89429deed624ec1cbfb537bc',

        'source /etc/profile',
        'module add opence',
        'conda activate $CONDADIR',  # ...and activate the conda environment
    ])
    script_name = os.path.realpath(sys.argv[0])  # Find this scripts name...
    trialArgs = __get_hopt_params(trialconfig)

    sub_commands.append('srun {} {} {}'.format(comm, script_name, trialArgs))
    sub_commands = [x.lstrip() for x in sub_commands]

    full_command = '\n'.join(sub_commands)
    return full_command


def __get_hopt_params(trial):
    """
    Converts hyperparameter optimization trials into script parameters.

    Args:
        trial (object): Hyperparameter optimization trial.

    Returns:
        str: Command-line arguments for the trial.
    """
    params = []
    trial.__dict__.update({"HPC": True})
    for k in trial.__dict__:
        v = trial.__dict__[k]
        if k == 'num_trials':
            v = 0
        if v is None or v is False:
            continue

        if __should_escape(v):
            cmd = '--{} \"{}\"'.format(k, v)
        else:
            cmd = '--{} {}'.format(k, v)
        params.append(cmd)

    full_cmd = ' '.join(params)
    return full_cmd


def __should_escape(v):
    """
    Determines if a value should be escaped in the command line.

    Args:
        v (str): Value to check.

    Returns:
        bool: True if the value should be escaped, False otherwise.
    """
    v = str(v)
    return '[' in v or ';' in v or ' ' in v


class baseparser(HyperOptArgumentParser):
    """
    A custom argument parser for hyperparameter optimization.

    Args:
        *args: Positional arguments.
        strategy (str): Optimization strategy (e.g., 'random_search').
        **kwargs: Additional keyword arguments.

    Attributes:
        keys_of_interest (list): List of keys to track during optimization.
    """
    def __init__(self, *args, strategy="random_search", **kwargs):
        super().__init__(*args, strategy=strategy, add_help=False)
        self.add_argument("--dir", default=os.path.join(os.getenv("global_scratch", "/data"), "data"), type=str, )
        self.opt_list("--learning_rate", default=0.001, type=float, options=[2e-3, 1e-4, 5e-5], tunable=True)
        self.opt_list("--embed_dim", default=64, type=int, options=[64, 256, 128, 512], tunable=True)
        self.opt_list("--HPC", default=os.getenv("HPC", False), type=bool, tunable=False)
        self.opt_list("--batch_size", default=2, type=int, options=[4, 8, 10], tunable=True)
        self.opt_list("--MINIOHost", type=str, default="10.45.1.250", tunable=False)
        self.opt_list("--MINIOPort", type=int, default=9000, tunable=False)
        self.opt_list("--MINIOAccesskey", type=str, default="minioadmin", tunable=False)
        self.opt_list("--MINIOSecret", type=str, default="minioadmin", tunable=False)
        self.opt_list("--bucket_name", type=str, default="convmap", tunable=False)
        self.opt_list("--preProcess", type=bool, default=False, tunable=False)
        self.opt_list("--time_step", type=int, default=3, options=[1, 2, 3, 4], tunable=False)
        self.opt_list("--grid_size", type=int, default=480, options=[120, 300, 480], tunable=True)
        self.opt_list("--data_dir", type=str, default=os.path.join(os.getenv("global_scratch", "/data"), "convmap_data"),
                      tunable=False)
        self.opt_list("--method", type=str, default="grid", options=["grid"], tunable=True)
        self.opt_list("--WindowsMinutes", type=int, default=20, options=[10, 20, 30, 60, 90, 120, 240], tunable=True)
        self.opt_list("--cache_first", type=bool, default=True, tunable=False)
        self.opt_list("--mlp_ratio", type=int, default=2, options=[2, 3, 4], tunable=True)
        self.opt_list("--noise_factor", type=float, default=0.1, options=[0.0, 0.01, 0.05, 0.1, 0.2, 0.005], tunable=True)
        self.opt_list("--precision", default='16-mixed', options=[16], tunable=False)
        self.opt_list("--accelerator", default='auto', type=str, options=['gpu'], tunable=False)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)
        self.opt_list("--fast_dev_run", default=False, type=bool, tunable=False)
        self.keys_of_interest = ["dir", "learning_rate", "batch_size", "precision", "grid_size", "mlp_ratio",
                                 "accelerator", "num_trials", "WindowsMinutes", "embed_dim"]


class parser(baseparser):
    """
    Extends baseparser to handle Neptune and Weights & Biases trials.

    Args:
        *args: Positional arguments.
        strategy (str): Optimization strategy (e.g., 'random_search').
        **kwargs: Additional keyword arguments.

    Methods:
        generate_trials: Generates hyperparameter optimization trials.
        generate_neptune_trials: Generates trials using Neptune API.
        generate_wandb_trials: Generates trials using Weights & Biases API.
    """
    def __init__(self, *args, strategy="random_search", **kwargs):
        super().__init__(*args, strategy=strategy, add_help=False, **kwargs)
        self.run_configs = set()
        self.keys = set()

    def generate_trials(self):
        hyperparams = self.parse_args()
        NumTrials = hyperparams.num_trials if hyperparams.num_trials > 0 else 1
        trials = super().generate_trials(NumTrials)
        return trials

    def generate_neptune_trials(self, project):
        import neptune
        from neptunecontrib.api import search_runs

        neptune.init(project_qualified_name=project)
        runs = search_runs(project)
        print("checking prior runs")
        for run in tqdm(runs):
            config = run.get_parameters()
            sortedkeys = list([str(i) for i in config.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values = list([str(config[i]) for i in sortedkeys])
            code = "_".join(values)
            self.run_configs.add(code)
        hyperparams = self.parse_args()
        NumTrials = hyperparams.num_trials if hyperparams.num_trials > 0 else 1
        trials = hyperparams.generate_trials(NumTrials)
        print("checking if already done...")
        trial_list = []
        for trial in tqdm(trials):
            sortedkeys = list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values = list([str(trial.__dict__[k]) for k in sortedkeys])

            code = "_".join(values)
            while code in self.run_configs:
                trial = hyperparams.generate_trials(1)[0]
                sortedkeys = list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
                sortedkeys.sort()
                values = list([str(trial.__dict__[k]) for k in sortedkeys])
                code = "_".join(values)
            trial_list.append(trial)
        return trial_list

    def generate_wandb_trials(self, entity, project):
        api = wandb.Api()

        runs = api.runs(entity + "/" + project)
        print("checking prior runs")
        for run in tqdm(runs):
            config = run.config
            sortedkeys = list([str(i) for i in config.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values = list([str(config[i]) for i in sortedkeys])
            code = "_".join(values)
            self.run_configs.add(code)
        hyperparams = self.parse_args()
        NumTrials = hyperparams.num_trials if hyperparams.num_trials > 0 else 1
        trials = hyperparams.generate_trials(NumTrials)
        print("checking if already done...")
        trial_list = []
        for trial in tqdm(trials):

            sortedkeys = list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values = list([str(trial.__dict__[k]) for k in sortedkeys])

            code = "_".join(values)
            while code in self.run_configs:
                trial = hyperparams.generate_trials(1)[0]
                sortedkeys = list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
                sortedkeys.sort()
                values = list([str(trial.__dict__[k]) for k in sortedkeys])

                code = "_".join(values)
            trial_list.append(trial)
        return trial_list


if __name__ == "__main__":
    from subprocess import call

    myparser = parser()
    hyperparams = myparser.parse_args()

    defaultConfig = hyperparams

    NumTrials = hyperparams.num_trials
    if NumTrials == -1:
        trial = None
        try:
            while True:
                trial = hyperparams.generate_trials()[0]
                print("Running trial: {}".format(trial))
                wandbtrain(trial)
        except Exception as e:
            print("Error running trial: {}".format(e))
            wandb.finish()

    elif NumTrials == 0 and not str(os.getenv("HOSTNAME", "localhost")).startswith("login"):
        if os.getenv("WANDB_API_KEY"):
            wandbtrain(defaultConfig)
        elif os.getenv("NEPTUNE_API_TOKEN"):
            print("NEPTUNE API KEY found")
            neptunetrain(defaultConfig)
        else:
            print("No logging API found, using default config")
            train(defaultConfig.__dict__)
    else:
        if os.getenv("WANDB_API_KEY"):
            print("WANDB API KEY found")
            trials = myparser.generate_wandb_trials(WANDBUSER, YOURPROJECTNAME)
        elif os.getenv("NEPTUNE_API_TOKEN"):
            print("NEPTUNE API KEY found")
            trials = myparser.generate_neptune_trials(NEPTUNEUSER, YOURPROJECTNAME)
        else:
            print("No logging API found, using default config")
            trials = hyperparams.generate_trials()
        for i, trial in enumerate(trials):
            command = SlurmRun(trial)
            slurm_cmd_script_path = os.path.join(defaultConfig.__dict__.get("dir", "."),
                                                 "slurm_cmdtrial{}.sh".format(i))
            os.makedirs(defaultConfig.__dict__.get("dir", "."), exist_ok=True)
            with open(slurm_cmd_script_path, "w") as f:
                f.write(command)
            print('\nlaunching exp...')
            result = call('{} {}'.format("sbatch", slurm_cmd_script_path), shell=True)
            if result == 0:
                print('launched exp ', slurm_cmd_script_path)
            else:
                print('launch failed...')

