from test_tube import HyperOptArgumentParser
import wandb
import torch
from tqdm import tqdm
import datetime
import pytorch_lightning
from pytorch_lightning.callbacks import TQDMProgressBar,EarlyStopping
import datetime
import os,sys
from model import Pangu
from DataModule import DatasetFromMinioBucket
YOURPROJECTNAME="TestDeploy"
WANDBUSER="st7ma784"
NEPTUNEUSER="st7ma784"

def train(config={
        "batch_size":16, # ADD MODEL ARGS HERE
         "codeversion":"-1",
    },dir=None,devices=None,accelerator=None,Dataset=None,logtool=None,EvalOnLaunch=False):


    #### EDIT HERE FOR DIFFERENT VERSIONS OF A MODEL
    
    minioClient = {"host": config["MINIOHost"], "port": config["MINIOPort"], "access_key": config["MINIOAccesskey"]
                    , "secret_key": config["MINIOSecret"]}
    config.update({"minioClient": minioClient})
    model=Pangu(**config)
    dataModule=DatasetFromMinioBucket(**config)

    print("building model")
    if devices is None:
        devices=config.get("devices","auto")
    if accelerator is None:
        accelerator=config.get("accelerator","auto")
    # print("Training with config: {}".format(config))
    filename="model-{}".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    callbacks=[
        TQDMProgressBar(),
        EarlyStopping(monitor="train_loss", mode="min",patience=10,check_finite=True,stopping_threshold=0.001),
        #save best model
        pytorch_lightning.callbacks.ModelCheckpoint(
            monitor='train_loss',
            dirpath=dir,
            filename=filename,
            save_top_k=1,
            mode='min',
            save_last=True,),
    ]
    p=config['precision']
    if isinstance(p,str):
        p=16 if p=="bf16" else int(p)  ##needed for BEDE

    trainer=pytorch_lightning.Trainer(
            devices= 1,
            num_nodes= 1,
            accelerator= accelerator,
            max_epochs=200,
            #profiler="advanced",
            #plugins=[SLURMEnvironment()],
            #https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
            logger=logtool,
            # strategy='ddp_find_unused_parameters_true', #given the size, we might need to split the model between GPUs instead of 
            # FSDPStrategy(accelerator="gpu",
            #                        parallel_devices= 1,
            #                        cluster_environment=SLURMEnvironment(),
            #                        timeout=datetime.timedelta(seconds=1800),
            #                        #cpu_offload=True,
            #                        #mixed_precision=None,
            #                        #auto_wrap_policy=True,
            #                        #activation_checkpointing=True,
            #                        #sharding_strategy='FULL_SHARD',
            #                        #state_dict_type='full'
            # ),
            callbacks=callbacks,
            gradient_clip_val=0.25,# Not supported for manual optimization
            fast_dev_run=config.get("fast_dev_run", False),
            precision=p
    )
    # model=torch.compile(model,mode="reduce-overhead",fullgraph=True)

    trainer.fit(model,dataModule)


#### This is a wrapper to make sure we log with Weights and Biases, You'll need your own user for this.
def wandbtrain(config=None,dir=None,devices=None,accelerator=None,Dataset=None):

    if config is not None:
        config=config.__dict__
        dir=config.get("dir",dir)
        logtool= pytorch_lightning.loggers.WandbLogger( project="TestDeploy",entity="st7ma784", save_dir=os.getenv("global_scratch",dir),name="TestDeploy")
        print(config)

    else:
        #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
        print("Would recommend changing projectname according to config flags if major version swithching happens")
        run=wandb.init(project="TestDeploy",entity="st7ma784",name="TestDeploy",config=config)
        logtool= pytorch_lightning.loggers.WandbLogger( project="TestDeploy",entity="st7ma784",experiment=run, save_dir=os.getenv("global_scratch",dir))
        config=run.config.as_dict()

    train(config,dir,devices,accelerator,Dataset,logtool)


def neptunetrain(config=None,dir=None,devices=None,accelerator=None,Dataset=None):
    import neptune
    from neptunecontrib.api import search_runs

    import pytorch_lightning
    if config is not None:
        config=config.__dict__
        dir=config.get("dir",dir)
        logtool= pytorch_lightning.loggers.NeptuneLogger( project="TestDeploy",entity="st7ma784", save_dir=dir)
        print(config)

    else:
        #We've got no config, so we'll just use the default, and hopefully a trainAgent has been passed
        print("Would recommend changing projectname according to config flags if major version swithching happens")
        run=neptune.init(project="TestDeploy",entity="st7ma784",name="TestDeploy",config=config)
        logtool= pytorch_lightning.loggers.NeptuneLogger( project="TestDeploy",entity="st7ma784",experiment=run, save_dir=dir)
        config=run.config.as_dict()

    train(config,dir,devices,accelerator,Dataset,logtool)


def SLURMEval(ModelPath,config):
    job_with_version = '{}v{}'.format("EVAL", 0)
    sub_commands =['#!/bin/bash',
        '#SBATCH --time={}'.format( '24:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version),
        '#SBATCH --nodes=1',
        '#SBATCH --ntasks-per-node=1',
        '#SBATCH --gres=gpu:1',
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format('st7ma784@gmail.com'),]
    comm="python"
    slurm_commands={}

    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
                '#SBATCH --account MYACOCUNT',
                'export CONDADIR=/nobackup/projects/<BEDEPROJECT>/$USER/miniconda',
                'export NCCL_SOCKET_IFNAME=ib0'])
        comm="python3"
    else:

        sub_commands.extend(['#SBATCH -p gpu-medium',
                             '#SBATCH --mem=96G',
                             'export CONDADIR=/storage/hpc/46/manders3/conda4/open-ce',
                             'export NCCL_SOCKET_IFNAME=enp0s31f6',])
    sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        'source /etc/profile',
        'module add opence',
        'conda activate $CONDADIR',# ...and activate the conda environment
    ])




def SlurmRun(trialconfig):

    job_with_version = '{}v{}'.format("SINGLEGPUTESTLAUNCH", 0)

    sub_commands =['#!/bin/bash',
        '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',
        '#SBATCH --time={}'.format( '24:00:00'),# Max run time
        '#SBATCH --job-name={}'.format(job_with_version),
        '#SBATCH --nodes=1',  #Nodes per experiment
        '#SBATCH --ntasks-per-node=1',# Set this to GPUs per node.
        '#SBATCH --gres=gpu:1',  #{}'.format(per_experiment_nb_gpus),
        f'#SBATCH --signal=USR1@{5 * 60}',
        '#SBATCH --mail-type={}'.format(','.join(['END','FAIL'])),
        '#SBATCH --mail-user={}'.format('st7ma784@gmail.com'),
    ]
    comm="python"
    slurm_commands={}

    if str(os.getenv("HOSTNAME","localhost")).endswith("bede.dur.ac.uk"):
        sub_commands.extend([
                '#SBATCH --account MYACOCUNT',
                '''
                arch=$(uname -i) # Get the CPU architecture
                if [[ $arch == "aarch64" ]]; then
                   # Set variables and source scripts for aarch64
                   export CONDADIR=/nobackup/projects/<project>/$USER/ # Update this with your <project> code.
                   source $CONDADIR/aarchminiconda/etc/profile.d/conda.sh
                fi
                '''
                'export CONDADIR=/nobackup/projects/<BEDEPROJECT>/$USER/miniconda',
                'export NCCL_SOCKET_IFNAME=ib0'])
        comm="python3"
    else:

        sub_commands.extend(['#SBATCH -p gpu-medium',
                             'export CONDADIR=/storage/hpc/46/manders3/conda4/open-ce',
                             'export NCCL_SOCKET_IFNAME=enp0s31f6',])
    sub_commands.extend([ '#SBATCH --{}={}\n'.format(cmd, value) for  (cmd, value) in slurm_commands.items()])
    sub_commands.extend([
        'export SLURM_NNODES=$SLURM_JOB_NUM_NODES',
        'export wandb=9cf7e97e2460c18a89429deed624ec1cbfb537bc',
        'source /etc/profile',
        'module add opence',
        'conda activate $CONDADIR',# ...and activate the conda environment
    ])
    script_name= os.path.realpath(sys.argv[0]) #Find this scripts name...
    trialArgs=__get_hopt_params(trialconfig)
    #If you're deploying prototyping code and often changing your pip env,
    # consider adding in a 'scopy requirements.txt
    # and then append command 'pip install -r requirements.txt...
    # This should add your pip file from the launch dir to the run location, then install on each node.

    sub_commands.append('srun {} {} {}'.format(comm, script_name,trialArgs))
    #when launched, this script will be called with no trials, and so drop into the wandbtrain section,
    sub_commands = [x.lstrip() for x in sub_commands]

    full_command = '\n'.join(sub_commands)
    return full_command

def __get_hopt_params(trial):
    """
    Turns hopt trial into script params
    :param trial:
    :return:
    """
    params = []
    trial.__dict__.update({"HPC":True})
    for k in trial.__dict__:
        v = trial.__dict__[k]
        if k == 'num_trials':
            v=0
        # don't add None params
        if v is None or v is False:
            continue

        # put everything in quotes except bools
        if __should_escape(v):
            cmd = '--{} \"{}\"'.format(k, v)
        else:
            cmd = '--{} {}'.format(k, v)
        params.append(cmd)

    # this arg lets the hyperparameter optimizer do its thin
    full_cmd = ' '.join(params)
    return full_cmd

def __should_escape(v):
    v = str(v)
    return '[' in v or ';' in v or ' ' in v

class baseparser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search
        self.add_argument("--dir",default="/data/data",type=str,)
        self.opt_list("--learning_rate", default=0.00001, type=float, options=[2e-4,1e-4,5e-5,1e-5,4e-6], tunable=True)
        self.opt_list("--embed_dim", default=64, type=int, options=[64,256,128,512,1024], tunable=True)
        
        self.opt_list("--HPC", default=os.getenv("HPC",False), type=bool, tunable=False)
        self.opt_list("--batch_size", default=6, type=int,options=[4,8,16,32,64],tunable=True)
        self.opt_list("--MINIOHost", type=str, default="10.48.163.59", tunable=False)
        self.opt_list("--MINIOPort", type=int, default=9000, tunable=False)
        self.opt_list("--MINIOAccesskey", type=str, default="minioadmin", tunable=False)
        self.opt_list("--MINIOSecret", type=str, default="minioadmin", tunable=False)
        self.opt_list("--bucket_name", type=str, default="convmap", tunable=False)
        self.opt_list("--preProcess", type=bool, default=False, tunable=False)
        self.opt_list("--time_step", type=int, default=1,options=[1,2,3,4,5,6,7], tunable=False)#currently not implemented in model
        self.opt_list("--grid_size", type=int, default=300,options=[100,300,500,1000], tunable=True)
        self.opt_list("--data_dir", type=str, default=os.path.join(os.getenv("global_scratch","/data"),"convmap_data"), tunable=False)
        self.opt_list("--method", type=str, default="grid",options=["flat","grid"], tunable=True)
        self.opt_list("--WindowsMinutes", type=int, default=40,options=[10,20,30,60,90,120,240], tunable=True) #The number of minutes each snapshot represents
        self.opt_list("--cache_first", type=bool, default=True, tunable=False)
        self.opt_list("--mlp_ratio", type=int, default=2, options=[2,3,4,8], tunable=True)
        self.opt_list("--noise_factor", type=float, default=0.0, options=[0.0,0.01,0.05,0.1,0.2,0.3], tunable=True)
        #INSERT YOUR OWN PARAMETERS HERE
        self.opt_list("--precision", default=16, options=[32], tunable=False)
        self.opt_list("--accelerator", default='auto', type=str, options=['gpu'], tunable=False)
        self.opt_list("--num_trials", default=0, type=int, tunable=False)
        self.opt_list("--fast_dev_run", default=False, type=bool, tunable=False)
        #self.opt_range('--neurons', default=50, type=int, tunable=True, low=100, high=800, nb_samples=8, log_base=None)
        #This is important when passing arguments as **config in launcher
        self.keys_of_interest=["dir","learning_rate","batch_size","precision","grid_size","mlp_ratio","accelerator","num_trials","WindowsMinutes","embed_dim"]
        # self.argNames=["dir","learning_rate","batch_size","precision","grid_size","mlp_ratio","accelerator","num_trials","WindowsMinutes","embed_dim"]
    # def __dict__(self):
    #     return {k:self.parse_args().__dict__[k] for k in self.argNames}



class parser(baseparser):
    def __init__(self,*args,strategy="random_search",**kwargs):

        super().__init__( *args,strategy=strategy, add_help=False,**kwargs) # or random search
        self.run_configs=set()
        self.keys=set()
    def generate_trials(self):
        hyperparams = self.parse_args()
        NumTrials=hyperparams.num_trials if hyperparams.num_trials>0 else 1
        trials=super().generate_trials(NumTrials)
        return trials
    def generate_neptune_trials(self,project):
        #this function uses the nepune api to get the trials that exist, 
        #and then generates new trials based on the hyperparameters

        neptune.init(project_qualified_name=project)
        runs = search_runs(project)
        print("checking prior runs")
        for run in tqdm(runs):
            config=run.get_parameters()
            sortedkeys=list([str(i) for i in config.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(config[i]) for i in sortedkeys])
            code="_".join(values)
            self.run_configs.add(code)
        hyperparams = self.parse_args()
        NumTrials=hyperparams.num_trials if hyperparams.num_trials>0 else 1
        trials=hyperparams.generate_trials(NumTrials)
        print("checking if already done...")
        trial_list=[]
        for trial in tqdm(trials):
            sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
            code="_".join(values)
            while code in self.run_configs:
                trial=hyperparams.generate_trials(1)[0]
                sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
                sortedkeys.sort()
                values=list([str(trial.__dict__[k]) for k in sortedkeys])
                code="_".join(values)
            trial_list.append(trial)
        return trial_list

    def generate_wandb_trials(self,entity,project):
        api = wandb.Api()

        runs = api.runs(entity + "/" + project)
        print("checking prior runs")
        for run in tqdm(runs):
            config=run.config
            sortedkeys=list([str(i) for i in config.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(config[i]) for i in sortedkeys])
            code="_".join(values)
            self.run_configs.add(code)
        hyperparams = self.parse_args()
        NumTrials=hyperparams.num_trials if hyperparams.num_trials>0 else 1
        trials=hyperparams.generate_trials(NumTrials)
        print("checking if already done...")
        trial_list=[]
        for trial in tqdm(trials):

            sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
            sortedkeys.sort()
            values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
            code="_".join(values)
            while code in self.run_configs:
                trial=hyperparams.generate_trials(1)[0]
                sortedkeys=list([str(i) for i in trial.__dict__.keys() if i in self.keys_of_interest])
                sortedkeys.sort()
                values=list([str(trial.__dict__[k]) for k in sortedkeys])
            
                code="_".join(values)
            trial_list.append(trial)
        return trial_list
        
# Testing to check param outputs
if __name__== "__main__":
    from subprocess import call

    myparser=parser()
    hyperparams = myparser.parse_args()

    defaultConfig=hyperparams

    NumTrials=hyperparams.num_trials
    #BEDE has Env var containing hostname  #HOSTNAME=login2.bede.dur.ac.uk check we arent launching on this node
    if NumTrials==-1:
        #debug mode - We want to just run in debug mode...
        #pick random config and have at it!

        trial=hyperparams.generate_trials()[0]
        #We'll grab a random trial, BUT have to launch it with KWARGS, so that DDP works.
        #result = call('{} {} --num_trials=0 {}'.format("python",os.path.realpath(sys.argv[0]),__get_hopt_params(trial)), shell=True)

        print("Running trial: {}".format(trial))

        wandbtrain(trial)

    elif NumTrials ==0 and not str(os.getenv("HOSTNAME","localhost")).startswith("login"): #We'll do a trial run...
        #means we've been launched from a BEDE script, so use config given in args///
        
        if os.getenv("WANDB_API_KEY"):

            wandbtrain(defaultConfig)
        elif os.getenv("NEPTUNE_API_TOKEN"):
            print("NEPTUNE API KEY found")
            neptunetrain(defaultConfig)
        else:
            print("No logging API found, using default config")
            train(defaultConfig.__dict__)
    #OR To run with Default Args
    else:
        # check for wandb login details in env vars
        if os.getenv("WANDB_API_KEY"):
            print("WANDB API KEY found")
            trials= myparser.generate_wandb_trials(WANDBUSER,YOURPROJECTNAME)
        #check for neptune login details in env vars
        elif os.getenv("NEPTUNE_API_TOKEN"):
            print("NEPTUNE API KEY found")
            trials= myparser.generate_neptune_trials(NEPTUNEUSER,YOURPROJECTNAME)
        else:
            print("No logging API found, using default config")
            trials= hyperparams.generate_trials()  
        for i,trial in enumerate(trials):
            command=SlurmRun(trial)
            slurm_cmd_script_path =  os.path.join(defaultConfig.__dict__.get("dir","."),"slurm_cmdtrial{}.sh".format(i))
            os.makedirs(defaultConfig.__dict__.get("dir","."),exist_ok=True)
            with open(slurm_cmd_script_path, "w") as f:
                f.write(command)
            print('\nlaunching exp...')
            result = call('{} {}'.format("sbatch", slurm_cmd_script_path), shell=True)
            if result == 0:
                print('launched exp ', slurm_cmd_script_path)
            else:
                print('launch failed...')
    
