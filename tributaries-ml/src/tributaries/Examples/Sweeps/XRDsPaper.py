from tributaries import my_sweep, my_plots

# Launching


# List of hyperparams to launch
my_sweep.hyperparams = [
    # Large + RRUFF, No-Pool-CNN
    """task=npcnn
    task_name='${num_classes}-Way_ICSD-true_Open-Access-false_RRUFF-true_Soup-true'
    num_classes=7
    train_steps=5e5
    save_per_steps=1e5
    +'dataset.roots=["/gpfs/fs2/scratch/public/jsalgad2/icsd1.2m_large/","./Data/Generated/XRDs_RRUFF/"]'
    +'dataset.train_eval_splits=[1, 0.5]'
    stream=false
    num_workers=6
    num_gpus=1
    mem=80""",
]

my_sweep.app = 'XRDs'
my_sweep.branch = 'Dev'

# Plotting

# Lists of experiments to plot together
my_plots.plots.append(['NPCNN'])

my_plots.title = 'Disjoint 50% RRUFF - NPCNN - Trained on synthetic + 50% RRUFF'
