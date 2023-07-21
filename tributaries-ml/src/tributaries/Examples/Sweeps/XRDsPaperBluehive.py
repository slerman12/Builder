from tributaries import my_sweep, my_plots


# Launching

# List of hyperparams to launch
my_sweep.hyperparams = [
    # Large + RRUFF, No-Pool-CNN
    f"""task=NPCNN
    num_classes={num_classes}
    train_steps=5e5
    save_per_steps=1e5
    dataset.sources='["/gpfs/fs2/scratch/public/jsalgad2/Data/Generated/XRDs_ICSD/","./Data/Generated/XRDs_RRUFF/"]'
    dataset.train_eval_splits='[1, 0.5]'
    Dataset=XRD.XRD
    ram_capacity=2e6
    stream=false""" for num_classes in (7, 230)
]

my_sweep.app = 'XRDs'
my_sweep.branch = 'Dev'

my_sweep.mem = 80

# Plotting

my_plots.plots.append(['NPCNN'])  # Lists of experiments to plot together
my_plots.title = 'Disjoint 50% RRUFF - NPCNN - Trained on synthetic + 50% RRUFF'

