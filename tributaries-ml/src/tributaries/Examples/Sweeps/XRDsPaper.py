from tributaries import my_sweep, my_plots, my_checkpoints


# Launching

# List of hyperparams to launch
my_sweep.hyperparams = [
    # Large + RRUFF, No-Pool-CNN, 7-Way Crystal Systems & 230-Way Space Groups
    f"""task=NPCNN
    num_classes={num_classes}
    save_per_steps=1e5
    ram_capacity=2e6
    """ for num_classes in (7, 230)
]

my_sweep.app = 'XRDs'
my_sweep.branch = 'Dev'

my_sweep.mem = 60

# Plotting

my_plots.plots.append(['NPCNN'])  # Lists of experiments to plot together, supports regex .*!
my_plots.title = 'Disjoint 50% RRUFF - NPCNN - Trained on synthetic + 50% RRUFF'

# Checkpoints

my_checkpoints.experiments = my_plots.plots[0]
