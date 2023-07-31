from tributaries import my_sweep, my_plots, my_checkpoints


# Launching

# List of hyperparams to launch
my_sweep.hyperparams = [
    f"""task=atari/pong
    experiment=Pong
    log_media=true
    """,
]

my_sweep.mem = 60

# Plotting

my_plots.plots.append(['Pong'])  # Lists of experiments to plot together, supports regex .*!
my_plots.title = 'Atari Pong'

# Checkpoints

my_checkpoints.experiments = my_plots.plots[0]
