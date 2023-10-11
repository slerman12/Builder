from tributaries import my_sweep, my_plots, my_checkpoints


# Launching

# List of hyperparams to launch
my_sweep.hyperparams = [
    # Object detector
    f"""task=regression
    experiment=Bittle
    dataset=World.Datasets.YouTube.YouTube
    dataset.url='https://youtu.be/M6Vu_FHUvAs'
    dataset.transform='transforms.Resize([302,170],antialias=None)'
    dataset.aug=Agents.Blocks.Architectures.Vision.FoundationModels.GroundingDINO.AutoLabel
    dataset.aug.caption='little robot dog'
    test_dataset.url='https://youtu.be/_XpAAd0lr9E'
    aug=Identity
    action_spec.shape='(4,)' 
    batch_size=256
    """,
]

# Plotting

my_plots.plots.append(['Bittle'])  # Lists of experiments to plot together, supports regex .*!
my_plots.title = 'Bittle Object Detector Automatic Distillation'

# Checkpoints

my_checkpoints.experiments = my_plots.plots[0]
