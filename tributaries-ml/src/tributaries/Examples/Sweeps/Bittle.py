from tributaries import my_sweep, my_plots, my_checkpoints


# Launching

# List of hyperparams to launch
my_sweep.hyperparams = [
    # Object detector
    f"""task=regression
    experiment=Bittle
    dataset=World.Datasets.YouTube.YouTube
    dataset.file=bittle_yt_detector_train.txt
    dataset.resolution=(302,170)
    dataset.fps=10
    test_dataset=World.Datasets.YouTube.YouTube
    test_dataset.file=bittle_yt_detector_test.txt
    test_dataset.resolution=(302,170)
    test_dataset.fps=10
    aug=Agents.Blocks.Architectures.Vision.FoundationModels.GroundingDINO.AutoLabel
    aug.caption='little robot dog'
    batch_size=16
    """,
]

# Plotting

my_plots.plots.append(['Bittle'])  # Lists of experiments to plot together, supports regex .*!
my_plots.title = 'Bittle Object Detector Automatic Distillation'

# Checkpoints

my_checkpoints.experiments = my_plots.plots[0]
