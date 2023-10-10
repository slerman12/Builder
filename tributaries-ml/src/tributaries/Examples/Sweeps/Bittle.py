from tributaries import my_sweep, my_plots, my_checkpoints


# Launching

# List of hyperparams to launch
my_sweep.hyperparams = [
    # Object detector
    # f"""task=regression
    # experiment=Bittle
    # dataset=World.Datasets.YouTube.YouTube
    # dataset.file=bittle_yt_detector_train.txt
    # dataset.resolution=(302,170)
    # dataset.fps=10
    # test_dataset=World.Datasets.YouTube.YouTube
    # test_dataset.file=bittle_yt_detector_test.txt
    # test_dataset.resolution=(302,170)
    # test_dataset.fps=10
    # aug=Agents.Blocks.Architectures.Vision.FoundationModels.GroundingDINO.AutoLabel
    # aug.caption='little robot dog'
    # batch_size=16
    # """,

    # Object detector
    f"""task=regression
    experiment=Bittle
    dataset=World.Datasets.YouTube.YouTube
    dataset.url='https://youtu.be/M6Vu_FHUvAs'
    dataset.transform=Sequential
    dataset.transform._targets_='["transforms.Resize([302,170])","Agents.Blocks.Architectures.Vision.FoundationModels.GroundingDINO.AutoLabel"]'
    dataset.transform.caption='little robot dog'
    dataset.fps=10
    action_spec.shape='(4,)' 
    batch_size=1
    evaluate_per_steps=0
    stream=true
    """,
]

# Plotting

my_plots.plots.append(['Bittle'])  # Lists of experiments to plot together, supports regex .*!
my_plots.title = 'Bittle Object Detector Automatic Distillation'

# Checkpoints

my_checkpoints.experiments = my_plots.plots[0]
