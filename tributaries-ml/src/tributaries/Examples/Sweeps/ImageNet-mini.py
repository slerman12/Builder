from tributaries import my_sweep, my_plots, my_checkpoints


# Launching

# List of hyperparams to launch
my_sweep.hyperparams = [
    # 6e4 ImageNet examples in RAM, no hard disk reformatting/loading
    f"""task=classify/imagenet
    Eyes=ResNet50
    experiment=ImageNet-mini
    ram_capacity=6e4
    hd_capacity=0
    dataset.transform='transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64)])'
    dataset.root=/home/cxu-serve/p1/datasets/imagenet/
    """
]

my_sweep.mem = 30

# Plotting

my_plots.plots.append(['ImageNet-mini'])  # Lists of experiments to plot together, supports regex .*!
my_plots.title = 'ResNet50 On ImageNet-mini'

# Checkpoints

my_checkpoints.experiments = my_plots.plots[0]
