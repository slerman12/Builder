from tributaries import my_sweep, my_plots


# Launching

# List of hyperparams to launch
my_sweep.hyperparams = [
    # Large + RRUFF, No-Pool-CNN, 7-Way Crystal Systems & 230-Way Space Groups
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

my_plots.plots.append(['ImageNet'])  # Lists of experiments to plot together
my_plots.title = 'ResNet50 On ImageNet-mini'
