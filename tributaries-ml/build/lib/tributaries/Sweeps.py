from ML.Hyperparams.minihydra import Args


my_sweep = Args(
    # List of hyperparams (strings)
    hyperparams=[],
    # Directory & GitHub parameters
    app=None,
    branch=None,
    # SLURM parameters
    num_gpus=1,
    gpu='K80|V100|A100|RTX',
    mem=20,
    time='3-00:00:00',
    reservation_id=None,
    wandb_key=None,
    pseudonym=None  # Naming protocol for SLURM tasks; can use minihydra interpolation syntax with UnifiedML args
)

my_plots = Args(
    plots=[],  # List of lists of strings (list of grouped plots)
    sftp=True,
    write_tabular=False,
    steps=None,
    title='UnifiedML',
    x_axis='Step',
    tasks=[],
    agents=[],
    suites=[]
)
