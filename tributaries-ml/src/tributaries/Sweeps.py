# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from minihydra import Args


my_sweep = Args(
    # List of hyperparams (strings)
    hyperparams=[],
    # Directory & GitHub parameters
    app=None,
    branch=None,
    level=1,  # Directory level for naming plotting directories
    # SLURM parameters
    num_cpus=None,  # Will default to num_workers + 1 of ML script
    num_gpus=1,
    gpu='K80|V100|A100|RTX',
    mem=20,
    time='3-00:00:00',
    reservation_id=None,
    wandb_key=None,
    pseudonym=None  # Naming protocol for SLURM tasks; can apply minihydra interpolation syntax w/ UnifiedML args
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

my_checkpoints = Args(
    experiments=[],  # List of experiments (strings)
)
