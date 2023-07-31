<picture>
  <source width="20%" media="(prefers-color-scheme: dark)" srcset="https://github.com/AGI-init/Assets/assets/92597756/e411328b-a51a-416c-ba97-7b7939ec3351">
  <img width="20%" alt="Text changing depending on mode. Light: 'Light' Dark: 'Dark'" src="https://github.com/AGI-init/Assets/assets/92597756/f3df44c8-b989-4951-9443-d2b4203b5c4e">
<br><br>
</picture>

# Welcome

See our library [Tributaries](../../../tributaries-ml/src/tributaries) for mass-deploying UnifiedML apps on remote servers.

Check out [minihydra / leviathan](../../../minihydra/src/minihydra) for how we handle sys args & hyperparams.

## Install

```console
pip install UnifiedML
```

# What is UnifiedML?

<p align="center">
<a href="https://github.com/AGI-init/Assets/assets/92597756/d92e6b3f-9625-427c-87ef-909b3ec40f08">
<picture>
  <source width="40%" media="(prefers-color-scheme: dark)" srcset="https://github.com/AGI-init/Assets/assets/92597756/f8b74f97-7a5a-4643-b08d-a23f8305b5b8">
  <img width="40%" alt="Text changing depending on mode. Light: 'Light' Dark: 'Dark'" src="https://github.com/AGI-init/Assets/assets/92597756/d92e6b3f-9625-427c-87ef-909b3ec40f08">
<br><br>
</picture>
</a>
</p>

UnifiedML is a toolbox & engine for defining ML tasks and training them individually, or together in a single general intelligence.

# Generalist agent example

As a fun little example, let's train a neural network to simultaneously learn to play Super Mario Bros. and classify ImageNet:

```console
ML multi_task='["task=RL Env=Mario", "Dataset=ImageNet dataset.root=imagenet/"]' Eyes=ResNet50
```

[See full docs here.]()

#

By [Sam Lerman](https://www.github.com/slerman12).