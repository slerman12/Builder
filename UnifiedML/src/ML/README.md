<img width="20%" src="https://github.com/AGI-init/Assets/assets/92597756/f3df44c8-b989-4951-9443-d2b4203b5c4e"><br><br>

# Welcome

See our library [Tributaries](../../../tributaries/src/tributaries) for mass-deploying UnifiedML apps on remote servers.

Check out [minihydra / leviathan]() for how we handle sys args & hyperparams.

## Install

```console
pip install UnifiedML
```

If you're on Linux, you must first manually install Pytorch: ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu```.

## What is UnifiedML?

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

## Quick start

Wherever you run ```ML```, it'll search the current directory for any specified paths.

Paths to architectures, agents, environments, etc. via dot notation:
```console
ML Model=MyFile.model
``` 
or regular directory paths:
```console
ML Model=./MyFile.py.model
```

### Training example

```python
# Run.py

from torch import nn

model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))
```

**Run:**

```console
ML Model=Run.model Dataset=CIFAR10
```

There are many [built-in](#built-ins) datasets, architectures, and so on, such as CIFAR10.

### Equivalent pure-code training example

```python
# Run.py

from torch import nn

from ML import main

model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))

if __name__ == '__main__':
    main(Model=model, Dataset='CIFAR10')
```

**Run:**

```console
python Run.py
```

### If you're feeling brave, this also works:

Not exactly scalable, but:

```console
ML Model='nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))' Dataset=CIFAR10
```

### Architecture shapes

UnifiedML automatically detects the shape signature of your model.

```diff
# Run.py

from torch import nn

class Model(nn.Module): 
+   def __init__(self, in_features, out_features):
        super().__init__()
        
        self.model = nn.Sequential(nn.Linear(in_features, 128), nn.Linear(128, out_features))

    def forward(self, x):
        return self.model(x)
```

**Run:**

```console
ML Model=Run.Model Dataset=CIFAR10
```

Inferrable signature arguments include ```in_shape```, ```out_shape```, ```in_features```, ```out_features```, ```in_channels```, ```out_channels```, ```in_dim```, ```out_dim```.

Just include them as args to your model and UnifiedML will detect and fill them in.

Thus, you can pass classes to command-line, not just objects. Later, we'll see [how to instantiate them with custom arguments](#syntax).

### Acceleration

* Hard disk memory mapping.
* Adaptive RAM, CUDA, and pinned-memory allocation & caching, with [customizable storage distributions]().
* Shared-RAM parallelism.
* Automatic 16-bit mixed precision with ```mixed_precision=true```.
* Multi-GPU automatic detection and parallel training with ```parallel=true```.

Works across domains, including reinforcement learning and generative modeling.

# Syntax

1. The ```hyperparam.``` syntax is used to modify arguments of flag ```Hyperparam```. We reserve ```Uppercase=Path.To.Class``` for the class itself and ```lowercase.key=value``` for argument tinkering, as in ```env.game=pong``` or ```model.depth=5``` (shown in [Methods 1, 2, and 4 below](#heres-how-to-write-the-same-program-in-5-different-ways)).
2. Executable code such as lists, tuples, dictionaries, and functions should be passed in quotes e.g. ```model.dims='[128, 64, 32]'```.
3. Note: we often use the "task" and "recipe" terms interchangeably. Both refer to the ```task=``` flag.

## Here's how to write the same program in 6 different ways.

Train a simple 5-layer CNN to play Atari Pong:

<img src="https://camo.githubusercontent.com/38d38c836102c4487b79af81f79005a26a990119464ce337b5230bc34695ccc0/687474703a2f2f6d7573796f6b752e6769746875622e696f2f696d616765732f706f73742f323031362d30332d30362f706f6e675f726573756c742e676966" data-canonical-src="https://camo.githubusercontent.com/38d38c836102c4487b79af81f79005a26a990119464ce337b5230bc34695ccc0/687474703a2f2f6d7573796f6b752e6769746875622e696f2f696d616765732f706f73742f323031362d30332d30362f706f6e675f726573756c742e676966" width="64" height="84" alt=""/>

<details>
<summary>
Method 1. Purely command-line
</summary>
<br>

```console
ML task=RL Env=Atari env.game=pong Model=CNN model.depth=5
```

</details>

<details>
<summary>
Method 2. Command line
</summary>
<br>

```python
# Run.py

from ML import main

if __name__ == '__main__':
    main()
```

**Run:**

```console
python Run.py task=RL Env=Atari env.game=pong Model=CNN model.depth=5
```

</details>

<details>
<summary>
Method 3. Inferred Code
</summary>
<br>

```python
# Run.py

from ML import main

if __name__ == '__main__':
    main('env.game=pong', 'model.depth=5', task='RL', Env='Atari', Model='CNN')
```

**Run:**

```console
python Run.py
```

</details>

<details>
<summary>
Method 4. Purely Code
</summary>
<br>

```python
# Run.py

from ML import main
from ML.Blocks.Architectures import CNN
from ML.World.Environments import Atari

if __name__ == '__main__':
    main(task='RL', Env=Atari(game='pong'), Model=CNN(depth=5))
```

**Run:**

```console
python Run.py
```

</details>

<details>
<summary>
Method 5. Recipes
</summary>
<br>

Define recipes in a ```.yaml``` file like this one:

```yaml
# recipe.yaml

imports:
  - RL
  - self
Env: Atari
env:
  game: pong
Model: CNN
model:
  depth: 5
```

**Run:**

```console
ML task=recipe
```

The ```imports:``` syntax allows importing multiple tasks/recipes from different sources, with the last item in the list having the highest priority when arguments conflict.

</details>

<details>
<summary>
Method 6. All of the above
</summary>
<br>

The order of hyperparam priority is ```command-line > code > recipe```.

Here's a combined example:

```yaml
# recipe.yaml

imports:
  - RL
  - self
Model: CNN
model:
  depth: 5
```

```python
# Run.py

from ML import main
from ML.World.Environments.Atari import Atari

if __name__ == '__main__':
    main(Env=Atari(game='pong'))
```

**Run:**

```console
python Run.py task=recipe 
```

</details>

Find more details about the grammar and syntax possibilities at [minihydra / leviathan](github.com/AGI-init/minihydra).

# Tutorials

## Custom datasets

Paths or instances to Pytorch Datasets can be fed to the ```Dataset=``` flag.

Here's ImageNet using the built-in torchvision Dataset with a custom transform:

```console
ML Dataset=torchvision.datasets.ImageNet dataset.root='./' dataset.transform=transforms.Resize(64)
```

Generally, a custom Dataset class may look like this:

```python
# Run.py

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, train=True):
        self.classes = ['dog', 'cat']
        ...
    
    def __getitem__(self, index):
        ...
    
    def __len__(self):
        ...
```

**Run:**

```console
ML Dataset=Run.MyDataset
```

If you define your own classify Dataset, include a ```.classes``` attribute listing the classes in your dataset. Otherwise, UnifiedML will automatically count unique classes, which may be different across training and test sets.

You can include a ```train=``` boolean arg to your custom Dataset to use it for both training and testing s.t. it is toggled between the two or pass in a different custom test Dataset via ```TestDataset=``` and the same syntax.

## How to write custom loss functions, backwards, optim, etc.

Let's look at the ```Model``` [from earlier](#architecture-shapes):

```python
# Run.py

from torch.nn.functional import cross_entropy

class Agent(Model):
    def learn(self, replay, logger):  # Add a learn(·) method to the Model from before
        batch = next(replay)
        
        y = self(batch.obs)
        
        loss = cross_entropy(y, batch.label)
        logger.log(loss=loss)
        
        return loss
```

**Run:**

```console
ML Agent=Run.Agent Dataset=CIFAR10
```

We've now added a custom ```learn(·)``` method to our original ```Model``` that does basic cross-entropy and passed it into ```Agent=Run.Agent```, overriding the default Agent. 

For more sophisticated optimization schemes, we may optimize directly within the ```learn(·)``` method (e.g. ```loss.backward(); optim.step()```) and not return a loss. 

[```replay```](World/Replay.py) allows us to sample batches. [```logger```](Logger.py) allows us to keep track of metrics. 

[We provide many Agent examples across domains, including RL and generative modeling.](Agents)

## Custom Environments

## Saving & Loading

## Multi-Task

# Examples

---

To be continued ...

---

#

[MIT license Included.](MIT_LICENSE)