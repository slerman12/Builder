<picture>
  <source width="20%" media="(prefers-color-scheme: dark)" srcset="https://github.com/AGI-init/Assets/assets/92597756/e411328b-a51a-416c-ba97-7b7939ec3351">
  <img width="20%" alt="Text changing depending on mode. Light: 'Light' Dark: 'Dark'" src="https://github.com/AGI-init/Assets/assets/92597756/f3df44c8-b989-4951-9443-d2b4203b5c4e">
<br><br>
</picture>

# Welcome

See our library [Tributaries](../../../tributaries-ml/src/tributaries) for mass-deploying UnifiedML apps on remote servers.

Check out [minihydra / leviathan]() for how we handle sys args & hyperparams.

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

# Quick start

## Training example

```python
# Run.py

from torch import nn

model = nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))
```

**Run:**

```console
ML Model=Run.model Dataset=CIFAR10
```

In this example, we use the Dataset ```CIFAR10```. There are many [built-in](#built-ins) datasets, architectures, and so on, such as ```Dataset=CIFAR10```. The default domain is classification and can be changed with the ```task=``` flag.

The above demonstrates ***dot notation***. Equivalently, it's possible to use ***regular directory paths***:
```console
ML Model=./Run.py.model Dataset=CIFAR10
```

Wherever you run ```ML```, it'll search from the current directory for any specified paths.

## Apps

It's possible to do this entirely from code without using ```ML```, as per below:

```python
# Run.py

# Equivalent pure-code training example

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

We call this a UnifiedML ***app***.

## If you're feeling brave

Not exactly scalable, but:

```console
ML Model='nn.Sequential(nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 10))' Dataset=CIFAR10
```

Direct code execution also works.

## Architecture shapes

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

Thus, you can pass classes to command-line, not just objects.

## Syntax

- **Argument tinkering** The ```hyperparam.``` syntax is used to modify arguments of flag ```Hyperparam```. We reserve ```Uppercase=Path.To.Class``` for the class itself and ```lowercase.key=value``` for argument tinkering, as in ```env.game=pong``` or ```model.depth=5``` (shown in [ways 1, 2, and 4 below](#way-1-purely-command-line)).
- **Executable arguments** Executable code such as lists, tuples, dictionaries, and functions should be passed in quotes e.g. ```model.dims='[128, 64, 32]'```.
- **Saving arguments as recipes** Note: we often use the "task" and "recipe" terms interchangeably. Both refer to the ```task=``` flag. [Ways 6 and 7 below](#way-6-recipes) show how to define a task/recipe.

<details>
<summary>
<h3>
Here's how to write the same program in 7 different ways. (Click to expand)
</h3>
</summary>

Train a simple 5-layer CNN to play Atari Pong:

<img src="https://camo.githubusercontent.com/38d38c836102c4487b79af81f79005a26a990119464ce337b5230bc34695ccc0/687474703a2f2f6d7573796f6b752e6769746875622e696f2f696d616765732f706f73742f323031362d30332d30362f706f6e675f726573756c742e676966" data-canonical-src="https://camo.githubusercontent.com/38d38c836102c4487b79af81f79005a26a990119464ce337b5230bc34695ccc0/687474703a2f2f6d7573796f6b752e6769746875622e696f2f696d616765732f706f73742f323031362d30332d30362f706f6e675f726573756c742e676966" width="64" height="84" alt=""/>

### Way 1. Purely command-line

```console
ML task=RL Env=Atari env.game=pong Model=CNN model.depth=5
```

### Way 2. Command-line code

```console
ML task=RL Env='Atari(game="pong")' Model='CNN(depth=5)'
```

### Way 3. Command-line

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

### Way 4. Inferred Code

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

### Way 5. Purely Code

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

### Way 6. Recipes

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

### Way 7. All of the above

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
    main(Env=Atari)
```

**Run:**

```console
python Run.py task=recipe env.game=pong
```

</details>

Find more details about the grammar and syntax possibilities at [minihydra / leviathan](github.com/AGI-init/minihydra).

## Acceleration

With ```accelerate=true```:
* Hard disk memory mapping.
* Adaptive RAM, CUDA, and pinned-memory allocation & caching, with [customizable storage distributions]().
* Shared-RAM parallelism.
* Automatic 16-bit mixed precision with ```mixed_precision=true```.
* Multi-GPU automatic detection and parallel training with ```parallel=true```.

Fully supported across domains, including reinforcement learning and generative modeling.

# Tutorials

<details>
<summary>
<h2>
&nbsp;&nbsp;&nbsp;Custom datasets
</h2>
</summary>

Paths or instances to Pytorch Datasets can be fed to the ```Dataset=``` flag.

Here's ImageNet using the built-in torchvision Dataset with a custom transform:

```console
ML Dataset=ImageNet dataset.root='imagenet/' dataset.transform='transforms.Resize(64)'
```

---

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
        
        return obs, label
    
    def __len__(self):
        ...
```

**Run:**

```console
ML Dataset=Run.MyDataset
```

**Classification**

Since the default task is ```task=classify```, the above script will learn to classify ```MyDataset```.

If you define your own classify Dataset, include a ```.classes``` attribute listing the classes in your Dataset. Otherwise, UnifiedML will automatically count unique classes, which may be different across training and test sets. If not, don't worry about this.

**Test datasets**

You can include a ```train=``` boolean arg to your custom Dataset to define different behaviors for training and testing, or use a different custom test Dataset via ```TestDataset=```.

**Transforms & augmentations**

All passed-in Datasets will support the ```dataset.transform=``` argument. ```dataset.transform=``` is distinct from ```transform=``` and ```Aug=```, as ```transform=``` runs a transform on CPU at runtime and ```Aug=``` runs a batch-vectorized augmentation on GPU at runtime, whereas ```dataset.transform=``` transforms/pre-compiles the transformed dataset before training begins. One-time operations like Resize are most efficient here.  

**Standardization & normalization**

Stats will automatically be computed for standardization and normalization, and saved in the corresponding Memory ```card.yaml``` in ```World/ReplayBuffer```. Disable standardization with ```standardize=false```. This will trigger to use normalization instead. Disable both with ```standardize=false norm=false```. You may learn more about the differences at [GeeksforGeeks](https://www.geeksforgeeks.org/normalization-vs-standardization/). By default, an agent loaded from a checkpoint will reuse its original tabulated stats of the data that it was trained on even when evaluated or further trained on a new dataset, to keep conditions consistent.

**Subsets**

Sub-classing is possible with the ```dataset.subset='[0, 5, 2]'``` keyword. In this example, only classes ```0```, ```5```, and ```2``` of the given Dataset will be used for training and evaluation.

</details>

<details>
<summary>
<h2>
&nbsp;&nbsp;&nbsp;How to write custom loss functions, backwards, optim, etc.
</h2>
</summary>

Let's look at the ```Model``` [from earlier](#architecture-shapes):

```python
# Run.py

from torch.nn.functional import cross_entropy

class Model_(Model):
    def learn(self, replay, logger):  # Add a learn(·) method to the Model from before
        batch = next(replay)
        
        y = self(batch.obs)
        
        loss = cross_entropy(y, batch.label)
        logger.log(loss=loss)
        
        return loss
```

**Run:**

```console
ML Model=Run.Model_ Dataset=CIFAR10
```

We've now added a custom ```learn(·)``` method to our original ```Model``` that does basic cross-entropy.

For more sophisticated optimization schemes, we may optimize directly within the ```learn(·)``` method (e.g. ```loss.backward(); optim.step()```) and not return a loss.

[```replay```](World/Replay.py) allows us to sample batches. [```logger```](Logger.py) allows us to keep track of metrics.

#
- By the way, there's no difference between ```Model=``` and ```Agent=```. The two are interchangeable. However, ```Model=``` in this example demonstrates a simplified version of the full capacity of Agents, which includes multi-task learning and *generalism*.
- [We provide many Agent/Model examples across domains, including RL and generative modeling.](Agents)

#
Use ```Optim=``` or ```Scheduler=``` to define a custom optimizer or scheduler:

```console
ML Model=Run.Model_ Dataset=CIFAR10 Optim=Adam optim.lr=1e2 Scheduler=CosineAnnealingLR scheduler.T_max=1000
```

or one of the existing shorthands for the above-equivalent:

```console
ML Model=Run.Model_ Dataset=CIFAR10 lr=1e2 lr_decay_epochs=1000
```

</details>

<details>
<summary>
<h2>
&nbsp;&nbsp;&nbsp;Custom Environments
</h2>
</summary>
</details>

[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Plotting, Logging, Stats, & Media )

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

<details>
<summary>
<h2>
&nbsp;&nbsp;&nbsp;Saving & Loading
</h2>
</summary>
</details>

<details>
<summary>
<h2>
&nbsp;&nbsp;&nbsp;Multi-Task
</h2>
</summary>
</details>

[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Multi-Modal)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Cheatsheet of built-in learning modes & features)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Cheatsheet: Built-in features of default Agent.learn)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # (# Examples)

[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;CIFAR10 in 10 seconds)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;ImageNet on 1 GPU)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Imagen: Text to image)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Stable Diffusion)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Humanoid from pixels)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;BittleBot: Real-time robotics with RL)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Image Segmentation)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Atari)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # ()
[//]: # (<details>)

[//]: # (<summary>)

[//]: # (<h2>)

[//]: # (&nbsp;&nbsp;&nbsp;Text prediction)

[//]: # (</h2>)

[//]: # (</summary>)

[//]: # (</details>)

[//]: # ()
[//]: # (# Apps built with UnifiedML)

[//]: # ()
[//]: # (- [XRDs modeling project]&#40;https://www.github.com/AGI-init/XRDs&#41;)

[//]: # (Step 1. Define a [Generator]&#40;&#41; and [Discriminator]&#40;&#41;.)

[//]: # (Step 2. ...)

[//]: # (Step N.These are all the parts that are pointed to in the [```dcgan recipe```]&#40;&#41;.)

[//]: # ()
[//]: # (**Run:**)

[//]: # ()
[//]: # (```console)

[//]: # (ML task=dcgan)

[//]: # (```)

[//]: # (# What is novel about UnifiedML?)

[//]: # ()
[//]: # (- Adaptive accelerations)

[//]: # (- Multi-block framework)

[//]: # (- Universal generalism)

To be continued ...

---

#

By [Sam Lerman](https://www.github.com/slerman12).

[MIT license included.](MIT_LICENSE)