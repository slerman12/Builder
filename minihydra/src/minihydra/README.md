<img width="50%" alt="logo" src="https://github.com/Cave-Dwellers-Tree-People/minihydra/assets/153214101/8907d6da-e65b-4ca6-a1a9-4f154a4b719b">

```console
pip install leviathan
```

[//]: # (:fleur_de_lis:)

[//]: # ()
[//]: # (I left most of the documentation ambiguous, but to my knowledge, *minihydra / leviathan* supports everything you could need. It's ~400 lines of code if you wish to figure it out. Generally speaking, I ask that you put the equal effort that I did into integrating with **[Hydra]&#40;https://www.github.com/facebookresearch/hydra&#41;** before blatantly copying or ripping off this work. See if **[Hydra]&#40;https://www.github.com/facebookresearch/hydra&#41;** or this suits your needs. If not, and you really make sure, then go ahead. But be careful, hydras and leviathans are beautiful beasts and they are not to mess with.)

[//]: # ()
[//]: # (:fleur_de_lis:)

### Reading in args

```python
# run.py

from minihydra import just_args

args = just_args()

print(args, '\n')
print(args.hello)
print(args.number)
print(args.goodbye.cruel)
```

```
> python run.py hello=world number=42 goodbye.cruel='[world]'

{'hello': 'world', 'number': 42, {'goodbye': {'cruel': ['world']}}}

world
42
[world]
```

### Via yaml

```yaml
# path/to/args.yaml

hello: world
number: 42
goodbye:
  cruel:
    - world
```

```python
# run.py

from minihydra import just_args

args = just_args(source='path/to/args.yaml')

print(args, '\n')
print(args.hello)
print(args.number)
print(args.goodbye.cruel)
```

```
> python run.py number=43

{'hello': 'world', 'number': 43, {'goodbye': {'cruel': ['world']}}}

world
43
[world]
```

### As a decorator

```python
# run.py

from minihydra import get_args

@get_args(source='path/to/args.yaml')
def main(args):
    print(args)

if __name__ == '__main__':
    main()
```

```
> python run.py

{'hello': 'world', 'number': 42, {'goodbye': {'cruel': ['world']}}}
```

### Advanced

**Further features include literals, function calls, instantiation, imports, interpolation, custom grammars, expanding module and yaml search paths, project directory inference, instantiation tinkering-syntax inference, portals, and pseudonyms.**

For deeper documentation, please consider [sponsoring](https://github.com/sponsors/Cave-Dwellers-Tree-People).

[//]: # (:fleur_de_lis:)

[//]: # (### Literals: )

[//]: # (lists, dicts, floats, ints, booleans, null, inf, strings; put lists and dicts in quotes)

[//]: # ()
[//]: # (### imports)

[//]: # (Or via reserved task= keyword argument)

[//]: # ()
[//]: # (### instantiate)

[//]: # (Path/To.py -> Cow "Moo")

[//]: # (signature matching)

[//]: # ()
[//]: # (### interpolation)

[//]: # ()
[//]: # (### grammars)

[//]: # ()
[//]: # (### yaml search paths)

[//]: # (project directory inference, instantiation tinkering inference, portals, pseudonyms)

<img width="60%" alt="logo" src="https://github.com/Cave-Dwellers-Tree-People/minihydra/assets/153214101/0ba65a1c-fa62-41a2-bfe3-77358146f18e">

[Licensed under the MIT license.](MIT_LICENSE)

