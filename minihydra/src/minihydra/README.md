<img width="50%" alt="logo" src="https://github.com/AGI-init/Assets/assets/92597756/5a32b2d4-1ad4-4386-8ea1-b3f991e60979">

```console
pip install minihydra
```

---

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
```

```
> python run.py

{'hello': 'world', 'number': 42, {'goodbye': {'cruel': ['world']}}}
```

### Advanced

**Further features include literals, function calls, instantiation, imports, interpolation, custom grammars, and expanding module and yaml search paths.**

For deeper documentation and allowing this work to continue to be open source, please consider [donating](https://github.com/sponsors/AGI-init).

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

---

<img width="60%" alt="logo" src="https://github.com/AGI-init/Assets/assets/92597756/e55fc36b-2d94-431e-82ec-2fcdcbd01bbf">

[Licensed under the MIT license.](MIT_LICENSE)