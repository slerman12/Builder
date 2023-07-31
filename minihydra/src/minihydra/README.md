[//]: # (<img width="15%" alt="logo" src="https://github.com/AGI-init/Assets/assets/92597756/5a32b2d4-1ad4-4386-8ea1-b3f991e60979">)

![Medieval-rounded](https://github.com/AGI-init/Assets/assets/92597756/e55fc36b-2d94-431e-82ec-2fcdcbd01bbf)

### Reading in args

```python
# run_script.py

from minihydra import just_args

args = just_args()

print(args, '\n')

print(args.hello)
print(args.number)
print(args.goodbye.cruel)
```

```
> python run_script.py hello=world number=42 goodbye.cruel='[world]'

{'hello': 'world', 'number', 42, {'goodbye': {'cruel': ['world']}}}

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
# run_script.py

from minihydra import just_args

args = just_args(source='path/to/args.yaml')

print(args, '\n')

print(args.hello)
print(args.number)
print(args.goodbye.cruel)
```

```
> python run_script.py number=43

{'hello': 'world', 'number', 43, {'goodbye': {'cruel': ['world']}}}

world
43
[world]
```

### As a decorator

```python
# run_script.py

from minihydra import get_args

@get_args(source='path/to/args.yaml')
def main(args):
    print(args)
```

```
> python run_script.py

{'hello': 'world', 'number', 42, {'goodbye': {'cruel': ['world']}}}
```

### Literals: 
lists, dicts, floats, ints, booleans, null, inf, strings; put lists and dicts in quotes

### imports
Or via reserved task= keyword argument

### instantiate
Path/To.py -> Cow "Moo"

### interpolation

### grammars

### yaml search paths
    
MIT
