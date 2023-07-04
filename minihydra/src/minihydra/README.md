from minihydra import get_args


@get_args('Hyperparams/args.yaml')
def main(args):
print(args)


main()


"""
"""

"""
@get_args()
def main(args):
print(args)

    The following script:
    
    ```console
    python minimal.py a=5 b=2
    ```
    
    prints: {'a': 5, 'b': 2}
    or better print(args.a) and:
    prints: 5
    
    
    minihydra / leviathan (cute-ish logo) all on one line
    
    
    Minimal example
        in-code, with $
        console Output:
        
    Literals: lists, dicts, floats, ints, booleans, null, inf, strings; put lists and dicts in quotes
    
    .yaml file
    
    imports
        Or via reserved task= keyword argument
    
    instantiate
        Path/To.py -> Cow "Moo"
    
    interpolation
    
    grammars
    
    yaml search paths
    
    MIT
"""
