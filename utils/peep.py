from more_itertools import grouper
from tabulate import tabulate
from inspect import signature
from textwrap import fill

def peep(obj, exclude="_", include=None, num_columns=4):
    if include is None:
        contents = [k for k in dir(obj) if not k.startswith(exclude)]
    else:
        contents = [k for k in dir(obj) if (not k.startswith(exclude)) and (k.startswith(include))]
    
    print(tabulate(grouper(contents, num_columns)))


def get_func_args(func):
    assert callable(func), "Argument is not a function"
    sig = signature(func)
    return list(sig.parameters.keys())


def fprint(x: str, width: int = 70) -> None:
    print(fill(x, width=width))