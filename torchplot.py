## This code is released into the public domain. Use as you please!
## Author: Søren Hauberg

import matplotlib.pyplot as plt
import numpy as np
import torch
from inspect import getmembers, isfunction, getdoc

# Function to convert a list of arguments containing torch tensors, into
# a corresponding list of arguments containing numpy arrays
def __torch2np__(*args, **kwargs):
    def convert(arg):
        return arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg

    # first unnamed arguments
    outargs = [convert(arg) for arg in args]

    # then keyword arguments
    outkwargs = dict()
    for key, value in kwargs.items():
        outkwargs[key] = convert(value)

    return outargs, kwargs

# Iterate over all members of 'plt' in order to duplicate them
for name, member in getmembers(plt):
    if isfunction(member):
        doc = getdoc(member)
        strdoc = "" if doc is None else doc
        exec(('def {name}(*args, **kwargs):\n' +
              '\t"""{doc}"""\n' +
              '\tnew_args, new_kwargs = __torch2np__(*args, **kwargs)\n' +
              '\treturn plt.{name}(*new_args, **new_kwargs)').format(name=name, doc=strdoc))
    else:
        exec('{name} = plt.{name}'.format(name=name))
    #break
