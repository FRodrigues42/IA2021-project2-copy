# -*- coding: utf-8 -*-
"""
Grupo al029
Student id #77777
Student id #77777
"""

from __future__ import print_function
import numpy as np

def plurality_value(examples):
    np.random.choice(np.flatnonzero(examples == examples.max()))

def importance(arg, examples):
    pass

def argmax_importance(attributes, examples):
    max = 0
    amax = None
    for a in attributes.keys():
        imp = importance(a, examples)
        if imp > max:
            max = imp
            amax = a

    return amax

def filter_examples(examples, argument, value):
    filtered_ex = {}
    for key in examples.keys():
        if key[argument] == value:
            filtered_ex[key] = examples[key]

    return filtered_ex

def filter_attributes(attributes, a_to_remove):
    filtered_attributes = {}
    for a in attributes.keys():
        if a != a_to_remove:
            filtered_attributes[a] = attributes[a]

    return filtered_attributes

def decision_tree_learning(examples, attributes, parent_examples=None):

# examples: dicionario -> { (valor do atributo 0, valor do atributo 1, ...): classificacao, ... }
# attributes: dicionario -> { posicao do atributo 0: (valor 0 do atributo 0, valor 1 do atributo 0, ...), ...}
# parent_examples: não existe se for o inicio da construcao da decision tree

    if len(examples) == 0:
        return plurality_value(parent_examples)
    elif len(set(examples.values())) == 1:
        return list(d.values())[0] # d? aqui imagino que seja examples.values, certo?
    elif len(attributes) == 0:
        return plurality_value(examples)
    else:
        a = argmax_importance(attributes, examples)
        tree = [a]
        for v in attributes[a]:
            exs = filter_examples(examples, a, v)
            subtree = decision_tree_learning(exs, filter_attributes(attributes, a), examples)
            tree.append([a, subtree])
        return tree

def createdecisiontree(D,Y, noise = False):


    return [0,0,1]
