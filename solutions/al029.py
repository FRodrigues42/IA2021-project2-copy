# -*- coding: utf-8 -*-
"""
Grupo al029
Fernando Rodrigues #66326
Otavio Serra #98444
"""

from __future__ import print_function
import numpy as np

from collections import Counter
import math

def plurality_value(examples):
    classification_counts = Counter(examples.values()).most_common()

    if len(classification_counts) == 1:
        return classification_counts[0][0]

    i = 1
    most_common_classifications = [classification_counts[0][0]]
    while i < len(classification_counts) and classification_counts[i-1][1] == classification_counts[i][0]:
        most_common_classifications.append(classification_counts[i][0])
        i += 1

    rng = np.random.default_rng()
    return rng.choice(most_common_classifications).item()


def b(q: float) -> float:
    if q <= 0 or q >= 1:
        return 1

    return -(q*math.log2(q) + (1-q)*math.log2(1-q))


def remainder(arg, attributes, examples):
    d = len(attributes[arg])

    sum = 0
    for k in range(d):
        filtered_ex = filter_examples(examples, arg, attributes[arg][k])
        classification_counts = dict(Counter(filtered_ex.values()))
        if len(classification_counts) == 0:
            continue
        sum += len(classification_counts)/len(examples) + \
              b(classification_counts.get(1, 0)/len(classification_counts))

    return sum


def importance(arg, attributes, examples):
    classification_counts = dict(Counter(examples.values()))

    return b(classification_counts.get(1, 0)/len(examples.values())) - remainder(arg, attributes, examples)


def argmax_importance(attributes, examples):
    max = 0
    amax = list(attributes.keys())[0]
    for a in attributes.keys():
        imp = importance(a, attributes, examples)
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
    elif len(set(examples.values())) == 1 and parent_examples is None:
        return [list(attributes.keys())[0], list(examples.values())[0], list(examples.values())[0]]
    elif len(set(examples.values())) == 1:
        return list(examples.values())[0]
    elif len(attributes) == 0:
        return plurality_value(examples)
    else:
        a = argmax_importance(attributes, examples)
        tree = [a]
        for v in attributes[a]:
            exs = filter_examples(examples, a, v)
            subtree = decision_tree_learning(exs, filter_attributes(attributes, a), examples)
            tree.append(subtree)
        return tree


def prune_binary_tree(t):
    if type(t[1]) is not list or type(t[2]) is not list:
        if type(t[1]) is list:
            return [t[0], prune_binary_tree(t[1]), t[2]]
        elif type(t[2]) is list:
            return [t[0], t[1], prune_binary_tree(t[2])]

        return t

    if t[1] == t[2]:
        return prune_binary_tree(t[1])

    return [t[0], prune_binary_tree(t[1]), prune_binary_tree(t[2])]


def createdecisiontree(D,Y, noise = False):
    examples = {tuple(D[i]):int(Y[i]) for i in range(len(Y))}
    attributes = {i:tuple(set([int(v) for v in D.T[i]])) for i in range(len(D[0]))}

    return prune_binary_tree(decision_tree_learning(examples, attributes))
