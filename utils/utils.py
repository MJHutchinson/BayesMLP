import os
import itertools

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir)
        if os.path.isfile(os.path.join(a_dir, name))]


def num_to_name(number):
    names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    return names[number]


def full_search_space(hs, hidden_layers):
    search_space = list(itertools.permutations(hs, hidden_layers))
    search_space.extend([tuple([h]) * hidden_layers for h in hs])
    return sorted(search_space, key=lambda t: t[0], reverse=True)


def small_search_space(hs, hidden_layers):
    search_space = [tuple([h]) * hidden_layers for h in hs]
    return sorted(search_space, key=lambda t: t[0], reverse=True)


def gen_hidden_combinations(search_space, hs, hidden_layers):
    if search_space == 'full':
        return full_search_space(hs, hidden_layers)
    elif search_space == 'small':
        return small_search_space(hs, hidden_layers)
    else:
        return []


def parameter_combinations(hs, lrs, prior_vars):
    output = []
    for h in hs:
        for lr in lrs:
            for prior_var in prior_vars:
                output.append((h, lr, prior_var))

    return output


