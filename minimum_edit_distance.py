import numpy as np


def get_levenshtein_delete_cost(x):
    return len(x)


def get_levenshtein_insert_cost(x):
    return len(x)


def get_levenshtein_substitute_cost(x, y):
    if not len(x) == len(y) == 1:
        raise Exception('substitution is supported for strings with length 1 only')

    if x == y:
        return 0

    return 2


def np_minimum_edit_distance(source, target):
    """
    Function for computing minimum edit distance between to strings. The costs
    are calculated based on Levenshtein distance algorithm (delete:1, insert: 1,
    substitute: 2).
    Note that uppercase and lowercase characters are not equal, for example 'a'
    and 'A' are not equal.

    Args:
        source (str): source string.
        target (str): target string.

    Returns:
        tuple: (distance_matrix, backtrace_matrix)
            distance_matrix: m * n numpy.array of minimum edit distance for two
                strings source and target with lengths of m and n respectively.
            path_matrix: m * n numpy.chararray for two strings source and target
                with lengths of m and n respectively.
    Example:
        >>> source = 'minimum'
        >>> target = 'minimom'
        >>> distance, backtrace = np_minimum_edit_distance(source, target)
        >>> print(distance)
        [[0. 1. 2. 3. 4. 5. 6. 7.]
         [1. 0. 1. 2. 3. 4. 5. 6.]
         [2. 1. 0. 1. 2. 3. 4. 5.]
         [3. 2. 1. 0. 1. 2. 3. 4.]
         [4. 3. 2. 1. 0. 1. 2. 3.]
         [5. 4. 3. 2. 1. 0. 1. 2.]
         [6. 5. 4. 3. 2. 1. 2. 3.]
         [7. 6. 5. 4. 3. 2. 3. 2.]]
        >>> print(backtrace)
        [['' '' '' '' '' '' '' '']
         ['' 'diag' 'left' 'left' 'left' 'diag' 'left' 'diag']
         ['' 'up' 'diag' 'left' 'diag' 'left' 'left' 'left']
         ['' 'up' 'up' 'diag' 'left' 'left' 'left' 'left']
         ['' 'up' 'diag' 'up' 'diag' 'left' 'left' 'left']
         ['' 'diag' 'up' 'up' 'up' 'diag' 'left' 'diag']
         ['' 'up' 'up' 'up' 'up' 'up' 'diag' 'diag']
         ['' 'diag' 'up' 'up' 'up' 'diag' 'diag' 'diag']]
    """

    n = len(source)
    m = len(target)

    distance_mat = np.zeros((n + 1, m + 1))
    path_mat = np.chararray((n + 1, m + 1), itemsize=4, unicode=True)

    distance_mat[0, 0] = 0
    for i in range(1, n + 1):
        distance_mat[i, 0] = distance_mat[i - 1, 0] + get_levenshtein_delete_cost(source[i - 1])

    for j in range(1, m + 1):
        distance_mat[0, j] = distance_mat[0, j - 1] + get_levenshtein_insert_cost(target[j - 1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            possible_choices = [
                {
                    'state': 'diag',
                    'cost': (distance_mat[i - 1, j - 1] +
                             get_levenshtein_substitute_cost(source[i - 1], target[j - 1])),
                },
                {
                    'state': 'up',
                    'cost': distance_mat[i - 1, j] + get_levenshtein_delete_cost(source[i - 1]),
                },
                {
                    'state': 'left',
                    'cost': distance_mat[i, j - 1] + get_levenshtein_insert_cost(target[j - 1]),
                }
            ]

            min_cost = min(possible_choices, key=lambda x: x['cost'])
            distance_mat[i, j] = min_cost['cost']
            path_mat[i, j] = min_cost['state']

    return distance_mat, path_mat