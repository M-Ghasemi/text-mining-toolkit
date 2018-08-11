import numpy as np
import pandas as pd


def get_levenshtein_delete_cost(x: str) -> int:
    """
    The deletion cost of each character is 1.
    """
    return len(x)


def get_levenshtein_insert_cost(x: str) -> int:
    """
    The insertion cost of each character is 1.
    """
    return len(x)


def get_levenshtein_substitute_cost(x: str, y: str) -> int:
    """
    The substitution cost of character `x` with  character y is 2.

    Raises:
        ValueError: If `x` or `y` are not of length of 1.
    """
    if not len(x) == len(y) == 1:
        raise ValueError('substitution is supported for strings with length 1 only')

    if x == y:
        return 0

    return 2


def np_levenshtein_minimum_edit_distance(source, target):
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
            backtrace_matrix: m * n numpy.chararray for two strings source and
                target with lengths of m and n respectively.
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
    backtrace_mat = np.chararray((n + 1, m + 1), itemsize=4, unicode=True)

    distance_mat[0, 0] = 0
    for i in range(1, n + 1):
        distance_mat[i, 0] = (distance_mat[i - 1, 0] +
                              get_levenshtein_delete_cost(source[i - 1]))

    for j in range(1, m + 1):
        distance_mat[0, j] = (distance_mat[0, j - 1] +
                              get_levenshtein_insert_cost(target[j - 1]))

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            possible_choices = [
                {
                    'state': 'diag',
                    'cost': (distance_mat[i - 1, j - 1] +
                             get_levenshtein_substitute_cost(source[i - 1],
                                                             target[j - 1])),
                },
                {
                    'state': 'up',
                    'cost': (distance_mat[i - 1, j] +
                             get_levenshtein_delete_cost(source[i - 1])),
                },
                {
                    'state': 'left',
                    'cost': (distance_mat[i, j - 1] +
                             get_levenshtein_insert_cost(target[j - 1])),
                }
            ]

            min_cost = min(possible_choices, key=lambda x: x['cost'])
            distance_mat[i, j] = min_cost['cost']
            backtrace_mat[i, j] = min_cost['state']

    return distance_mat, backtrace_mat


def pd_levenshtein_minimum_edit_distance(source: str, target: str) -> 'pandas.DataFrame':
    """
    Function for computing minimum edit distance between to strings. The costs
    are calculated based on Levenshtein distance algorithm (delete:1, insert: 1,
    substitute: 2).
    Note that uppercase and lowercase characters are not equal, for example 'a'
    and 'A' are not equal.

    Returns:
        pandas.DataFrame: distance_matrix
            distance_matrix: m * n pandas.DataFrame of minimum edit distance
                for two strings source and target with lengths of m and n respectively.
                Each cell of DataFram contains a tuple of the cost number and
                also a string for backtrace.
    Example:
        >>> source = 'minimum'
        >>> target = 'minimom'
        >>> distance = pd_levenshtein_minimum_edit_distance(source, target)
        >>> print(distance)
                   -1          0          1          2          3          4          5          6
        -1  (0, stop)  (1, left)  (2, left)  (3, left)  (4, left)  (5, left)  (6, left)  (7, left)
         0    (1, up)  (0, diag)  (1, left)  (2, left)  (3, left)  (4, diag)  (5, left)  (6, diag)
         1    (2, up)    (1, up)  (0, diag)  (1, left)  (2, diag)  (3, left)  (4, left)  (5, left)
         2    (3, up)    (2, up)    (1, up)  (0, diag)  (1, left)  (2, left)  (3, left)  (4, left)
         3    (4, up)    (3, up)  (2, diag)    (1, up)  (0, diag)  (1, left)  (2, left)  (3, left)
         4    (5, up)  (4, diag)    (3, up)    (2, up)    (1, up)  (0, diag)  (1, left)  (2, diag)
         5    (6, up)    (5, up)    (4, up)    (3, up)    (2, up)    (1, up)  (2, diag)  (3, diag)
         6    (7, up)  (6, diag)    (5, up)    (4, up)    (3, up)  (2, diag)  (3, diag)  (2, diag)
    """

    m = len(source)
    n = len(target)

    distance = pd.DataFrame(index=range(-1, m),
                            columns=range(-1, n))

    distance[-1][-1] = 0, 'stop'

    for i in range(m):
        distance[-1][i] = (
            distance[-1][i - 1][0] + get_levenshtein_delete_cost(source[i]), 'up')

    for j in range(n):
        distance[j][-1] = (
            distance[j - 1][-1][0] + get_levenshtein_insert_cost(target[j]), 'left')

    for i in range(m):
        for j in range(n):
            possible_choices = [
                {
                    'state': 'diag',
                    'cost': (distance[j - 1][i - 1][0] +
                             get_levenshtein_substitute_cost(source[i], target[j])),
                },
                {
                    'state': 'left',
                    'cost': (distance[j - 1][i][0] +
                             get_levenshtein_insert_cost(target[j]))
                },
                {
                    'state': 'up',
                    'cost': (distance[j][i - 1][0] +
                             get_levenshtein_delete_cost(source[i]))
                }
            ]

            min_cost = min(possible_choices, key=lambda x: x['cost'])
            distance[j][i] = min_cost['cost'], min_cost['state']

    return distance


def print_minimum_distance(source, target):
    """
    Function for printing minimum edit distance between to strings. The costs
    are calculated based on Levenshtein distance algorithm (delete:1, insert: 1,
    substitute: 2).
    Note that uppercase and lowercase characters are not equal, for example 'a'
    and 'A' are not equal.

    Example:
        >>> source = 'playing'
        >>> target = 'pray'
        >>> print_minimum_distance(source, target)
        Minimum distance: 5
        ['p', 'l', 'a', 'y', 'i', 'n', 'g']
        ['p', 'r', 'a', 'y', '*', '*', '*']
    """
    m = len(source)
    n = len(target)

    src = []
    dst = []
    idx = m - 1
    col = n - 1
    distance = pd_levenshtein_minimum_edit_distance(source, target)

    while not idx == col == -1:
        direction = distance[col][idx][1]
        if direction == 'diag':
            src.append(source[idx])
            dst.append(target[col])

            idx -= 1
            col -= 1
        elif direction == 'left':
            src.append('*')
            dst.append(target[col])

            col -= 1
        else:
            src.append(source[idx])
            dst.append('*')
            idx -= 1
    src.reverse()
    dst.reverse()
    print(f'Minimum distance: {distance[n - 1][m - 1][0]}')
    print(src)
    print(dst)
