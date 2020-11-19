from typing import Any, cast, Callable, Dict, Iterable, Iterator, List, NamedTuple, Set, Tuple, Union, Optional
from numbers import Number

TransMat = Tuple[float, float, float, float, float, float]
# print(sorted(''.replace(' ','').split(',')))

__all__ = ['Any', 'cast', 'Callable', 'Dict', 'Iterable', 'Iterator', 'List', 'NamedTuple', 'Set', 'Tuple', 'Union', 'Optional']
__all__ += ['Number']
__all__ += ['TransMat', 'unused']


def unused(*_ignored):
    """Dummy function to declare a parameter knowingly unused"""
    pass
