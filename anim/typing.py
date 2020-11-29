from typing import Any, cast, Callable, Dict, Iterable, Iterator, List, NamedTuple, Set, Tuple, Union, Optional
from numbers import Number

__all__ = ['Any', 'cast', 'Callable', 'Dict', 'Iterable', 'Iterator', 'List', 'NamedTuple', 'Set', 'Tuple', 'Union', 'Optional']
__all__ += ['Number']
__all__ += ['unused']


def unused(*_ignored):
    """Dummy function to declare a parameter knowingly unused"""
    pass
