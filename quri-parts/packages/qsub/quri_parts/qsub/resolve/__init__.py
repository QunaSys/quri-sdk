"""
Sub-routine resolution (:mod:`quri_parts.qsub.resolve`)
=======================================================

Re-exports the repository and resolver types used to map an
:class:`~quri_parts.qsub.op.Op` to its
:class:`~quri_parts.qsub.sub.Sub` implementation.

Contents:

- :class:`SubRepository`: resolver-registry protocol
- :class:`SimpleSubRepository`: mutable, standalone registry of sub-resolvers
- :class:`CompositeSubRepository`: overrides a base repository with an addition
- :class:`SubCollector`: recursively gathers all subs for an Op tree
- :func:`default_repository`: module-global singleton repository
- :func:`resolve_sub`: convenience one-shot resolution
"""

from .resolve import (
    CompositeSubRepository,
    SimpleSubRepository,
    SimpleSubResolver,
    SubRepository,
    SubResolver,
    SubResolverCondition,
    default_repository,
    resolve_sub,
)
from .subcollector import SubCollector

__all__ = [
    "SubResolver",
    "SubRepository",
    "CompositeSubRepository",
    "SimpleSubRepository",
    "default_repository",
    "SubCollector",
    "resolve_sub",
    "SimpleSubResolver",
    "SubResolverCondition",
]
