from collections.abc import Iterable

from quri_parts.qsub.codegen import CodeGenerator
from quri_parts.qsub.link import link
from quri_parts.qsub.machineinst import MachineSub
from quri_parts.qsub.op import AbstractOp, Op
from quri_parts.qsub.resolve import SubCollector, SubRepository, default_repository
from quri_parts.qsub.sub import Sub
from quri_parts.qsub.trans import SequentialTranspiler, SubTranspiler


def compile(
    entry_op: Op,
    primitives: Iterable[AbstractOp],
    repository: SubRepository = default_repository(),
    sub_transpilers: Iterable[SubTranspiler] = (),
) -> MachineSub:
    collector = SubCollector(repository)
    subs = collector.collect_subs(entry_op)
    if sub_transpilers:
        trans = SequentialTranspiler(sub_transpilers)
        subs = {k: trans(v) for k, v in subs.items()}
    codegen = CodeGenerator(primitives)
    msubs = {op: codegen.lower(sub) for op, sub in subs.items()}
    entry_msub = msubs[entry_op]
    return link(entry_msub, msubs)


def compile_sub(
    entry_sub: Sub,
    primitives: Iterable[AbstractOp],
    repository: SubRepository = default_repository(),
    sub_transpilers: Iterable[SubTranspiler] = (),
) -> MachineSub:
    collector = SubCollector(repository)
    subs = collector.collect_subs(entry_sub)
    if sub_transpilers:
        trans = SequentialTranspiler(sub_transpilers)
        entry_sub = trans(entry_sub)
        subs = {k: trans(v) for k, v in subs.items()}
    codegen = CodeGenerator(primitives)
    msubs = {op: codegen.lower(sub) for op, sub in subs.items()}
    entry_msub = codegen.lower(entry_sub)
    return link(entry_msub, msubs)
