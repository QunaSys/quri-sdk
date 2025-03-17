from .op import (
    NonUnitaryDef,
    Op,
    OpDef,
    OpFactory,
    ParamNonUnitaryDef,
    ParamOpDef,
    Params,
    ParamUnitaryDef,
    UnitaryDef,
    op,
    param_op,
)
from .resolve import SubRepository, default_repository
from .sub import ParamSubDef, Sub, SubDef, SubFactory, param_sub, sub


class OpSubDef(OpDef, SubDef):
    ...


class UnitarySubDef(OpSubDef, UnitaryDef):
    ...


class NonUnitarySubDef(OpSubDef, NonUnitaryDef):
    ...


def opsub(
    opsub_def: type[OpSubDef], repository: SubRepository | None = default_repository()
) -> tuple[Op, Sub]:
    _op = op(opsub_def)
    _sub = sub(opsub_def)
    if repository is not None:
        repository.register_sub(_op, _sub)
    return _op, _sub


class ParamOpSubDef(ParamOpDef[Params], ParamSubDef[Params]):
    ...


class ParamUnitarySubDef(ParamOpSubDef[Params], ParamUnitaryDef[Params]):
    ...


class ParamNonUnitarySubDef(ParamOpSubDef[Params], ParamNonUnitaryDef[Params]):
    ...


def param_opsub(
    opsub_def: type[ParamOpSubDef[Params]],
    repository: SubRepository | None = default_repository(),
) -> tuple[OpFactory[Params], SubFactory[Params]]:
    _op = param_op(opsub_def)
    _sub = param_sub(opsub_def)
    if repository is not None:
        repository.register_sub(_op, _sub)
    return _op, _sub
