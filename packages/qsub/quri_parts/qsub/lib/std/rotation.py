from quri_parts.qsub.op import ParamUnitaryDef, param_op

from . import NS


class _RX(ParamUnitaryDef[float]):
    ns = NS
    name = "RX"
    qubit_count = 1


RX = param_op(_RX)


class _RY(ParamUnitaryDef[float]):
    ns = NS
    name = "RY"
    qubit_count = 1


RY = param_op(_RY)


class _RZ(ParamUnitaryDef[float]):
    ns = NS
    name = "RZ"
    qubit_count = 1


RZ = param_op(_RZ)


class _Phase(ParamUnitaryDef[float]):
    ns = NS
    name = "Phase"
    qubit_count = 1


Phase = param_op(_Phase)
