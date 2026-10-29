"""算法注册表：汇总所有算法并按分类组织"""
from .base import Algorithm
from .statistical import (HotAlgorithm, ColdAlgorithm, OmissionAlgorithm,
                          MarkovAlgorithm, PeriodAlgorithm, MonteCarloAlgorithm)
from .ensemble import EnsembleAlgorithm
from . import advanced, optimize, stacking
from . import ml_heavy, dl_numpy


def _build():
    algos = [
        EnsembleAlgorithm(), stacking.StackingAlgorithm(),
        HotAlgorithm(), ColdAlgorithm(), OmissionAlgorithm(),
        MarkovAlgorithm(), PeriodAlgorithm(), MonteCarloAlgorithm(),
    ]
    algos += advanced.ALGORITHMS      # 生存分析 / HMM / 卡尔曼 / 变点 / KDE / MCMC / n-gram / Thompson
    algos += optimize.ALGORITHMS      # 遗传 / 共现网络 / 反热号 / 覆盖设计
    algos += ml_heavy.ALGORITHMS      # 逻辑回归 / GBDT / 随机森林 / MLP / 时序分解（需 sklearn）
    algos += dl_numpy.ALGORITHMS      # RNN / 注意力 / CNN / GNN / GAN / VAE（需 numpy）
    return algos


ALGORITHMS = _build()
_ALGO_MAP = {a.key: a for a in ALGORITHMS}

CATEGORY_ORDER = ["综合推荐", "经典统计", "高级概率", "机器学习", "深度学习", "时序预测", "优化与组合"]


def get_algorithm(key):
    if key not in _ALGO_MAP:
        raise ValueError(f"未知算法: {key}")
    return _ALGO_MAP[key]


def availability():
    return {
        "numpy": ml_heavy.AVAILABLE or dl_numpy.AVAILABLE,
        "sklearn": ml_heavy.AVAILABLE,
        "total": len(ALGORITHMS),
    }
