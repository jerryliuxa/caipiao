"""Stacking 自适应集成：以 6 种基础算法的输出为特征，滚动训练二级逻辑回归"""
import math

from .base import Algorithm, sigmoid, top_text
from .. import stats
from .statistical import HotAlgorithm, ColdAlgorithm, OmissionAlgorithm, MarkovAlgorithm, MonteCarloAlgorithm, PeriodAlgorithm

_BASES = [HotAlgorithm(), ColdAlgorithm(), OmissionAlgorithm(), MarkovAlgorithm(), MonteCarloAlgorithm(), PeriodAlgorithm()]
_BASE_KEYS = [a.key for a in _BASES]


def _train_logistic(rows, labels, l2=0.02, epochs=350, lr=0.6):
    """纯 Python 二级逻辑回归（含偏置、L2），返回 [w0, w1..wm]"""
    m = len(rows[0])
    w = [0.0] * (m + 1)
    n = len(rows)
    for _ in range(epochs):
        grad = [0.0] * (m + 1)
        for row, y in zip(rows, labels):
            z = w[0] + sum(wi * xi for wi, xi in zip(w[1:], row))
            e = sigmoid(z) - y
            grad[0] += e
            for i, xi in enumerate(row):
                grad[i + 1] += e * xi
        for i in range(m + 1):
            w[i] -= lr * (grad[i] / n + (l2 * w[i] if i else 0.0))
    return w


class StackingAlgorithm(Algorithm):
    key = "stacking"
    name = "Stacking 自适应集成"
    category = "综合推荐"
    desc = "把热号/冷号/遗漏回归/马尔可夫/蒙特卡洛/周期六种基础算法的输出作为特征，滚动窗口训练二级逻辑回归，自动学习「最近哪种算法更准」的动态权重——综合评分的数据驱动升级版。"

    def _base_scores(self, hist, game):
        return {a.key: a.predict(hist, game, {}) for a in _BASES}

    def predict(self, draws, game, params):
        T = len(draws)
        step = 3
        start = max(50, T - 32)
        rows, labels = [], []
        for t in range(start, T - 1, 4):
            hist = draws[max(0, t - 120):t]
            if len(hist) < 40:
                continue
            feats = self._base_scores(hist, game)
            for pool in game.pools:
                for n in pool.numbers:
                    rows.append([feats[k][pool.key][n - 1] for k in _BASE_KEYS])
                    labels.append(1 if n in draws[t]["numbers"][pool.key] else 0)
        if len(rows) < 30:  # 样本不足：退化为等权融合
            weights = [0.0] + [1.0 / len(_BASE_KEYS)] * len(_BASE_KEYS)
        else:
            weights = _train_logistic(rows, labels)
        # 用最新一期的基础算法评分做预测
        feats = self._base_scores(draws, game)
        out = {}
        for pool in game.pools:
            scores = []
            for n in pool.numbers:
                z = weights[0] + sum(w * feats[k][pool.key][n - 1] for w, k in zip(weights[1:], _BASE_KEYS))
                scores.append(sigmoid(z))
            out[pool.key] = stats.normalize(scores)
        self.generation_note = (f"二级逻辑回归权重（滚动训练 {len(rows)} 样本）："
                                + "、".join(f"{k}={w:+.2f}" for k, w in zip(_BASE_KEYS, weights[1:])))
        return out

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"自适应集成（滚动训练，各基础算法权重见下方说明）给出的 {m.name}高分号：{top_text(scores[m.key], m.count)}。"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


ALGORITHMS = [StackingAlgorithm()]
