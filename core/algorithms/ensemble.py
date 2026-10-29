"""综合评分：多算法固定权重融合"""
from .base import Algorithm, top_text
from .. import stats
from .statistical import HotAlgorithm, ColdAlgorithm, OmissionAlgorithm, MarkovAlgorithm, MonteCarloAlgorithm


class EnsembleAlgorithm(Algorithm):
    key = "ensemble"
    name = "综合评分"
    desc = "融合热号、冷号、遗漏回归、马尔可夫、蒙特卡洛五种信号加权平均，兼顾短期热度与长期回补节奏，通常最稳健。"
    category = "综合推荐"

    WEIGHTS = {"hot": 0.26, "cold": 0.12, "omission": 0.20, "markov": 0.27, "monte": 0.15}

    def predict(self, draws, game, params):
        parts = {
            "hot": HotAlgorithm().predict(draws, game, {}),
            "cold": ColdAlgorithm().predict(draws, game, {}),
            "omission": OmissionAlgorithm().predict(draws, game, {}),
            "markov": MarkovAlgorithm().predict(draws, game, {}),
            "monte": MonteCarloAlgorithm().predict(draws, game, {}),
        }
        out = {}
        for pool in game.pools:
            combined = [0.0] * pool.max_number
            for key, weight in self.WEIGHTS.items():
                for i, v in enumerate(parts[key][pool.key]):
                    combined[i] += weight * v
            out[pool.key] = stats.normalize(combined)
        return out

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"五种算法加权融合（热号 0.26 / 马尔可夫 0.27 / 遗漏回归 0.20 / 蒙特卡洛 0.15 / 冷号 0.12），样本 {len(draws)} 期。"
                f"{m.name}综合评分最高：{top_text(scores[m.key], m.count)}；{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")
