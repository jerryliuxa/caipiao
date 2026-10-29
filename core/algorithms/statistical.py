"""经典统计算法：热号 / 冷号 / 遗漏回归 / 马尔可夫 / 周期 / 蒙特卡洛"""
from .base import Algorithm, pool_predict, top_text
from .. import stats


class HotAlgorithm(Algorithm):
    key = "hot"
    name = "热号追踪"
    desc = "对近 N 期按时间衰减加权统计出现频率，越近出现权重越高，追随当前活跃热号。适合认为“热者恒热”的思路。"
    params = [{"key": "decay", "label": "时间衰减系数 λ", "default": 0.97, "min": 0.90, "max": 0.999, "step": 0.001}]

    def predict(self, draws, game, params):
        decay = float(params.get("decay", 0.97))
        return pool_predict(lambda d, p: stats.normalize(stats.weighted_frequency(d, p, decay)), draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"对近 {len(draws)} 期做指数衰减加权（λ={params.get('decay', 0.97)}），近期权重更高。"
                f"{m.name}热号 Top{m.count}：{top_text(scores[m.key], m.count)}；"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class ColdAlgorithm(Algorithm):
    key = "cold"
    name = "冷号回补"
    desc = "优先选择当前遗漏期数最大的冷号，押注“冷久了该出了”的回补效应。风险较高，与热号策略互补。"

    def predict(self, draws, game, params):
        return pool_predict(lambda d, p: stats.normalize([float(g) for g in stats.omission(d, p)]), draws, game)


class OmissionAlgorithm(Algorithm):
    key = "omission"
    name = "遗漏均值回归"
    desc = "以「当前遗漏 ÷ 平均遗漏」衡量每个号码的到期程度，比值越高说明越超出其历史回补节奏，越接近回归。"
    params = [{"key": "ratio_cap", "label": "比值封顶", "default": 3.0, "min": 1.5, "max": 6.0, "step": 0.1}]

    def predict(self, draws, game, params):
        cap = float(params.get("ratio_cap", 3.0))

        def score(d, p):
            cur = stats.omission(d, p)
            avg = stats.avg_gaps(d, p)
            return stats.normalize([min(c / a, cap) for c, a in zip(cur, avg)])

        return pool_predict(score, draws, game)


class MarkovAlgorithm(Algorithm):
    key = "markov"
    name = "马尔可夫链"
    desc = "统计历史上每个号码出现后，下一期各号码的条件出现频率，用最新一期开出的号码加权推断下一期分布（一阶转移）。"

    def predict(self, draws, game, params):
        alpha = 0.5

        def score(d, p):
            maxn = p.max_number
            trans = [[0] * maxn for _ in range(maxn)]
            for a, b in zip(d, d[1:]):
                for i in a["numbers"][p.key]:
                    for j in b["numbers"][p.key]:
                        trans[i - 1][j - 1] += 1
            last = d[-1]["numbers"][p.key]
            scores = [0.0] * maxn
            for i in last:
                row = trans[i - 1]
                tot = sum(row)
                for j in range(maxn):
                    scores[j] += (row[j] + alpha) / (tot + alpha * maxn)
            return stats.normalize(scores)

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        last = "、".join(f"{n:02d}" for n in draws[-1]["numbers"][m.key])
        return (f"一阶马尔可夫：由上期 {m.name}（{last}）的条件转移概率推断。"
                f"下一期 {m.name}高概率号：{top_text(scores[m.key], m.count)}；{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class PeriodAlgorithm(Algorithm):
    key = "period"
    name = "周期波动分析"
    desc = "对每个号码的出现序列检测其平均回补周期与稳定性，若当前遗漏正处于周期的尾段（相位接近 1），给出高分。"

    def predict(self, draws, game, params):
        def score(d, p):
            maxn = p.max_number
            n_draws = len(d)
            out = [0.12] * maxn
            for num in p.numbers:
                idxs = [t for t, dr in enumerate(d) if num in dr["numbers"][p.key]]
                if len(idxs) < 3:
                    continue
                gaps = [b - a for a, b in zip(idxs, idxs[1:])]
                mu = sum(gaps) / len(gaps)
                std = (sum((g - mu) ** 2 for g in gaps) / len(gaps)) ** 0.5
                strength = 1 / (1 + std / mu)
                age = n_draws - 1 - idxs[-1]
                phase = (age % mu) / mu if mu > 0 else 0
                reward = max(0.0, (phase - 0.5) * 2)
                out[num - 1] = 0.12 + 0.88 * strength * reward
            return stats.normalize(out)

        return pool_predict(score, draws, game)


class MonteCarloAlgorithm(Algorithm):
    key = "monte"
    name = "蒙特卡洛模拟"
    desc = "以「先验 + 历史频率」构造 Dirichlet 多项式后验，作为各号码的出现概率分布，再按概率抽样生成组合。平滑处理避免冷号概率为零。"
    params = [{"key": "prior", "label": "先验强度 α", "default": 1.0, "min": 0.2, "max": 5.0, "step": 0.1}]

    def predict(self, draws, game, params):
        prior = float(params.get("prior", 1.0))

        def score(d, p):
            freq = stats.frequency(d, p)
            total = sum(freq) + prior * p.max_number
            return stats.normalize([(prior + c) / total for c in freq])

        return pool_predict(score, draws, game)
