"""霍华德经典方法三算法（源自盖尔·霍华德《彩票中奖指南》）

邻近数字动量：新开号码的相邻号（±1/±2）在下期延续出现的倾向——书中统计约八成期次
  含至少一个近期邻近号，平均每期 1.6 个；本书称之为「相邻数字 / 邻近数字系统」。
多窗口频率动量：书中的短/长窗口频率对照表（10/25/50/1354 期）数值化——比值大于 1
  的号码正在「升温」，小于 1 的正在「降温」。
伴侣数字链：书中「伴侣数字」表（历史最常同期出现的号码对）操作化为生成机制——
  以边际评分起步，每新增一个号码按「与已选号码的平均共现」加权逐号抽取。
"""
from .base import Algorithm, pool_predict, top_text, weighted_sample
from .. import stats


class AdjacencyAlgorithm(Algorithm):
    key = "adjacency"
    name = "邻近数字动量"
    category = "高级概率"
    desc = ("源自《彩票中奖指南》的「邻近数字」系统：对最近 W 期每个开出号码，向其 ±R 邻域"
            "（R=1 相邻号，R=2 含次邻近）注入时间衰减权重，捕捉「热区扩散」效应——历史统计约"
            "八成期次含至少一个近期邻近号。")
    params = [
        {"key": "window", "label": "回看期数", "default": 10, "min": 3, "max": 50, "step": 1},
        {"key": "radius", "label": "邻域半径", "default": 1, "min": 1, "max": 2, "step": 1},
    ]

    def predict(self, draws, game, params):
        W = int(params.get("window", 10))
        R = int(params.get("radius", 1))

        def score(d, p):
            N = p.max_number
            w = [0.0] * N
            recent = d[-W:] if len(d) > W else d
            for age, rec in enumerate(reversed(recent)):
                weight = 0.9 ** age
                for n in rec["numbers"][p.key]:
                    for delta in range(-R, R + 1):
                        if delta == 0:
                            continue
                        m = n + delta
                        if 1 <= m <= N:
                            w[m - 1] += weight
            return stats.normalize(w)

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        W = int(params.get("window", 10))
        R = int(params.get("radius", 1))
        main = game.main_pool
        hits = total = 0
        for i in range(max(W, 1), len(draws)):
            prev = draws[i - 1]["numbers"][main.key]
            neigh = {n + dd for n in prev for dd in range(-R, R + 1) if dd != 0}
            if set(draws[i]["numbers"][main.key]) & neigh:
                hits += 1
            total += 1
        rate = f"{hits / total * 100:.0f}%" if total else "—"
        return (f"近 {W} 期开号 ±{R} 邻域的热区扩散评分最高：{top_text(scores[main.key], main.count)}。"
                f"本游戏历史上「下期含上期邻近号」的期次占比约 {rate}（书中统计为 79%~95%）；"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class MomentumAlgorithm(Algorithm):
    key = "momentum"
    name = "多窗口频率动量"
    category = "经典统计"
    desc = ("把书中的多窗口频率对照表数值化为动量信号：短窗口出现率 ÷ 长窗口出现率 >1 为"
            "「升温号」、<1 为「降温号」，锐化指数放大比值差异。与时间衰减热号的区别在于"
            "它显式对比两个尺度，能同时排除「一直冷」与「正在降温」的号码。")
    params = [
        {"key": "short", "label": "短窗口(期)", "default": 10, "min": 5, "max": 30, "step": 1},
        {"key": "long", "label": "长窗口(期)", "default": 100, "min": 30, "max": 300, "step": 10},
        {"key": "sharpen", "label": "锐化指数", "default": 1.5, "min": 0.5, "max": 3.0, "step": 0.5},
    ]

    def predict(self, draws, game, params):
        s = int(params.get("short", 10))
        l = int(params.get("long", 100))
        e = float(params.get("sharpen", 1.5))

        def score(d, p):
            N = p.max_number
            recent = d[-s:] if len(d) > s else d
            long_d = d[-l:] if len(d) > l else d
            fs = stats.frequency(recent, p)
            fl = stats.frequency(long_d, p)
            ratio = [(fs[n - 1] / max(len(recent), 1) + 0.005) /
                     (fl[n - 1] / max(len(long_d), 1) + 0.005) for n in range(1, N + 1)]
            return stats.normalize([r ** e for r in ratio])

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        s = int(params.get("short", 10))
        main = game.main_pool
        recent = draws[-s:] if len(draws) > s else draws
        fs = stats.frequency(recent, main)
        fl = stats.frequency(draws, main)
        rising = sorted(range(1, main.max_number + 1),
                        key=lambda n: -(fs[n - 1] / max(s, 1)) / (fl[n - 1] / max(len(draws), 1) + 0.005))[:main.count]
        rise_text = "、".join(f"{n:02d}" for n in rising)
        return (f"以近 {s} 期与全程频率之比衡量，当前升温最快的{main.name}：{rise_text}。"
                f"综合动量评分 Top：{top_text(scores[main.key], main.count)}；"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class PairChainAlgorithm(Algorithm):
    key = "pairchain"
    name = "伴侣数字链"
    category = "优化与组合"
    desc = ("书中「伴侣数字」表的生成式实现：先按边际评分抽第一个号，之后每新增一个号码，"
            "权重 = 边际分 × (1 + β × 该号与已选号码的平均历史共现次数) 逐号抽取——把"
            "「哪些号爱一起出现」直接编进出票过程，而非事后过滤。")
    params = [{"key": "synergy", "label": "协同系数β", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.25}]

    def predict(self, draws, game, params):
        return pool_predict(lambda d, p: stats.normalize(stats.weighted_frequency(d, p, 0.97)), draws, game)

    def _pair_matrix(self, draws, pool):
        N = pool.max_number
        pair = [[0] * N for _ in range(N)]
        for combo in [d["numbers"][pool.key] for d in draws]:
            for i in combo:
                for j in combo:
                    if i != j:
                        pair[i - 1][j - 1] += 1
        return pair

    def _top_pair(self, draws, pool):
        pair = self._pair_matrix(draws, pool)
        best, bi, bj = -1, 1, 2
        for i in range(pool.max_number):
            for j in range(i + 1, pool.max_number):
                if pair[i][j] > best:
                    best, bi, bj = pair[i][j], i + 1, j + 1
        return bi, bj, best

    def generate_sets(self, draws, game, params, scores, combo_filter, n_sets, rng):
        main = game.main_pool
        K = main.count
        N = main.max_number
        beta = float(params.get("synergy", 1.0))
        pair = self._pair_matrix(draws, main)
        marg = stats.normalize(stats.weighted_frequency(draws, main, 0.97))

        sets, seen = [], set()
        attempts = 0
        budget = max(n_sets * 800, 4000)
        while len(sets) < n_sets and attempts < budget:
            attempts += 1
            chosen, remaining = [], list(range(1, N + 1))
            for _ in range(K):
                cands, ws = [], []
                for n in remaining:
                    syn = (sum(pair[n - 1][c - 1] for c in chosen) / len(chosen)) if chosen else 0.0
                    cands.append(n)
                    ws.append(max(marg[n - 1], 1e-6) * (1.0 + beta * syn))
                pick = weighted_sample(cands, ws, 1, rng)[0]
                chosen.append(pick)
                remaining.remove(pick)
            cand = tuple(sorted(chosen))
            if cand in seen or not combo_filter.ok(cand):
                continue
            seen.add(cand)
            sets.append(list(chosen))
        while len(sets) < n_sets:  # 兜底：纯边际抽样
            cand = tuple(weighted_sample(list(range(1, N + 1)),
                                         [max(marg[n - 1], 1e-6) for n in range(1, N + 1)], K, rng))
            if cand not in seen:
                seen.add(cand)
                sets.append(sorted(cand))
        self.generation_note = (f"伴侣数字链出票：每注按「边际分 × (1+β×平均共现)」逐号抽取"
                                f"（β={beta}），β 越大越倾向抱团的历史配对。")
        return sets, None

    def summarize(self, draws, game, params, scores):
        main = game.main_pool
        bi, bj, cnt = self._top_pair(draws, main)
        return (f"历史最常同期的「伴侣数字」对：{bi:02d} 与 {bj:02d}（同期 {cnt} 次）。"
                f"出票以时间衰减边际评分起步、按共现协同逐号抽取；{game.sub_pool.name}首选："
                f"{top_text(scores[game.sub_pool.key], 1)}。")


ALGORITHMS = [MomentumAlgorithm(), AdjacencyAlgorithm(), PairChainAlgorithm()]
