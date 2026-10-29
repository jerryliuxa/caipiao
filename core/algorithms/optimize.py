"""优化与组合类算法（纯标准库实现）
遗传算法 / 共现网络 / 反热号·最大熵 / 覆盖设计·旋转缩水
"""
from itertools import combinations

from .base import Algorithm, pool_predict, ranked, top_text, weighted_sample
from .. import stats


class GeneticAlgorithm(Algorithm):
    key = "genetic"
    name = "遗传算法"
    category = "优化与组合"
    desc = "把「号码评分 + 历史形态约束」作为适应度函数，随机生成组合种群，经精英保留、锦标赛选择、均匀交叉、变异进化出高适应度组合，多注之间保持多样性。"
    params = [{"key": "generations", "label": "进化代数", "default": 80, "min": 30, "max": 200, "step": 10}]

    def predict(self, draws, game, params):
        return pool_predict(lambda d, p: stats.normalize(stats.weighted_frequency(d, p, 0.97)), draws, game)

    def generate_sets(self, draws, game, params, scores, combo_filter, n_sets, rng):
        main = game.main_pool
        K, N = main.count, main.max_number
        gens = int(params.get("generations", 80))
        pop_size = 60
        all_nums = list(range(1, N + 1))
        weights = [max(scores[n - 1], 1e-6) for n in all_nums]

        def fitness(combo):
            base = sum(scores[n - 1] for n in combo) / K
            return base * (1.0 if combo_filter.ok(combo) else 0.55)

        pop, seen = [], set()
        while len(pop) < pop_size:
            ind = tuple(weighted_sample(all_nums, weights, K, rng))
            if ind not in seen:
                seen.add(ind)
                pop.append(ind)

        for _ in range(gens):
            fits = [fitness(ind) for ind in pop]
            order = sorted(range(pop_size), key=lambda i: -fits[i])
            newpop = [pop[i] for i in order[:6]]  # 精英保留
            while len(newpop) < pop_size:
                i, j = rng.randrange(pop_size), rng.randrange(pop_size)
                parent = pop[i] if fits[i] >= fits[j] else pop[j]
                i, j = rng.randrange(pop_size), rng.randrange(pop_size)
                parent2 = pop[i] if fits[i] >= fits[j] else pop[j]
                gene = list({*parent, *parent2})          # 均匀交叉
                if len(gene) > K:
                    gene = sorted(gene, key=lambda n: -scores[n - 1])[:K]
                while len(gene) < K:
                    c = rng.choice(all_nums)
                    if c not in gene:
                        gene.append(c)
                if rng.random() < 0.25:                   # 变异
                    pos = rng.randrange(K)
                    outs = [n for n in all_nums if n not in gene]
                    if outs:
                        gene[pos] = weighted_sample(outs, [max(scores[n - 1], 1e-6) for n in outs], 1, rng)[0]
                ind = tuple(sorted(gene))
                newpop.append(ind)
            pop = newpop

        fits = sorted(((fitness(ind), ind) for ind in pop), key=lambda x: -x[0])
        sets, used = [], set()
        for _, ind in fits:
            if ind not in used:
                used.add(ind)
                sets.append(list(ind))
            if len(sets) >= n_sets:
                break
        while len(sets) < n_sets:
            ind = tuple(weighted_sample(all_nums, weights, K, rng))
            if ind not in used:
                used.add(ind)
                sets.append(list(ind))
        return sets[:n_sets], None

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"以衰减加权评分为基础适应度，进化 {params.get('generations', 80)} 代（种群 60）得到的 {m.name}高分基因："
                f"{top_text(scores[m.key], m.count)}。{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class CooccurrenceAlgorithm(Algorithm):
    key = "cooccurrence"
    name = "共现网络"
    category = "优化与组合"
    desc = "把「同期共现」视为网络连线构建号码关系图，用 PageRank 幂迭代计算每个号码的网络中心性——常与热号抱团出现的号码获得更高权重，是马尔可夫思路的静态版。"
    params = [{"key": "damping", "label": "阻尼系数", "default": 0.85, "min": 0.5, "max": 0.95, "step": 0.01}]

    def predict(self, draws, game, params):
        damp = float(params.get("damping", 0.85))

        def score(d, p):
            N = p.max_number
            w = [[0.0] * N for _ in range(N)]
            for combo in [x["numbers"][p.key] for x in d]:
                for i in combo:
                    for j in combo:
                        if i != j:
                            w[i - 1][j - 1] += 1
            pr = [1.0 / N] * N
            for _ in range(60):
                new = [(1 - damp) / N] * N
                for i in range(N):
                    row_sum = sum(w[i])
                    if row_sum:
                        share = pr[i] * damp / row_sum
                        for j in range(N):
                            new[j] += share * w[i][j]
                    else:
                        share = pr[i] * damp / N
                        for j in range(N):
                            new[j] += share
                pr = new
            return stats.normalize(pr)

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"共现网络 PageRank（阻尼 {params.get('damping', 0.85)}）中心性最高、最“抱团”的号码：{top_text(scores[m.key], m.count)}。"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class AntiHeatAlgorithm(Algorithm):
    key = "antiheat"
    name = "反热号·最大熵"
    category = "优化与组合"
    desc = "反向操作：压低近期高频与刚开出号码的权重，倾向均匀分散的组合，与热号策略互为镜像；组合形态仍由历史分布约束兜底。"

    def predict(self, draws, game, params):
        def score(d, p):
            wf = stats.weighted_frequency(d, p, 0.97)
            return stats.normalize([1.0 / (v + 0.05) for v in wf])

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"最大熵思路下最“冷偏均匀”的号码：{top_text(scores[m.key], m.count)}（刻意避开热门与重号）。"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class CoverAlgorithm(Algorithm):
    key = "cover"
    name = "覆盖设计·旋转缩水"
    category = "优化与组合"
    desc = "先选出心水号码池（评分 Top-P），再用贪心法求解彩票设计（lotto design）：只要开奖号全部落在池内，必有一注至少命中 G 个——唯一带数学保证的出票结构（保证命中下限，不是中奖率）。"
    params = [{"key": "pool_size", "label": "心水池大小", "default": 9, "min": 7, "max": 12, "step": 1},
              {"key": "guarantee", "label": "保证命中数", "default": 5, "min": 3, "max": 6, "step": 1}]

    def predict(self, draws, game, params):
        return pool_predict(lambda d, p: stats.normalize(stats.weighted_frequency(d, p, 0.97)), draws, game)

    def generate_sets(self, draws, game, params, scores, combo_filter, n_sets, rng):
        main = game.main_pool
        K = main.count
        P = min(int(params.get("pool_size", 9)), main.max_number)
        G = min(int(params.get("guarantee", 5)), K, P)
        pool_nums = ranked(scores)[:P]
        # 位掩码：universe = 开奖号全落池内的所有可能（C(P,K) 个 K 子集）
        universe = [sum(1 << i for i in comb) for comb in combinations(range(P), K)]
        candidates = list(universe)
        tickets = []
        while universe:
            best, best_cov = None, -1
            for x in candidates:
                cov = sum(1 for T in universe if (x & T).bit_count() >= G)
                if cov > best_cov:
                    best, best_cov = x, cov
            tickets.append(best)
            universe = [T for T in universe if (best & T).bit_count() < G]

        main_sets = [[pool_nums[i] for i in range(P) if (t >> i) & 1] for t in tickets]
        full_count = len(main_sets)
        # 永不截断：截断会破坏数学保证；注数可多于请求数
        self.generation_note = (f"覆盖设计（池 {P} 选 {K}、保证 {G} 中）最优近似解 {full_count} 注，全部保留以维持数学保证"
                                + ("（多于请求数）" if full_count > n_sets else "") + "。")
        if full_count < n_sets:
            seen = {tuple(s) for s in main_sets}
            attempts = 0
            while len(main_sets) < n_sets and attempts < 3000:
                attempts += 1
                cand = tuple(weighted_sample(pool_nums, [max(scores[n - 1], 1e-6) for n in pool_nums], K, rng))
                if cand not in seen:
                    seen.add(cand)
                    main_sets.append(list(cand))
        return main_sets, None

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"心水池取评分 Top-{params.get('pool_size', 9)}，目标「开奖全在池内时必有一注命中 {params.get('guarantee', 5)} 个」。"
                f"池内评分参考：{top_text(scores[m.key], m.count)}。")


ALGORITHMS = [GeneticAlgorithm(), CooccurrenceAlgorithm(), AntiHeatAlgorithm(), CoverAlgorithm()]
