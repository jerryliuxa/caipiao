"""高级概率类算法（纯标准库实现）
生存分析 / 隐马尔可夫 / 卡尔曼滤波 / 变点检测 / 核密度估计 / MCMC / n-gram / Thompson 采样
"""
import math
import random

from .base import Algorithm, indicator, pool_predict, top_text
from .. import stats


def _gaps_and_age(seq, n, T):
    """号码 n 的历史出现间隔列表与当前遗漏"""
    gaps, last = [], -1
    for t in range(T):
        if seq[t][n]:
            if last >= 0:
                gaps.append(t - last)
            last = t
    return gaps, (T - 1 - last if last >= 0 else T)


class SurvivalAlgorithm(Algorithm):
    key = "survival"
    name = "生存分析"
    category = "高级概率"
    desc = "把每个号码的遗漏期数当作「生存时间」，用历史间隔估计经验风险函数 h(t)：已遗漏 t 期时下期出现的条件概率。比固定回补周期更贴合数据本身。"

    def predict(self, draws, game, params):
        def score(d, p):
            seq = indicator(d, p)
            T = len(seq)
            out = []
            for n in range(p.max_number):
                gaps, age = _gaps_and_age(seq, n, T)
                if len(gaps) < 3:
                    out.append(0.3)
                    continue
                mu = sum(gaps) / len(gaps)
                # 经验风险：h(age) = P(间隔==age | 间隔>=age-1)，±1 平滑 + 拉普拉斯
                near = sum(1 for g in gaps if abs(g - age) <= 1)
                surv = sum(1 for g in gaps if g >= age - 1)
                h_emp = (near + 0.5) / (surv + 2.0)
                out.append(0.6 * h_emp + 0.4 / max(mu, 1.0))
            return stats.normalize(out)

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"经验风险函数显示，当前遗漏处于「高风险区」的号码：{top_text(scores[m.key], m.count)}"
                f"（样本 {len(draws)} 期）。{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


def _hmm_predict(y, iters=14):
    """两状态伯努利 HMM（冷/热），Baum-Welch 估计，返回下期出现概率"""
    T = len(y)
    mean = sum(y) / T
    if mean <= 0.005 or mean >= 0.995:
        return mean
    p = [max(0.03, mean * 0.6), min(0.95, max(0.12, mean * 1.6))]
    A = [[0.92, 0.08], [0.08, 0.92]]
    for _ in range(iters):
        # 前向（缩放）
        alpha = [[0.0, 0.0] for _ in range(T)]
        c = [0.0] * T
        for i in range(2):
            alpha[0][i] = 0.5 * (p[i] if y[0] else 1 - p[i])
        c[0] = sum(alpha[0]) or 1e-12
        alpha[0] = [a / c[0] for a in alpha[0]]
        for t in range(1, T):
            for j in range(2):
                alpha[t][j] = (alpha[t - 1][0] * A[0][j] + alpha[t - 1][1] * A[1][j]) * (p[j] if y[t] else 1 - p[j])
            c[t] = sum(alpha[t]) or 1e-12
            alpha[t] = [a / c[t] for a in alpha[t]]
        # 后向（缩放）
        beta = [[1.0, 1.0] for _ in range(T)]
        for t in range(T - 2, -1, -1):
            for i in range(2):
                beta[t][i] = sum(A[i][j] * (p[j] if y[t + 1] else 1 - p[j]) * beta[t + 1][j]
                                 for j in range(2)) / c[t + 1]
        # E 步：gamma / xi
        gamma = [[alpha[t][i] * beta[t][i] for i in range(2)] for t in range(T)]
        gsum = [sum(g) or 1e-12 for g in gamma]
        gamma = [[g / s for g in gamma[t]] for t, s in enumerate(gsum)]
        xi_sum = [[0.0, 0.0], [0.0, 0.0]]
        for t in range(T - 1):
            for i in range(2):
                for j in range(2):
                    xi_sum[i][j] += (alpha[t][i] * A[i][j] * (p[j] if y[t + 1] else 1 - p[j])
                                     * beta[t + 1][j] / c[t + 1])
        # M 步
        for i in range(2):
            denom = sum(gamma[t][i] for t in range(T)) or 1e-12
            p[i] = min(0.97, max(0.02, sum(gamma[t][i] * y[t] for t in range(T)) / denom))
            row = sum(xi_sum[i]) or 1e-12
            for j in range(2):
                A[i][j] = xi_sum[i][j] / row
    return sum(gamma[T - 1][i] * sum(A[i][j] * p[j] for j in range(2)) for i in range(2))


class HmmAlgorithm(Algorithm):
    key = "hmm"
    name = "隐马尔可夫 HMM"
    category = "高级概率"
    desc = "假设每个号码背后存在「冷/热」两个隐状态并按马尔可夫链切换，用 Baum-Welch(EM) 从出现序列学习状态转移与出号率，再由滤波分布推断下期出现概率。"

    def predict(self, draws, game, params):
        def score(d, p):
            seq = indicator(d, p)
            T = len(seq)
            return stats.normalize([_hmm_predict([row[n] for row in seq]) for n in range(p.max_number)])

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"双状态 HMM 滤波推断的 {m.name}下期高概率号：{top_text(scores[m.key], m.count)}"
                f"（样本 {len(draws)} 期）。{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class KalmanAlgorithm(Algorithm):
    key = "kalman"
    name = "卡尔曼滤波"
    category = "高级概率"
    desc = "把每个号码的真实出现率建模为缓慢漂移的隐状态（随机游走），用卡尔曼滤波逐期修正估计——衰减力度由数据自动决定，是热号追踪的理论化版本。"
    params = [{"key": "q", "label": "漂移方差 Q", "default": 0.002, "min": 0.0005, "max": 0.01, "step": 0.0005}]

    def predict(self, draws, game, params):
        q = float(params.get("q", 0.002))

        def score(d, p):
            seq = indicator(d, p)
            T = len(seq)
            out = []
            for n in range(p.max_number):
                m, var = sum(seq[t][n] for t in range(T)) / T, 0.05
                for t in range(T):
                    var += q                                   # 预测：出现率漂移
                    r = min(max(m, 0.02), 0.98)
                    R = r * (1 - r) + 0.02                     # 伯努利观测噪声近似
                    K = var / (var + R)
                    m += K * (seq[t][n] - m)                   # 观测更新
                    var *= (1 - K)
                out.append(m)
            return stats.normalize(out)

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"卡尔曼滤波估计的当前动态出现率（Q={params.get('q', 0.002)}）排名前列：{top_text(scores[m.key], m.count)}。"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class ChangepointAlgorithm(Algorithm):
    key = "changepoint"
    name = "变点检测"
    category = "高级概率"
    desc = "对每个号码的出现序列做 CUSUM 式在线变点检测，定位最近一次「冷热切换」，只用切换之后的样本估计当前出现率，避免被陈旧历史稀释。"
    params = [{"key": "z_th", "label": "切换显著性阈值", "default": 2.0, "min": 1.0, "max": 3.5, "step": 0.1}]

    def predict(self, draws, game, params):
        z_th = float(params.get("z_th", 2.0))

        def score(d, p):
            seq = indicator(d, p)
            T = len(seq)
            out = []
            for n in range(p.max_number):
                y = [row[n] for row in seq]
                total = sum(y)
                gmean = total / T
                best_s, best_z = T, 0.0
                for s in range(max(1, T - 120), T - 8):
                    pre, post = sum(y[:s]) / s, sum(y[s:]) / (T - s)
                    pbar = (pre * s + post * (T - s)) / T
                    se = math.sqrt(max(pbar * (1 - pbar) * (1 / s + 1 / (T - s)), 1e-9))
                    z = abs(post - pre) / se
                    if z > best_z:
                        best_z, best_s = z, s
                if best_s < T and best_z >= z_th:
                    out.append(sum(y[best_s:]) / (T - best_s))
                else:
                    out.append(gmean)
            return stats.normalize(out)

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"检测到冷热切换后处于「新状态高位」的号码：{top_text(scores[m.key], m.count)}"
                f"（阈值 z≥{params.get('z_th', 2.0)}）。{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class KdeAlgorithm(Algorithm):
    key = "kde"
    name = "核密度估计 KDE"
    category = "高级概率"
    desc = "对每个号码的历史出现间隔做高斯核密度估计（Silverman 带宽）：当前遗漏落在其间隔分布的密度峰附近时得分高，不假设单一周期、允许多峰。"

    def predict(self, draws, game, params):
        def score(d, p):
            seq = indicator(d, p)
            T = len(seq)
            out = []
            for n in range(p.max_number):
                gaps, age = _gaps_and_age(seq, n, T)
                if len(gaps) < 3:
                    out.append(0.3)
                    continue
                mu = sum(gaps) / len(gaps)
                var = sum((g - mu) ** 2 for g in gaps) / len(gaps)
                bw = max(1.06 * var ** 0.5 * len(gaps) ** (-0.2), 1.2)
                dens = sum(math.exp(-0.5 * ((age - g) / bw) ** 2) for g in gaps) / (len(gaps) * bw)
                out.append(dens)
            return stats.normalize(out)

        return pool_predict(score, draws, game)


class McmcAlgorithm(Algorithm):
    key = "mcmc"
    name = "MCMC 后验采样"
    category = "高级概率"
    desc = "为全部号码的出现率建立带「邻位平滑先验」的联合贝叶斯后验，用 Metropolis-Hastings 采样估计边际出现率——频率估计叠加了向相邻号码收缩的正则化。"
    params = [{"key": "samples", "label": "采样步数", "default": 1500, "min": 500, "max": 4000, "step": 250},
              {"key": "smooth", "label": "平滑强度 σ", "default": 0.03, "min": 0.01, "max": 0.1, "step": 0.01}]

    def predict(self, draws, game, params):
        steps = int(params.get("samples", 1500))
        sigma = float(params.get("smooth", 0.03))

        def score(d, p):
            seq = indicator(d, p)
            T = len(seq)
            N = p.max_number
            counts = [sum(seq[t][n] for t in range(T)) for n in range(N)]
            rng = random.Random(42 + T * 7 + N)

            def logpost(r):
                ll = sum(counts[n] * math.log(r[n]) + (T - counts[n]) * math.log(1 - r[n]) for n in range(N))
                lp = -sum((r[n] - r[n + 1]) ** 2 for n in range(N - 1)) / (2 * sigma * sigma)
                return ll + lp

            r = [(counts[n] + 1) / (T + 2) for n in range(N)]
            cur = logpost(r)
            acc = 0.0
            for _ in range(steps):
                n = rng.randrange(N)
                old = r[n]
                r[n] = min(0.98, max(0.02, old + rng.gauss(0, 0.02)))
                new = logpost(r)
                if math.log(rng.random() + 1e-12) < new - cur:
                    cur, acc = new, acc + 1
                else:
                    r[n] = old
            return stats.normalize(r)

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"MH 采样 {params.get('samples', 1500)} 步的边际出现率（含邻位平滑先验 σ={params.get('smooth', 0.03)}）最高："
                f"{top_text(scores[m.key], m.count)}。{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class NgramAlgorithm(Algorithm):
    key = "ngram"
    name = "n-gram 序列模型"
    category = "高级概率"
    desc = "把号码近三期的出现/未出现组合当作 8 种「上下文骨架」，统计每种骨架下期出现的条件频率（拉普拉斯平滑），用当前骨架查表打分。"

    def predict(self, draws, game, params):
        def score(d, p):
            seq = indicator(d, p)
            T = len(seq)
            out = []
            for n in range(p.max_number):
                table = [[1, 2] for _ in range(8)]  # [命中, 总数] 拉普拉斯初值
                for t in range(3, T):
                    ctx = (seq[t - 1][n] << 2) | (seq[t - 2][n] << 1) | seq[t - 3][n]
                    table[ctx][0] += seq[t][n]
                    table[ctx][1] += 1
                cur = (seq[T - 1][n] << 2) | (seq[T - 2][n] << 1) | seq[T - 3][n]
                succ, tot = table[cur]
                out.append(succ / tot)
            return stats.normalize(out)

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"以近三期出现骨架为上下文的条件频率（3-gram）最高：{top_text(scores[m.key], m.count)}。"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


class ThompsonAlgorithm(Algorithm):
    key = "thompson"
    name = "Thompson 采样"
    category = "高级概率"
    desc = "贝叶斯多臂老虎机：每个号码的出现率维护 Beta 后验，反复从后验抽样并选出 Top-K，用海量模拟轮次中各号码的「入选频率」作为评分——把选号决策的不确定性纳入模型。"
    params = [{"key": "rounds", "label": "模拟轮数", "default": 300, "min": 100, "max": 1000, "step": 100}]

    def predict(self, draws, game, params):
        rounds = int(params.get("rounds", 300))

        def score(d, p):
            seq = indicator(d, p)
            T = len(seq)
            N = p.max_number
            counts = [sum(seq[t][n] for t in range(T)) for n in range(N)]
            rng = random.Random(20260000 + T * 13 + N)
            picks = [0] * N
            k = p.count
            for _ in range(rounds):
                rates = [rng.betavariate(1 + counts[n], 1 + T - counts[n]) for n in range(N)]
                order = sorted(range(N), key=lambda i: -rates[i])
                for i in order[:k]:
                    picks[i] += 1
            return stats.normalize([c / rounds for c in picks])

        return pool_predict(score, draws, game)

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"Beta 后验抽样 {params.get('rounds', 300)} 轮、每轮选 Top-{m.count} 的入选频率排名：{top_text(scores[m.key], m.count)}。"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


ALGORITHMS = [SurvivalAlgorithm(), HmmAlgorithm(), KalmanAlgorithm(), ChangepointAlgorithm(),
              KdeAlgorithm(), McmcAlgorithm(), NgramAlgorithm(), ThompsonAlgorithm()]
