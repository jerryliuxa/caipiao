"""算法基类与公共工具"""
import math


class Algorithm:
    key = ""
    name = ""
    desc = ""
    category = "经典统计"
    heavy = False      # 是否依赖 numpy / sklearn
    params = []        # [{"key","label","default","min","max","step"}]
    generation_note = ""  # generate_sets 自定义出票时可写入的补充说明

    def predict(self, draws, game, params):
        """返回 {pool_key: [评分列表]}，评分越大越值得关注"""
        raise NotImplementedError

    def generate_sets(self, draws, game, params, scores, combo_filter, n_sets, rng):
        """可选：自定义主池出票方式。返回 (main_sets, sub_sets或None)；默认返回 None 走通用管线。"""
        return None

    def summarize(self, draws, game, params, scores):
        m = game.main_pool
        return (f"基于近 {len(draws)} 期数据分析，{m.name}评分最高：{top_text(scores[m.key], m.count)}；"
                f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")


def indicator(draws, pool):
    """T×N 的 0/1 出现矩阵（行=期，列=号码-1）"""
    N = pool.max_number
    return [[1 if n in d["numbers"][pool.key] else 0 for n in range(1, N + 1)] for d in draws]


def ranked(scores):
    return sorted(range(1, len(scores) + 1), key=lambda n: -scores[n - 1])


def top_text(scores, k):
    return "、".join(f"{n:02d}" for n in ranked(scores)[:k])


def pool_predict(func, draws, game):
    return {p.key: func(draws, p) for p in game.pools}


def weighted_sample(candidates, weights, k, rng):
    """按权重不放回抽 k 个，返回升序列表"""
    pool = list(zip(candidates, weights))
    out = []
    for _ in range(k):
        total = sum(w for _, w in pool)
        if total <= 0:
            rest = [c for c, _ in pool]
            out += rng.sample(rest, min(k - len(out), len(rest)))
            break
        r = rng.random() * total
        acc = 0.0
        for i, (c, w) in enumerate(pool):
            acc += w
            if acc >= r:
                out.append(pool.pop(i)[0])
                break
    if len(out) < k:
        out += [c for c, _ in pool[: k - len(out)]]
    return sorted(out)


def build_supervised(draws, pool, min_train=40):
    """构造逐(期,号码)监督样本，供 sklearn/深度模型使用。

    返回 (X, y, X_next, N)：X 形状 (rows, F)，X_next 为最新一期对下一期的 N 行特征。
    需要 numpy。
    """
    import numpy as np

    N = pool.max_number
    seq = indicator(draws, pool)
    T = len(seq)
    min_train = min(min_train, max(T - 2, 1))
    X = np.array(seq, dtype=float)
    cum = np.vstack([np.zeros(N), np.cumsum(X, axis=0)])  # cum[t] = 前 t 期累计
    lam = 0.97
    decay = np.zeros(N)
    last_seen = np.full(N, -1)
    F_BASE = 11

    rows, labels = [], []

    def emit(t):
        """用截至第 t 期(含)的数据，为预测第 t+1 期生成 N 行特征"""
        f30 = (cum[t + 1] - cum[max(0, t + 1 - 30)]) / 30.0
        f5 = (cum[t + 1] - cum[max(0, t + 1 - 5)]) / 5.0
        fall = cum[t + 1] / (t + 1)
        omissions = t - last_seen          # 从未出现则为 -1 → 修正为 t+1
        omissions = np.where(last_seen < 0, t + 1, omissions)
        counts = np.maximum(cum[t + 1], 0.5)
        avg_gap = (t + 1) / counts
        ratio = np.minimum(omissions / avg_gap, 3.0)
        zone = (np.arange(N) // math.ceil(N / 3)).clip(0, 2)
        for n in range(N):
            onehot = [0.0] * N
            onehot[n] = 1.0
            z = [0.0, 0.0, 0.0]
            z[int(zone[n])] = 1.0
            rows.append([f30[n], f5[n], fall[n], decay[n],
                         omissions[n] / (t + 1), ratio[n],
                         seq[t][n], (n + 1) / N, float(n % 2), float(n % 3) / 2.0,
                         1.0 if (n + 1) in _primes else 0.0] + z + onehot)
            labels.append(seq[t + 1][n] if t + 1 < T else None)

    for t in range(T):
        decay = decay * lam + X[t]
        last_seen[X[t] > 0] = t
        if t >= min_train:
            emit(t)

    # 拆分：除最后一期外为训练样本，最后一期为 X_next
    train_rows, train_labels, next_rows = [], [], []
    for row, lab in zip(rows, labels):
        (next_rows if lab is None else train_rows).append(row)
        if lab is not None:
            train_labels.append(lab)
    if not next_rows:  # 数据过短时退化为用最后一份训练特征
        next_rows = rows[-N:]
    return (np.array(train_rows), np.array(train_labels),
            np.array(next_rows), N)


_primes = {n for n in range(2, 36) if all(n % d for d in range(2, int(math.sqrt(n)) + 1))}


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
