"""numpy 轻量深度模型：RNN / 注意力 / 1D-CNN / GNN / GAN / VAE
需要 numpy；缺失时本模块注册为空，平台其余功能不受影响。
所有模型都是诚实的轻量实现（小网络 + 手写反向传播），用于展示建模思路而非追求预测力。
"""
from .base import Algorithm, indicator, pool_predict, top_text
from .. import stats

try:
    import numpy as np
    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

if AVAILABLE:
    def _seed(draws, pool):
        try:
            return int(str(draws[-1]["issue"])[-5:]) * 31 + pool.max_number * 7 + 1
        except (ValueError, KeyError):
            return 12345

    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _cumulative(seq):
        X = np.array(seq, dtype=float)
        return X, np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])

    # ---------------- RNN（Elman，BPTT） ----------------

    def _rnn_scores(seq, N, hidden=16, epochs=60, lr=0.12, seed=1, trunc=60):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T = len(X)
        Wxh = rng.normal(0, 0.25, (N, hidden))
        Whh = rng.normal(0, 0.25, (hidden, hidden))
        Why = rng.normal(0, 0.25, (hidden, N))
        bh, by = np.zeros(hidden), np.zeros(N)
        scale = 1.0 / max(T - 1, 1)
        t0 = max(0, T - trunc)
        for _ in range(epochs):
            hs = np.zeros((T, hidden))
            h = np.zeros(hidden)
            for t in range(T):
                h = np.tanh(X[t] @ Wxh + h @ Whh + bh)
                hs[t] = h
            loss_scale = 1.0 / max(T - 1 - t0, 1)
            dWxh, dWhh, dWhy = (np.zeros_like(a) for a in (Wxh, Whh, Why))
            dbh, dby = np.zeros_like(bh), np.zeros_like(by)
            dh_next = np.zeros(hidden)
            for t in range(T - 2, t0 - 1, -1):
                p = _sigmoid(hs[t] @ Why + by)
                do = (p - X[t + 1]) * scale
                dby += do
                dWhy += np.outer(hs[t], do)
                dh = do @ Why.T + dh_next
                dh_raw = (1 - hs[t] ** 2) * dh
                dbh += dh_raw
                dWxh += np.outer(X[t], dh_raw)
                if t > 0:
                    dWhh += np.outer(hs[t - 1], dh_raw)
                dh_next = dh_raw @ Whh.T
            for P, G in ((Wxh, dWxh), (Whh, dWhh), (Why, dWhy), (bh, dbh), (by, dby)):
                np.clip(G, -1, 1, out=G)
                P -= lr * G
        h = np.zeros(hidden)
        for t in range(T):
            h = np.tanh(X[t] @ Wxh + h @ Whh + bh)
        return _sigmoid(h @ Why + by)

    class RnnAlgorithm(Algorithm):
        key = "rnn"
        name = "循环神经网络 RNN"
        category = "深度学习"
        heavy = True
        desc = "Elman 循环网络（隐层 20，BPTT 训练），把每期开奖视为序列输入，用递归隐状态记忆历史、输出逐号码下期出现概率。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_rnn_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"RNN 递归隐状态读出的 {m.name}高概率号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- 注意力模型（单头 self-attention + 可训练读出） ----------------

    def _attention_scores(seq, N, d=16, epochs=150, lr=0.2, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T = len(X)
        Wq = rng.normal(0, 1 / np.sqrt(N), (N, d))
        Wk = rng.normal(0, 1 / np.sqrt(N), (N, d))
        Wv = rng.normal(0, 1 / np.sqrt(N), (N, N))
        Q, K = X @ Wq, X @ Wk
        S = K @ Q.T / np.sqrt(d)
        mask = np.triu(np.ones((T, T)), 1).astype(bool)
        S = np.where(mask, -1e9, S)
        S -= S.max(axis=0, keepdims=True)
        E = np.exp(S)
        A = E / E.sum(axis=0, keepdims=True)          # A[s, t]：第 t 期对 ≤t 各期的注意力
        W = rng.normal(0, 0.1, (2 * N, N))
        b = np.zeros(N)
        V = X @ Wv
        F_all = np.hstack([A.T @ V, X])               # T×2N
        F_train, Y_train = F_all[:-1], X[1:]          # 目标 = 下一期
        for _ in range(epochs):
            P = _sigmoid(F_train @ W + b)
            dO = (P - Y_train) / len(F_train)
            dW = F_train.T @ dO
            db = dO.sum(0)
            dF = dO @ W.T
            dC = dF[:, :N]
            dV = A[:, :dC.shape[0]] @ dC   # 训练只用前 T-1 个 query 位置
            dWv = X.T @ dV
            W -= lr * dW
            b -= lr * db
            np.clip(Wv, -2, 2, out=Wv)
            Wv -= lr * dWv
        return _sigmoid(F_all[-1] @ W + b)

    class AttentionAlgorithm(Algorithm):
        key = "attention"
        name = "Transformer 注意力"
        category = "深度学习"
        heavy = True
        desc = "单头自注意力的轻量实现：历史各期作为 token，注意力权重决定「回看哪些期」，值投影与读出头可训练（Q/K 为随机投影）。Transformer 的核心机制，去掉了堆叠与多头。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_attention_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"注意力机制聚合历史（可训练值投影）读出的 {m.name}高概率号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- 1D-CNN（多尺度卷积特征 + 线性头） ----------------

    _CONV_HEAD = None

    def _cnn_features_row(cum, t, N):
        def ma(w):
            lo = max(0, t + 1 - w)
            return (cum[t + 1] - cum[lo]) / (t + 1 - lo)
        ma3, ma5, ma10 = ma(3), ma(5), ma(10)
        lo3, lo5 = max(0, t + 1 - 3), max(0, t + 1 - 5)
        ma3p = (cum[max(1, t + 1 - 3)] - cum[max(0, t - 2)]) / max(1, t + 1 - 3 - max(0, t - 2))
        ma5p = (cum[max(1, t + 1 - 5)] - cum[max(0, t - 4)]) / max(1, t + 1 - 5 - max(0, t - 4))
        feats = np.vstack([ma3, ma5, ma10, ma3 - ma3p, ma5 - ma5p])  # 5×N
        return feats.T  # N×5

    def _cnn_scores(seq, N, epochs=250, lr=0.5, seed=1):
        X, cum = _cumulative(seq)
        T = len(X)
        rows, labels = [], []
        for t in range(30, T - 1):
            rows.append(_cnn_features_row(cum, t, N))
            labels.append(X[t + 1])
        if not rows:
            return X[-1].mean(axis=0)
        R = np.array(rows)          # (M, N, 5)
        Y = np.array(labels)        # (M, N)
        M = len(R)
        Fm = R.transpose(0, 2, 1)   # (M, 5, N)
        W = np.zeros((N, 6))        # 每号码独立线性头：[b, w1..w5]
        for _ in range(epochs):
            logits = W[:, 0][None, :] + np.einsum("mjn,nj->mn", Fm, W[:, 1:])
            P = _sigmoid(logits)
            E = (P - Y) / M
            dW0 = E.sum(0)
            dWj = np.einsum("mn,mjn->nj", E, Fm)
            W[:, 0] -= lr * dW0
            W[:, 1:] -= lr * dWj
        last = _cnn_features_row(cum, T - 1, N)  # N×5
        return _sigmoid(W[:, 0] + np.einsum("nj,nj->n", last, W[:, 1:]))

    class CnnAlgorithm(Algorithm):
        key = "cnn"
        name = "1D 卷积网络"
        category = "深度学习"
        heavy = True
        desc = "把号码出现序列当作一维信号，用多尺度卷积核（3/5/10 期均值、动量、差分）提取局部模式，再接可训练线性分类头——1D-CNN 的特征提取思想。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_cnn_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"多尺度卷积特征（3/5/10 期核）+ 线性头读出的 {m.name}高概率号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- GNN（共现图消息传递） ----------------

    def _gnn_scores(seq, N, hidden=16, epochs=150, lr=0.3, seed=1):
        rng = np.random.default_rng(seed)
        X, cum = _cumulative(seq)
        T = len(X)
        co = X[: min(T, 150)].T @ X[: min(T, 150)]
        np.fill_diagonal(co, 0)
        Ahat = (co + np.eye(N)) / (co.sum(1, keepdims=True) + 1)
        # 节点特征：[ma10, 遗漏, 衰减热度]（逐期增量构造）
        min_t = 30
        F_all, Y_all = [], []
        decay = np.zeros(N)
        last_seen = np.full(N, -1)
        for t in range(T):
            decay = decay * 0.97 + X[t]
            if t >= min_t:
                lo = max(0, t + 1 - 10)
                ma10 = (cum[t + 1] - cum[lo]) / (t + 1 - lo)
                omiss = np.where(last_seen < 0, t + 1, t - last_seen)
                if t < T - 1:
                    F_all.append(np.stack([ma10, omiss / (t + 1), decay], axis=1))  # N×3
                    Y_all.append(X[t + 1])
            last_seen[X[t] > 0] = t
        if not F_all:
            return X[-1].mean(axis=0)
        F_all = np.array(F_all)          # (M, N, 3)
        Y_all = np.array(Y_all)          # (M, N)
        M = len(F_all)
        W1 = rng.normal(0, 0.4, (3, hidden))
        W2 = rng.normal(0, 0.4, (hidden, hidden))
        w = rng.normal(0, 0.4, hidden)
        for _ in range(epochs):
            G1 = Ahat @ F_all                     # (M, N, 3)
            H1 = np.tanh(G1 @ W1)                 # (M, N, hidden)
            H2in = Ahat @ H1
            H2 = np.tanh(H2in @ W2)
            P = _sigmoid(H2 @ w)                  # (M, N)
            E = (P - Y_all) / (M * N)
            dH2 = E[..., None] * w                # (M, N, hidden)
            dW2 = np.einsum("knm,kni->mi", H2in, dH2) / M
            din2 = dH2 @ W2.T
            dH1 = np.matmul(Ahat.T, din2) * (1 - H1 ** 2)
            dW1 = np.einsum("kmj,kmi->ji", G1, dH1) / M
            dw = np.einsum("knm,kn->m", H2, E) / M
            W1 -= lr * np.clip(dW1, -1, 1)
            W2 -= lr * np.clip(dW2, -1, 1)
            w -= lr * np.clip(dw, -1, 1)
        # 最后一期前向
        lo = max(0, T - 10)
        ma10 = (cum[T] - cum[lo]) / (T - lo)
        last_seen = np.array([last_seen[n] for n in range(N)])
        omiss = np.where(last_seen < 0, T, T - 1 - last_seen)
        F = np.stack([ma10, omiss / T, decay], axis=1)
        H1 = np.tanh((Ahat @ F) @ W1)
        H2 = np.tanh((Ahat @ H1) @ W2)
        return _sigmoid(H2 @ w)

    class GnnAlgorithm(Algorithm):
        key = "gnn"
        name = "图神经网络 GNN"
        category = "深度学习"
        heavy = True
        desc = "以号码共现图为邻接、多窗口统计为节点特征，做两层图卷积消息传递（每个号码聚合其共现邻居的信息），再由读出层输出出现概率。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_gnn_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"两层图卷积（共现消息传递）读出的 {m.name}高概率号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- GAN（轻量生成对抗） ----------------

    def _gan_scores(seq, N, zdim=16, hidden=32, iters=250, lr=0.03, batch=32, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T = len(X)
        G1, Gb1 = rng.normal(0, 0.3, (zdim, hidden)), np.zeros(hidden)
        G2, Gb2 = rng.normal(0, 0.3, (hidden, N)), np.zeros(N)
        D1, Db1 = rng.normal(0, 0.3, (N, hidden)), np.zeros(hidden)
        D2, Db2 = rng.normal(0, 0.3, (hidden, 1)), np.zeros(1)

        def leaky(v):
            return np.where(v > 0, v, 0.2 * v)

        for it in range(iters):
            for _ in range(2):  # 判别器两步
                real = X[rng.integers(0, T, batch)]
                z = rng.normal(0, 1, (batch, zdim))
                zh = np.maximum(z @ G1 + Gb1, 0)
                fake = _sigmoid(zh @ G2 + Gb2)
                hr = leaky(real @ D1 + Db1)
                dr = _sigmoid(hr @ D2 + Db2)
                hf = leaky(fake @ D1 + Db1)
                df_ = _sigmoid(hf @ D2 + Db2)
                m = batch
                dlogit_r = (dr - 1) / m
                dlogit_f = df_ / m
                gD1, gDb1, gD2, gDb2 = (np.zeros_like(a) for a in (D1, Db1, D2, Db2))
                for xx, hh, dl in ((real, hr, dlogit_r), (fake, hf, dlogit_f)):
                    dh = (dl @ D2.T) * (hh > 0) + 0.2 * (dl @ D2.T) * (hh <= 0)
                    gD2 += hh.T @ dl
                    gDb2 += dl.sum(0)
                    gD1 += xx.T @ dh
                    gDb1 += dh.sum(0)
                for P, G in ((D1, gD1), (Db1, gDb1), (D2, gD2), (Db2, gDb2)):
                    P -= lr * np.clip(G, -1, 1)
            # 生成器一步
            z = rng.normal(0, 1, (batch, zdim))
            zh = np.maximum(z @ G1 + Gb1, 0)
            fake = _sigmoid(zh @ G2 + Gb2)
            hf = leaky(fake @ D1 + Db1)
            p_fake = _sigmoid(hf @ D2 + Db2)
            dlogit = p_fake / batch           # -log(1-D) 的梯度
            dh = (dlogit @ D2.T) * (hf > 0) + 0.2 * (dlogit @ D2.T) * (hf <= 0)
            dfake = dh @ D1.T
            dz2 = dfake * fake * (1 - fake)
            gG2 = zh.T @ dz2
            gGb2 = dz2.sum(0)
            dzh = dz2 @ G2.T
            dzh = dzh * (zh > 0)
            gG1 = z.T @ dzh
            gGb1 = dzh.sum(0)
            for P, G in ((G1, gG1), (Gb1, gGb1), (G2, gG2), (Gb2, gGb2)):
                P -= lr * np.clip(G, -1, 1)
        z = rng.normal(0, 1, (400, zdim))
        zh = np.maximum(z @ G1 + Gb1, 0)
        return _sigmoid(zh @ G2 + Gb2).mean(axis=0)

    class GanAlgorithm(Algorithm):
        key = "gan"
        name = "生成对抗 GAN"
        category = "深度学习"
        heavy = True
        desc = "轻量 GAN：生成器把随机噪声映射为「像开奖」的号码向量，判别器学习区分真开奖，对抗训练后用生成器的平均输出作为号码概率分布。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_gan_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"对抗训练后的生成器平均输出（生成分布的边际概率）最高：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- VAE（变分自编码器） ----------------

    def _vae_scores(seq, N, zdim=6, hidden=32, epochs=250, lr=0.05, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T = len(X)
        We1, be1 = rng.normal(0, 0.3, (N, hidden)), np.zeros(hidden)
        Wmu, bmu = rng.normal(0, 0.3, (hidden, zdim)), np.zeros(zdim)
        Wlv, blv = rng.normal(0, 0.3, (hidden, zdim)), np.zeros(zdim)
        Wd1, bd1 = rng.normal(0, 0.3, (zdim, hidden)), np.zeros(hidden)
        Wd2, bd2 = rng.normal(0, 0.3, (hidden, N)), np.zeros(N)
        beta = 0.3
        for _ in range(epochs):
            H = np.tanh(X @ We1 + be1)
            mu, lv = H @ Wmu + bmu, H @ Wlv + blv
            eps = rng.normal(0, 1, (T, zdim))
            z = mu + np.exp(0.5 * lv) * eps
            D1 = np.tanh(z @ Wd1 + bd1)
            P = _sigmoid(D1 @ Wd2 + bd2)
            # 反向
            dP = (P - X) / (T * N)
            dWd2 = D1.T @ dP
            dbd2 = dP.sum(0)
            dD1 = dP @ Wd2.T * (1 - D1 ** 2)
            dz = dD1 @ Wd1.T
            dWd1 = z.T @ dD1
            dbd1 = dD1.sum(0)
            dmu = dz + beta * mu / T
            dlv = dz * eps * np.exp(0.5 * lv) * 0.5 + beta * 0.5 * (np.exp(lv) - 1) / T
            dH = dmu @ Wmu.T + dlv @ Wlv.T
            dH *= (1 - H ** 2)
            dWe1 = X.T @ dH
            dbe1 = dH.sum(0)
            dWmu = H.T @ dmu
            dbmu = dmu.sum(0)
            dWlv = H.T @ dlv
            dblv = dlv.sum(0)
            for P_, G in ((We1, dWe1), (be1, dbe1), (Wmu, dWmu), (bmu, dbmu),
                          (Wlv, dWlv), (blv, dblv), (Wd1, dWd1), (bd1, dbd1),
                          (Wd2, dWd2), (bd2, dbd2)):
                P_ -= lr * np.clip(G, -1, 1)
        H = np.tanh(X[-1] @ We1 + be1)
        z = H @ Wmu + bmu
        D1 = np.tanh(z @ Wd1 + bd1)
        return _sigmoid(D1 @ Wd2 + bd2)

    class VaeAlgorithm(Algorithm):
        key = "vae"
        name = "变分自编码器 VAE"
        category = "深度学习"
        heavy = True
        desc = "编码器把开奖组合压入低维隐空间（重参数化采样 + ELBO 训练），解码器学习重建分布；用最新一期的后验均值解码，得到号码级重建概率。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_vae_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"VAE 隐空间重建概率最高的 {m.name}号码：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    ALGORITHMS = [RnnAlgorithm(), AttentionAlgorithm(), CnnAlgorithm(),
                  GnnAlgorithm(), GanAlgorithm(), VaeAlgorithm()]
else:
    ALGORITHMS = []
