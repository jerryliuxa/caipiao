"""numpy AI 模型第二批：LSTM / Transformer 多头 / RBM / ESN 储备池 / Skip-Gram 嵌入 /
相似期检索 kNN / 聚类原型 K-Means
需要 numpy；缺失时本模块注册为空，平台其余功能不受影响。
所有模型都是诚实的轻量实现，用于展示建模思路而非追求预测力。
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
            return int(str(draws[-1]["issue"])[-5:]) * 31 + pool.max_number * 13 + 5
        except (ValueError, KeyError):
            return 54321

    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _cumulative(seq):
        X = np.array(seq, dtype=float)
        return X, np.vstack([np.zeros(X.shape[1]), np.cumsum(X, axis=0)])

    def _pe(T, d):
        """正弦位置编码（固定，不训练）"""
        pe = np.zeros((T, d))
        pos = np.arange(T)[:, None].astype(float)
        i2 = np.arange(0, d, 2).astype(float)
        ang = pos / 10000 ** (i2 / d)
        pe[:, 0::2] = np.sin(ang)
        pe[:, 1::2] = np.cos(ang)
        return pe

    def _ln_backward(g, xhat, x_std):
        """LayerNorm 反向（无参数版）"""
        return (g - g.mean(1, keepdims=True)
                - xhat * (g * xhat).mean(1, keepdims=True)) / x_std

    # ---------------- LSTM（门控循环网络，截断 BPTT） ----------------

    def _lstm_scores(seq, N, hidden=12, epochs=25, lr=0.15, trunc=40, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T = len(X)
        H = hidden
        s = 1.0 / np.sqrt(N + H)
        Wx = {k: rng.normal(0, s, (N, H)) for k in "ifog"}
        Wh = {k: rng.normal(0, s, (H, H)) for k in "ifog"}
        b = {"i": np.zeros(H), "f": np.full(H, 1.0), "o": np.zeros(H), "g": np.zeros(H)}
        Wy = rng.normal(0, 0.25, (H, N))
        by = np.zeros(N)
        t0 = max(0, T - 1 - trunc)
        scale = 1.0 / max(T - 1 - t0, 1)
        for _ in range(epochs):
            hs = np.zeros((T, H))
            cs = np.zeros((T, H))
            acts = {k: np.zeros((T, H)) for k in "ifog"}
            tanh_c = np.zeros((T, H))
            h, c = np.zeros(H), np.zeros(H)
            for t in range(T):
                z = {k: X[t] @ Wx[k] + h @ Wh[k] + b[k] for k in "ifog"}
                acts["i"][t] = _sigmoid(z["i"])
                acts["f"][t] = _sigmoid(z["f"])
                acts["o"][t] = _sigmoid(z["o"])
                acts["g"][t] = np.tanh(z["g"])
                c = acts["f"][t] * c + acts["i"][t] * acts["g"][t]
                cs[t] = c
                tanh_c[t] = np.tanh(c)
                h = acts["o"][t] * tanh_c[t]
                hs[t] = h
            grads = {f"dWx{k}": np.zeros((N, H)) for k in "ifog"}
            dWh = {k: np.zeros((H, H)) for k in "ifog"}
            db = {k: np.zeros(H) for k in "ifog"}
            dWy, dby = np.zeros((H, N)), np.zeros(N)
            dh_next = np.zeros(H)
            dc_next = np.zeros(H)
            for t in range(T - 2, t0 - 1, -1):
                p = _sigmoid(hs[t] @ Wy + by)
                dp = (p - X[t + 1]) * scale
                dby += dp
                dWy += np.outer(hs[t], dp)
                dh = Wy @ dp + dh_next
                tanh_c_t = tanh_c[t]
                do_a = dh * tanh_c_t
                dc = dh * acts["o"][t] * (1 - tanh_c_t ** 2) + dc_next
                c_prev = cs[t - 1] if t > 0 else np.zeros(H)
                dg_z = dc * (1 - acts["g"][t] ** 2)
                di_z = dc * acts["g"][t] * acts["i"][t] * (1 - acts["i"][t])
                df_z = dc * c_prev * acts["f"][t] * (1 - acts["f"][t])
                do_z = do_a * acts["o"][t] * (1 - acts["o"][t])
                for k, dz in (("i", di_z), ("f", df_z), ("o", do_z), ("g", dg_z)):
                    grads[f"dWx{k}"] += np.outer(X[t], dz)
                    dWh[k] += np.outer(hs[t - 1] if t > 0 else np.zeros(H), dz)
                    db[k] += dz
                dh_next = sum(Wh[k] @ dz for k, dz in (("i", di_z), ("f", df_z),
                                                       ("o", do_z), ("g", dg_z)))
                dc_next = dc * acts["f"][t]
            Wy -= lr * np.clip(dWy, -1, 1)
            by -= lr * np.clip(dby, -1, 1)
            for k in "ifog":
                Wx[k] -= lr * np.clip(grads[f"dWx{k}"], -1, 1)
                Wh[k] -= lr * np.clip(dWh[k], -1, 1)
                b[k] -= lr * np.clip(db[k], -1, 1)
        return _sigmoid(hs[T - 1] @ Wy + by)

    class LstmAlgorithm(Algorithm):
        key = "lstm"
        name = "LSTM 门控循环网络"
        category = "深度学习"
        heavy = True
        desc = "长短期记忆网络：输入门/遗忘门/输出门与细胞状态解决长序列梯度消失（手写截断 BPTT），比朴素 RNN 更能保留远期开奖的记忆。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_lstm_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"LSTM 门控细胞状态读出的 {m.name}高概率号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- Transformer（多头自注意力编码器，手写反向） ----------------

    def _transformer_scores(seq, N, d_model=24, heads=2, epochs=50, lr=0.25, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T = len(X)
        dh = d_model // heads
        We = rng.normal(0, 0.25, (N, d_model))
        Wq, Wk = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model)), rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
        Wv, Wo = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model)), rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_model))
        W1, b1 = rng.normal(0, 0.2, (d_model, 2 * d_model)), np.zeros(2 * d_model)
        W2, b2 = rng.normal(0, 0.2, (2 * d_model, d_model)), np.zeros(d_model)
        Wro, bro = rng.normal(0, 0.25, (d_model, N)), np.zeros(N)
        PE = _pe(T, d_model)
        mask = np.triu(np.ones((T, T)), 1).astype(bool)  # 因果掩码
        for _ in range(epochs):
            E = X @ We + PE[:T]
            Eh = (E - E.mean(1, keepdims=True)) / (E.std(1, keepdims=True) + 1e-5)
            E_std = E.std(1, keepdims=True) + 1e-5
            Q = (Eh @ Wq).reshape(T, heads, dh)
            K = (Eh @ Wk).reshape(T, heads, dh)
            V = (Eh @ Wv).reshape(T, heads, dh)
            S = np.einsum("thq,shq->hts", Q, K) / np.sqrt(dh)
            S = np.where(mask, -1e9, S)
            S -= S.max(axis=2, keepdims=True)
            A = np.exp(S)
            A /= A.sum(axis=2, keepdims=True)
            Att = np.einsum("hts,shq->thq", A, V).reshape(T, d_model)
            Att = Att @ Wo
            H1 = E + Att
            Hn = (H1 - H1.mean(1, keepdims=True)) / (H1.std(1, keepdims=True) + 1e-5)
            Hn_std = H1.std(1, keepdims=True) + 1e-5
            F1 = np.tanh(Hn @ W1 + b1)
            F2 = F1 @ W2 + b2
            H2 = Hn + F2
            P = _sigmoid(H2[: T - 1] @ Wro + bro)
            dP = (P - X[1:]) / max(T - 1, 1)
            dWro = H2[: T - 1].T @ dP
            dbro = dP.sum(0)
            dH2 = np.zeros((T, d_model))
            dH2[: T - 1] = dP @ Wro.T
            dHn = dH2
            dF2 = dHn
            dF1 = dF2 @ W2.T * (1 - F1 ** 2)
            dW2 = F1.T @ dF2
            db2 = dF2.sum(0)
            dHn_ff = dF1 @ W1.T
            dW1 = Hn.T @ dF1
            db1 = dF1.sum(0)
            dH1 = dHn + dHn_ff
            dH1 = _ln_backward(dH1, Hn, Hn_std)
            dE = dH1.copy()
            dAtt_pre = dH1 @ Wo.T
            dWo = Att.T @ dH1
            dO = dAtt_pre.reshape(T, heads, dh)
            dA = np.einsum("thq,shq->hts", dO, V)
            dV = np.einsum("hts,thq->shq", A, dO)
            dS = A * (dA - (dA * A).sum(axis=2, keepdims=True))
            dS = np.where(mask, 0.0, dS / np.sqrt(dh))
            dQ = np.einsum("hts,shq->thq", dS, K)
            dK = np.einsum("hts,thq->shq", dS, Q)
            dEh = dQ.reshape(T, d_model) @ Wq.T + dK.reshape(T, d_model) @ Wk.T \
                + dV.reshape(T, d_model) @ Wv.T
            dWq = Eh.T @ dQ.reshape(T, d_model)
            dWk = Eh.T @ dK.reshape(T, d_model)
            dWv = Eh.T @ dV.reshape(T, d_model)
            dE += _ln_backward(dEh, Eh, E_std)
            dWe = X.T @ dE
            for P_, G in ((We, dWe), (Wq, dWq), (Wk, dWk), (Wv, dWv), (Wo, dWo),
                          (W1, dW1), (b1, db1), (W2, dW2), (b2, db2), (Wro, dWro), (bro, dbro)):
                np.clip(G, -1, 1, out=G)
                P_ -= lr * G
        Eh = (E - E.mean(1, keepdims=True)) / (E.std(1, keepdims=True) + 1e-5)
        Q = (Eh @ Wq).reshape(T, heads, dh)
        K = (Eh @ Wk).reshape(T, heads, dh)
        V = (Eh @ Wv).reshape(T, heads, dh)
        S = np.einsum("thq,shq->hts", Q, K) / np.sqrt(dh)
        S = np.where(mask, -1e9, S)
        S -= S.max(axis=2, keepdims=True)
        A = np.exp(S)
        A /= A.sum(axis=2, keepdims=True)
        Att = (np.einsum("hts,shd->thd", A, V).reshape(T, d_model)) @ Wo
        H1 = E + Att
        Hn = (H1 - H1.mean(1, keepdims=True)) / (H1.std(1, keepdims=True) + 1e-5)
        F1 = np.tanh(Hn @ W1 + b1)
        H2 = Hn + F1 @ W2 + b2
        return _sigmoid(H2[T - 1] @ Wro + bro)

    class TransformerAlgorithm(Algorithm):
        key = "transformer"
        name = "Transformer 编码器"
        category = "深度学习"
        heavy = True
        desc = "完整 Transformer 块：正弦位置编码 + 2 头因果自注意力 + 残差 + LayerNorm + 前馈网络（手写反向传播），逐位置预测下一期开奖。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_transformer_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"多头自注意力编码器（末位隐状态）读出的 {m.name}高概率号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- RBM（受限玻尔兹曼机，CD-1 对比散度） ----------------

    def _rbm_scores(seq, N, hidden=24, epochs=40, lr=0.08, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        W = rng.normal(0, 0.15, (N, hidden))
        a, b = np.zeros(N), np.zeros(hidden)
        T = len(X)
        for _ in range(epochs):
            h_pos = _sigmoid(X @ W + b)
            v_neg = _sigmoid(h_pos @ W.T + a)
            h_neg = _sigmoid(v_neg @ W + b)
            W += lr * ((X.T @ h_pos - v_neg.T @ h_neg) / T)
            a += lr * (X - v_neg).mean(axis=0)
            b += lr * (h_pos - h_neg).mean(axis=0)
        h = _sigmoid(X[-1] @ W + b)
        return _sigmoid(h @ W.T + a)

    class RbmAlgorithm(Algorithm):
        key = "rbm"
        name = "受限玻尔兹曼机 RBM"
        category = "深度学习"
        heavy = True
        desc = "可见层=号码、隐藏层=模式的能量模型，用对比散度（CD-1）逐期学习开奖向量的联合分布；以最新一期为条件，读出各号码的激活概率。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_rbm_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"RBM 条件激活概率最高的 {m.name}号码：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- ESN（储备池计算，脊回归读出） ----------------

    def _esn_scores(seq, N, hidden=50, rho=0.9, lam=1e-2, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T = len(X)
        W_in = rng.normal(0, 0.6 / np.sqrt(N), (N, hidden))
        W_res = rng.normal(0, 1, (hidden, hidden)) / np.sqrt(hidden)
        ev = max(abs(np.linalg.eigvals(W_res)).real)
        if ev > 0:
            W_res *= rho / ev
        states = np.zeros((T, hidden))
        s = np.zeros(hidden)
        for t in range(T):
            s = np.tanh(X[t] @ W_in + s @ W_res)
            states[t] = s
        S = np.hstack([np.ones((T - 1, 1)), states[: T - 1]])
        Y = X[1:]
        A = S.T @ S + lam * np.eye(hidden + 1) * np.trace(S.T @ S) / (hidden + 1)
        Wr = np.linalg.solve(A, S.T @ Y)          # (hidden+1, N)
        return _sigmoid(np.hstack([1.0, states[T - 1]]) @ Wr)

    class EsnAlgorithm(Algorithm):
        key = "esn"
        name = "储备池计算 ESN"
        category = "深度学习"
        heavy = True
        desc = "Echo State Network：固定随机递归储备池（谱半径 0.9）把开奖序列非线性映射到高维状态空间，只需脊回归训练线性读出——无反向传播，秒级训练。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_esn_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"储备池状态 + 脊回归读出的 {m.name}高概率号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- Skip-Gram 号码嵌入 ----------------

    def _skipgram_scores(seq, N, dim=10, neg=2, epochs=3, lr=0.05, seed=1, cap=6000):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T = len(X)
        pairs = []
        for t in range(T):
            combo = [n - 1 for n in seq[t] if 1 <= n <= N]
            for c in combo:
                for o in combo:
                    if c != o:
                        pairs.append((c, o))
        if len(pairs) > cap:
            idx = rng.choice(len(pairs), cap, replace=False)
            pairs = [pairs[i] for i in idx]
        if not pairs:
            return X.mean(axis=0)
        C_in = rng.normal(0, 0.1, (N, dim))
        C_out = rng.normal(0, 0.1, (N, dim))
        for _ in range(epochs):
            rng.shuffle(pairs)
            for c, o in pairs:
                v = C_in[c]
                negs = rng.integers(0, N, neg)
                negs = [n for n in negs if n != o][:neg]
                # 正样本 + 负样本的逻辑损失梯度
                score_pos = _sigmoid(C_out[o] @ v)
                g_out_pos = (score_pos - 1) * lr
                C_out[o] -= g_out_pos * v
                g_v = g_out_pos * C_out[o]
                for nk in negs:
                    sc = _sigmoid(C_out[nk] @ v)
                    g_out_neg = sc * lr
                    C_out[nk] -= g_out_neg * v
                    g_v += g_out_neg * C_out[nk]
                C_in[c] -= g_v
        recent = np.zeros(N)
        for t in range(max(0, T - 3), T):
            for n in seq[t]:
                if 1 <= n <= N:
                    recent[n - 1] = 1
        if recent.sum() == 0:
            recent = X[-1]
        centroid = (C_in * recent[:, None]).sum(axis=0) / max(recent.sum(), 1)
        cn = np.linalg.norm(C_in, axis=1) + 1e-9
        cnorm = np.linalg.norm(centroid) + 1e-9
        return (C_in @ centroid) / (cn * cnorm)

    class SkipGramAlgorithm(Algorithm):
        key = "skipgram"
        name = "Skip-Gram 号码嵌入"
        category = "深度学习"
        heavy = True
        desc = "把每期组合当作句子、号码当作词，负采样 Skip-Gram（Word2Vec 思想）学出每个号码的低维向量；与最近三期号码的语义中心越接近，评分越高——「物以类聚」的向量版。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_skipgram_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"嵌入空间中与近期号码最邻近的 {m.name}号码：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- 相似期检索 kNN ----------------

    def _knncase_scores(seq, N, ref=10, k=15, seed=1):
        X, cum = _cumulative(seq)
        T = len(X)
        ref = max(3, min(int(ref), T - 2))
        profiles = (X[ref:] - X[: T - ref]) / ref          # (T-ref) 个「近 ref 期频率画像」
        targets = X[ref:]                                   # targets[i] = 画像 i 的下一期
        cur = (cum[T] - cum[T - ref]) / ref                 # 当前画像（最近 ref 期）
        pn = np.linalg.norm(profiles, axis=1) + 1e-9
        cn = np.linalg.norm(cur) + 1e-9
        sims = profiles @ cur / (pn * cn)
        k = max(1, min(int(k), len(sims)))
        top = np.argsort(-sims)[:k]
        w = np.maximum(sims[top], 0) ** 2 + 1e-6
        return (targets[top] * w[:, None]).sum(axis=0) / w.sum()

    class KnnCaseAlgorithm(Algorithm):
        key = "knncase"
        name = "相似期检索 kNN"
        category = "机器学习"
        heavy = True
        desc = "基于实例的推理：把「近 N 期频率画像」当作指纹，在全部历史里检索与当前指纹最相似的 K 个时期，用它们下一期实际开出的号码按相似度加权投票——案例推理（CBR）思想。"

        params = [{"key": "ref", "label": "画像窗口", "default": 10, "min": 3, "max": 30, "step": 1},
                  {"key": "k", "label": "近邻数 K", "default": 15, "min": 5, "max": 60, "step": 5}]

        def predict(self, draws, game, params):
            ref = int(params.get("ref", 10))
            k = int(params.get("k", 15))

            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_knncase_scores(seq, N, ref=ref, k=k, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"与当前画像最相似的 K={params.get('k', 15)} 个历史案例加权投票的 {m.name}高频号："
                    f"{top_text(scores[m.key], m.count)}。{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- K-Means 聚类原型 ----------------

    def _kmeans_scores(seq, N, k=8, iters=15, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T = len(X)
        k = max(2, min(int(k), T - 2))
        Fn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        centroids = Fn[rng.choice(T, k, replace=False)].copy()
        assign = np.zeros(T, dtype=int)
        for _ in range(iters):
            dist = 1 - Fn @ centroids.T                      # 余弦距离
            new_assign = dist.argmin(axis=1)
            if (new_assign == assign).all():
                break
            assign = new_assign
            for c in range(k):
                members = Fn[assign == c]
                if len(members):
                    centroids[c] = members.mean(axis=0)
                    centroids[c] /= (np.linalg.norm(centroids[c]) + 1e-9)
        # 每个簇的「下一期画像」
        next_profile = np.zeros((k, N))
        cnt = np.zeros(k)
        for t in range(T - 1):
            next_profile[assign[t]] += X[t + 1]
            cnt[assign[t]] += 1
        cur = assign[T - 1]
        global_mean = X.mean(axis=0)
        if cnt[cur] > 0:
            proto = next_profile[cur] / cnt[cur]
        else:
            proto = global_mean
        w = min(cnt[cur] / max(T / k, 1), 1.0) * 0.5
        return proto * (0.5 + w) + global_mean * (0.5 - w)

    class KMeansAlgorithm(Algorithm):
        key = "kmeans"
        name = "聚类原型 K-Means"
        category = "机器学习"
        heavy = True
        desc = "把历史每期开奖向量聚成 K 个「形态原型」（余弦 K-Means++式），判定当前期属于哪个原型，用该原型之后一期的经验分布作为评分——形态.matching 思路。"

        params = [{"key": "k", "label": "原型数 K", "default": 8, "min": 3, "max": 20, "step": 1}]

        def predict(self, draws, game, params):
            k = int(params.get("k", 8))

            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_kmeans_scores(seq, N, k=k, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"当前形态所属原型（K={params.get('k', 8)}）下一期经验分布的 {m.name}高频号："
                    f"{top_text(scores[m.key], m.count)}。{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- BNN（贝叶斯神经网络，Bayes by Backprop） ----------------

    def _bnn_scores(seq, N, hidden=16, epochs=35, lr=0.08, mc=20, prior_std=1.0, beta=5e-4, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T, H = len(X), hidden
        shapes = {"W1": (N, H), "b1": (H,), "W2": (H, N), "b2": (N,)}
        mu = {"W1": rng.normal(0, 0.25, shapes["W1"]), "b1": np.zeros(shapes["b1"]),
              "W2": rng.normal(0, 0.25, shapes["W2"]), "b2": np.zeros(shapes["b2"])}
        rho = {k: np.full(shapes[k], -2.0) for k in shapes}   # σ = softplus(ρ) ≈ 0.13
        n_w = sum(np.size(v) for k, v in shapes.items())
        prior2 = prior_std ** 2

        def softplus(v):
            return np.log1p(np.exp(v))

        for _ in range(epochs):
            eps_w = {k: rng.normal(0, 1, shapes[k]) for k in shapes}
            W1 = mu["W1"] + softplus(rho["W1"]) * eps_w["W1"]
            b1 = mu["b1"] + softplus(rho["b1"]) * eps_w["b1"]
            W2 = mu["W2"] + softplus(rho["W2"]) * eps_w["W2"]
            b2 = mu["b2"] + softplus(rho["b2"]) * eps_w["b2"]
            Hn = np.maximum(X @ W1 + b1, 0)
            P = _sigmoid(Hn @ W2 + b2)
            dP = (P - X) / T                                  # BCE（按时间平均）
            dW2 = Hn.T @ dP
            db2 = dP.sum(0)
            dH = dP @ W2.T * (Hn > 0)
            dW1 = X.T @ dH
            db1 = dH.sum(0)
            dw = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
            for k in shapes:
                sigma = softplus(rho[k])
                # 重参数化：dμ += dL/dw；dρ += dL/dw · ε · sigmoid(ρ)；加 KL(N(μ,σ)||N(0,prior)) 项
                dmu = dw[k] + beta * mu[k] / prior2 / n_w
                drho = dw[k] * eps_w[k] * _sigmoid(rho[k]) \
                    + beta * (sigma / prior2 - 1.0 / sigma) * _sigmoid(rho[k]) / n_w
                mu[k] -= lr * np.clip(dmu, -2, 2)
                rho[k] -= lr * np.clip(drho, -2, 2)
        # 蒙特卡洛预测：对后验采样 mc 次，取预测均值（后验预测分布）
        outs = []
        for _ in range(mc):
            W1 = mu["W1"] + softplus(rho["W1"]) * rng.normal(0, 1, shapes["W1"])
            b1 = mu["b1"] + softplus(rho["b1"]) * rng.normal(0, 1, shapes["b1"])
            W2 = mu["W2"] + softplus(rho["W2"]) * rng.normal(0, 1, shapes["W2"])
            b2 = mu["b2"] + softplus(rho["b2"]) * rng.normal(0, 1, shapes["b2"])
            Hn = np.maximum(X[T - 1] @ W1 + b1, 0)
            outs.append(_sigmoid(Hn @ W2 + b2))
        return np.mean(outs, axis=0)

    class BnnAlgorithm(Algorithm):
        key = "bnn"
        name = "贝叶斯神经网络 BNN"
        category = "深度学习"
        heavy = True
        desc = "Bayes by Backprop：权重不是固定值而是「均值+方差」的变分高斯分布，训练目标为 ELBO（重构 + KL 到先验）；预测时对权重采样 20 次取均值——显式量化模型不确定性，适合小数据场景。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_bnn_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"贝叶斯后验预测分布（20 次权重采样均值）最高的 {m.name}号码：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- RSSM（循环状态空间模型，Dreamer 世界模型极简版） ----------------

    def _rssm_scores(seq, N, hidden=16, zdim=6, epochs=20, lr=0.06, beta=0.05, mc=8, seed=1):
        rng = np.random.default_rng(seed)
        X, _ = _cumulative(seq)
        T, H, Z = len(X), hidden, zdim
        W_hh = rng.normal(0, 1 / np.sqrt(H), (H, H))
        W_hz = rng.normal(0, 1 / np.sqrt(Z), (H, Z))
        W_ho = rng.normal(0, 1 / np.sqrt(N), (H, N))
        b_h = np.zeros(H)
        W_mq, W_sq = rng.normal(0, 0.2, (Z, H)), rng.normal(0, 0.2, (Z, H))
        b_mq, b_sq = np.zeros(Z), np.full(Z, -2.0)
        W_mp, W_sp = rng.normal(0, 0.2, (Z, H)), rng.normal(0, 0.2, (Z, H))
        W_d = rng.normal(0, 0.2, (H + Z, N))
        b_d = np.zeros(N)

        def run(h0, z0, obs, sample=True):
            """前向整条序列：返回逐期状态/后验/先验/重建。obs 为观测序列"""
            Tn = len(obs)
            h = h0.copy()
            z = z0.copy()
            hs = np.zeros((Tn, H))
            zs = np.zeros((Tn, Z))
            eps_z = np.zeros((Tn, Z))
            mu_q = np.zeros((Tn, Z))
            sg_q = np.zeros((Tn, Z))
            mu_p = np.zeros((Tn, Z))
            sg_p = np.zeros((Tn, Z))
            ps = np.zeros((Tn, N))
            for t in range(Tn):
                h = np.tanh(W_hh @ h + W_hz @ z + (W_ho @ obs[t - 1] if t > 0 else 0.0) + b_h)
                hs[t] = h
                mq = W_mq @ h + b_mq
                sq = np.exp(0.5 * np.clip(W_sq @ h + b_sq, -8, 8))
                mu_q[t], sg_q[t] = mq, sq
                if t > 0:
                    mu_p[t] = W_mp @ hs[t - 1]
                    sg_p[t] = np.exp(0.5 * np.clip(W_sp @ hs[t - 1], -8, 8))
                eps = rng.normal(0, 1, Z) if sample else np.zeros(Z)
                eps_z[t] = eps
                z = mq + sq * eps
                zs[t] = z
                ps[t] = _sigmoid(np.hstack([h, z]) @ W_d + b_d)
            return hs, zs, eps_z, mu_q, sg_q, mu_p, sg_p, ps

        for _ in range(epochs):
            hs, zs, eps_z, mu_q, sg_q, mu_p, sg_p, ps = run(np.zeros(H), np.zeros(Z), X)
            dW_hh = np.zeros((H, H))
            dW_hz = np.zeros((H, Z))
            dW_ho = np.zeros((H, N))
            db_h = np.zeros(H)
            dW_mq = np.zeros((Z, H))
            dW_sq = np.zeros((Z, H))
            db_mq = np.zeros(Z)
            db_sq = np.zeros(Z)
            dW_mp = np.zeros((Z, H))
            dW_sp = np.zeros((Z, H))
            dW_d = np.zeros((H + Z, N))
            db_d = np.zeros(N)
            dh_next = np.zeros(H)
            dz_next = np.zeros(Z)
            for t in range(T - 1, -1, -1):
                dP = (ps[t] - X[t]) / T
                hz_vec = np.hstack([hs[t], zs[t]])
                dW_d += np.outer(hz_vec, dP)
                db_d += dP
                dh = W_d[:H] @ dP + dh_next
                dz = W_d[H:] @ dP + dz_next
                if t > 0:  # KL(q(z_t|h_t) || p(z_t|h_{t-1}))，对 μ/σ 的闭式梯度
                    dmu_q_t = dz + beta * (mu_q[t] - mu_p[t]) / (sg_p[t] ** 2 * T)
                    dsg_q_t = dz * sg_q[t] * eps_z[t] * 0.5 \
                        + beta * (sg_q[t] / (sg_p[t] ** 2) - 1.0 / sg_q[t]) * sg_q[t] * 0.5 / T
                    dmu_p_t = -beta * (mu_q[t] - mu_p[t]) / (sg_p[t] ** 2 * T)
                    dsg_p_t = -beta * (1.0 / sg_p[t] - sg_q[t] ** 2 / (sg_p[t] ** 3)) * sg_p[t] * 0.5 / T
                    dW_mp += np.outer(dmu_p_t, hs[t - 1])
                    dW_sp += np.outer(dsg_p_t, hs[t - 1])
                    dh_prev_prior = W_mp.T @ dmu_p_t + W_sp.T @ dsg_p_t
                else:
                    dmu_q_t = dz
                    dsg_q_t = dz * sg_q[t] * eps_z[t] * 0.5
                    dh_prev_prior = np.zeros(H)
                dh += W_mq.T @ dmu_q_t + W_sq.T @ dsg_q_t
                dW_mq += np.outer(dmu_q_t, hs[t])
                dW_sq += np.outer(dsg_q_t, hs[t])
                db_mq += dmu_q_t
                db_sq += dsg_q_t
                dh_pre = dh * (1 - hs[t] ** 2)
                dW_hh += np.outer(dh_pre, hs[t - 1] if t > 0 else np.zeros(H))
                dW_hz += np.outer(dh_pre, zs[t - 1] if t > 0 else np.zeros(Z))
                dW_ho += np.outer(dh_pre, X[t - 1] if t > 0 else np.zeros(N))
                db_h += dh_pre
                dh_next = W_hh.T @ dh_pre
                dz_next = W_hz.T @ dh_pre
                if t > 0:
                    dh_next += dh_prev_prior
            for P_, G in ((W_hh, dW_hh), (W_hz, dW_hz), (W_ho, dW_ho), (b_h, db_h),
                          (W_mq, dW_mq), (W_sq, dW_sq), (b_mq, db_mq), (b_sq, db_sq),
                          (W_mp, dW_mp), (W_sp, dW_sp), (W_d, dW_d), (b_d, db_d)):
                np.clip(G, -1, 1, out=G)
                P_ -= lr * G
        # 「想象」下一期：先验采样 z_T → 递推 h_T → 解码，MC 平均
        hs, zs, _, _, _, _, sg_p_last_mu, _ = run(np.zeros(H), np.zeros(Z), X)
        h_last = hs[T - 1]
        z_last = zs[T - 1]
        outs = []
        for _ in range(mc):
            mu_p = W_mp @ h_last
            sg_p = np.exp(0.5 * np.clip(W_sp @ h_last, -8, 8))
            z_next = mu_p + sg_p * rng.normal(0, 1, Z)
            h_next = np.tanh(W_hh @ h_last + W_hz @ z_last + W_ho @ X[T - 1] + b_h)
            outs.append(_sigmoid(np.hstack([h_next, z_next]) @ W_d + b_d))
        return np.mean(outs, axis=0)

    class RssmAlgorithm(Algorithm):
        key = "rssm"
        name = "循环状态空间 RSSM"
        category = "深度学习"
        heavy = True
        desc = "Dreamer 世界模型的极简版：确定性状态 h 与随机状态 z 双链演化，后验 q(z|h) 与先验 p(z|h_{t-1}) 的 KL + 重建构成 ELBO；预测时在先验上采样「想象」下一期分布——处理时序不确定性的生成式思路。"

        def predict(self, draws, game, params):
            def score(d, p):
                seq = indicator(d, p)
                N = p.max_number
                return stats.normalize(_rssm_scores(seq, N, seed=_seed(d, p)))

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"状态空间「想象」（8 次先验采样均值）最高的 {m.name}号码：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    ALGORITHMS = [LstmAlgorithm(), TransformerAlgorithm(), RbmAlgorithm(), EsnAlgorithm(),
                  SkipGramAlgorithm(), KnnCaseAlgorithm(), KMeansAlgorithm(),
                  BnnAlgorithm(), RssmAlgorithm()]
else:
    ALGORITHMS = []
