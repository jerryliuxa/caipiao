"""sklearn 监督模型：逻辑回归 / 梯度提升树 / 随机森林 / MLP + 时序分解全家桶
需要 numpy 与 scikit-learn；缺失时本模块注册为空，平台其余功能不受影响。
"""
from .base import Algorithm, indicator, pool_predict, top_text
from .. import stats

import warnings

try:
    import numpy as np
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", message=".*Convergence.*")
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False

if AVAILABLE:
    from .base import build_supervised

    def _classifier_scores(model_cls, kwargs, draws, pool, seed_note=""):
        X, y, X_next, N = build_supervised(draws, pool)
        if len(X) < 50:
            freq = stats.frequency(draws, pool)
            return stats.normalize([c + 1.0 for c in freq])
        model = model_cls(**kwargs)
        model.fit(X, y)
        proba = model.predict_proba(X_next)[:, 1]
        return stats.normalize(proba.tolist())

    class LogisticAlgorithm(Algorithm):
        key = "logistic"
        name = "逻辑回归"
        category = "机器学习"
        heavy = True
        desc = "特征工程（窗口频率、衰减热度、遗漏、比值、位置、区间、质数等 14+N 维）+ L2 正则逻辑回归，逐(期,号码)训练下期出现概率。"
        params = [{"key": "C", "label": "正则化倒数 C", "default": 0.5, "min": 0.05, "max": 2.0, "step": 0.05}]

        def predict(self, draws, game, params):
            C = float(params.get("C", 0.5))

            def score(d, p):
                return _classifier_scores(LogisticRegression,
                                          {"C": C, "max_iter": 400}, d, p)

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"L2 逻辑回归（C={params.get('C', 0.5)}，特征 14+{m.max_number} 维）预测的 {m.name}出现概率最高："
                    f"{top_text(scores[m.key], m.count)}。{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    class GbdtAlgorithm(Algorithm):
        key = "gbdt"
        name = "梯度提升树 GBDT"
        category = "机器学习"
        heavy = True
        desc = "GradientBoosting 拟合特征到「下期是否出现」的非线性关系，表格数据上最主流的机器学习模型；能自动学习特征交互。"
        params = [{"key": "n_est", "label": "树数量", "default": 60, "min": 30, "max": 160, "step": 20}]

        def predict(self, draws, game, params):
            n_est = int(params.get("n_est", 60))

            def score(d, p):
                return _classifier_scores(GradientBoostingClassifier,
                                          {"n_estimators": n_est, "max_depth": 3,
                                           "learning_rate": 0.08, "random_state": 7}, d, p)

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"GBDT（{params.get('n_est', 60)} 棵深度 3 的树）预测的 {m.name}高概率号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    class RandomForestAlgorithm(Algorithm):
        key = "rf"
        name = "随机森林"
        category = "机器学习"
        heavy = True
        desc = "Bagging 思想的决策森林：多棵树在随机特征子集上投票，输出平均出现概率，比单棵树稳健、比 GBDT 不易过拟合。"

        def predict(self, draws, game, params):
            def score(d, p):
                return _classifier_scores(RandomForestClassifier,
                                          {"n_estimators": 120, "max_depth": 6,
                                           "random_state": 7, "n_jobs": -1}, d, p)

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"120 棵深度 6 的决策树投票平均的 {m.name}出现概率最高：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    class MlpAlgorithm(Algorithm):
        key = "mlp"
        name = "神经网络 MLP"
        category = "机器学习"
        heavy = True
        desc = "单隐层(24 神经元)多层感知机，学习特征的非线性组合；随机梯度 + Adam，早停防过拟合。"
        params = [{"key": "hidden", "label": "隐层神经元", "default": 24, "min": 8, "max": 64, "step": 8}]

        def predict(self, draws, game, params):
            hidden = int(params.get("hidden", 24))

            def score(d, p):
                return _classifier_scores(MLPClassifier,
                                          {"hidden_layer_sizes": (hidden,), "max_iter": 130,
                                           "random_state": 7}, d, p)

            return pool_predict(score, draws, game)

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"MLP（隐层 {params.get('hidden', 24)} 神经元）输出的 {m.name}高概率号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    # ---------------- 时序分解全家桶（和值预测 → 条件频率） ----------------

    def _arima_next(sums):
        """AR(p) 最小二乘 + 残差 AR(1)（MA 代理），AIC 选阶"""
        y = np.asarray(sums, dtype=float)
        best = (None, float("inf"), 0.0)
        for p_ord in (1, 2, 3):
            if len(y) <= p_ord + 8:
                continue
            M = np.column_stack([np.ones(len(y) - p_ord)] + [y[p_ord - k - 1: len(y) - k - 1] for k in range(p_ord)])
            target = y[p_ord:]
            coef, *_ = np.linalg.lstsq(M, target, rcond=None)
            resid = target - M @ coef
            aic = len(resid) * np.log(max(resid.var(), 1e-9)) + 2 * (p_ord + 1)
            if aic < best[1]:
                best = (coef, aic, p_ord)
        coef, _, p_ord = best
        if coef is None:
            return float(np.mean(y))
        lags = np.array([1.0] + [y[-1 - k] for k in range(p_ord)])
        return float(lags @ coef)

    def _holt_next(sums):
        """Holt 双参数线性趋势平滑，网格选 α/β"""
        y = np.asarray(sums, dtype=float)
        best = (float("inf"), 0.3, 0.1)
        for alpha in (0.2, 0.4, 0.6, 0.8):
            for beta in (0.05, 0.1, 0.3):
                level, trend = y[0], y[1] - y[0]
                sse = 0.0
                for v in y[1:]:
                    fit = level + trend
                    sse += (v - fit) ** 2
                    prev = level
                    level = alpha * v + (1 - alpha) * fit
                    trend = beta * (level - prev) + (1 - beta) * trend
                if sse < best[0]:
                    best = (sse, alpha, beta)
        _, alpha, beta = best
        level, trend = y[0], y[1] - y[0]
        for v in y[1:]:
            fit = level + trend
            prev = level
            level = alpha * v + (1 - alpha) * fit
            trend = beta * (level - prev) + (1 - beta) * trend
        return float(level + trend)

    def _wavelet_next(sums):
        """Haar 小波 3 层分解 → 细节去噪 → 重构 → 末段斜率外推"""
        x = np.asarray(sums, dtype=float)
        cur = x.copy()
        detail = np.zeros_like(x)
        for _ in range(3):
            if len(cur) < 4:
                break
            half = len(cur) // 2
            avg = (cur[0:2 * half:2] + cur[1:2 * half:2]) / 2
            dif = (cur[1:2 * half:2] - cur[0:2 * half:2]) / 2
            detail[:half] += dif  # 记录细节（最细层优先）
            cur = avg
        # 去掉最细两层细节（软阈值），只留趋势
        smoothed = cur
        slope = smoothed[-1] - smoothed[-3] if len(smoothed) >= 3 else 0.0
        return float(smoothed[-1] + 0.5 * slope)

    def _ssa_next(sums):
        """奇异谱分析：轨迹矩阵 SVD，取前 3 分量重构并线性外推"""
        y = np.asarray(sums, dtype=float)
        L = max(5, min(30, len(y) // 3))
        K = len(y) - L + 1
        traj = np.column_stack([y[i:i + L] for i in range(K)])
        u, s, vt = np.linalg.svd(traj, full_matrices=False)
        rank = 3
        rec = (u[:, :rank] * s[:rank]) @ vt[:rank]
        series = np.array([rec.diagonal(offset).mean()
                           for offset in range(-(rec.shape[0] - 1), rec.shape[1])])
        slope = series[-1] - series[-4] if len(series) >= 4 else 0.0
        return float(series[-1] + slope / 3.0)

    def _prophet_next(sums, dates, next_weekday):
        """Prophet 式：二次趋势 + 开奖星期季节项（最小二乘），外推一期"""
        T = len(sums)
        X_rows, y = [], []
        for t, (v, iso) in enumerate(zip(sums, dates)):
            wd = _weekday(iso)
            dummies = [1.0 if wd == d else 0.0 for d in range(7)]
            X_rows.append([1.0, t / max(T, 1), (t / max(T, 1)) ** 2] + dummies)
            y.append(v)
        X = np.array(X_rows)
        coef, *_ = np.linalg.lstsq(X, np.asarray(y), rcond=None)
        tt = T / max(T, 1)
        dummies = [1.0 if next_weekday == d else 0.0 for d in range(7)]
        return float(np.array([1.0, tt, tt * tt] + dummies) @ coef)

    def _garch_sigma(sums):
        """GARCH(1,1) 小网格拟合差分序列，预测下期波动率"""
        r = np.diff(np.asarray(sums, dtype=float))
        if len(r) < 20:
            return float(np.std(sums))
        base_var = float(r.var())
        best = (float("-inf"), base_var ** 0.5)
        for alpha in (0.05, 0.1, 0.2):
            for beta in (0.6, 0.8, 0.9):
                omega = base_var * (1 - alpha - beta)
                var = base_var
                ll = 0.0
                for x in r:
                    ll += -0.5 * (np.log(max(var, 1e-9)) + x * x / max(var, 1e-9))
                    var = omega + alpha * x * x + beta * var
                if ll > best[0]:
                    best = (ll, var ** 0.5)
        return best[1]

    class TimeSeriesAlgorithm(Algorithm):
        key = "timeseries"
        name = "时序分解全家桶"
        category = "时序预测"
        heavy = True
        desc = "对主池和值序列同时拟合 ARIMA(小阶网格)、Holt-Winters、Haar 小波、SSA 奇异谱、Prophet 式趋势+星期季节五种均值模型与 GARCH(1,1) 波动率，取中位数预测下期和值，再统计历史上「和值落在预测区间」时各号码的条件出现频率。"

        def predict(self, draws, game, params):
            main = game.main_pool
            sums = [sum(d["numbers"][main.key]) for d in draws]
            dates = [d.get("date", "") for d in draws]
            next_wd = _next_weekday(draws, game)

            preds = {
                "ARIMA": _arima_next(sums),
                "Holt-Winters": _holt_next(sums),
                "Haar小波": _wavelet_next(sums),
                "SSA": _ssa_next(sums),
                "趋势+季节": _prophet_next(sums, dates, next_wd),
            }
            mu = float(np.median(list(preds.values())))
            sigma = max(_garch_sigma(sums), float(np.std(sums)))
            band = (mu - 1.5 * sigma, mu + 1.5 * sigma)

            matched = [d for d in draws if band[0] <= sum(d["numbers"][main.key]) <= band[1]]
            ref = matched if len(matched) >= 15 else draws
            scores = {main.key: stats.normalize(stats.weighted_frequency(ref, main, 0.97))}
            scores[game.sub_pool.key] = stats.normalize(stats.weighted_frequency(draws, game.sub_pool, 0.97))
            self.generation_note = (f"子模型和值预测：" + "、".join(f"{k} {v:.0f}" for k, v in preds.items())
                                    + f"；GARCH 波动率 {sigma:.1f}；预测区间 [{band[0]:.0f}, {band[1]:.0f}]"
                                    + (f"（历史匹配 {len(matched)} 期）" if ref is matched else "（匹配不足，退化为全样本）"))
            return scores

        def summarize(self, draws, game, params, scores):
            m = game.main_pool
            return (f"和值时序预测与条件频率（明细见下方说明）给出的 {m.name}高分号：{top_text(scores[m.key], m.count)}。"
                    f"{game.sub_pool.name}首选：{top_text(scores[game.sub_pool.key], 1)}。")

    def _weekday(iso):
        import datetime
        try:
            return datetime.date.fromisoformat(iso).weekday()
        except (ValueError, TypeError):
            return -1

    def _next_weekday(draws, game):
        """按开奖星期推算下一期的星期（用于季节项外推）"""
        import datetime
        last = draws[-1].get("date", "")
        try:
            d = datetime.date.fromisoformat(last)
        except (ValueError, TypeError):
            return game.draw_weekdays[0]
        for i in range(1, 15):
            nd = d + datetime.timedelta(days=i)
            if nd.weekday() in game.draw_weekdays:
                return nd.weekday()
        return game.draw_weekdays[0]

    ALGORITHMS = [LogisticAlgorithm(), GbdtAlgorithm(), RandomForestAlgorithm(),
                  MlpAlgorithm(), TimeSeriesAlgorithm()]
else:
    ALGORITHMS = []
