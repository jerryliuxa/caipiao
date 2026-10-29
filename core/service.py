"""预测与回测编排：评分 → 约束抽样生成号码组合 → 回测验证"""
import hashlib
import json
import random

from . import filters, stats
from .algorithms import get_algorithm
from .algorithms.base import weighted_sample
from .games import get_game
from . import storage


def _make_rng(game_key, algo_key, window, nonce, last_issue):
    seed_src = f"{game_key}|{algo_key}|{window}|{nonce or 0}|{last_issue}"
    seed = int(hashlib.md5(seed_src.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


def _generate_main_sets(game, scores, combo_filter, n_sets, rng):
    """主池组合：首注取评分 Top-K，其余按评分加权抽样并逐级放宽形态约束"""
    k = game.main_pool.count
    ranked = sorted(game.main_pool.numbers, key=lambda n: -scores[n - 1])
    top = ranked[: k * 3]
    weights = [max(scores[n - 1], 1e-6) ** 3 for n in top]

    sets, seen = [], set()
    greedy = tuple(sorted(ranked[:k]))
    for level in (2, 1):
        if combo_filter.ok(greedy, level):
            sets.append(list(greedy))
            seen.add(greedy)
            break
    attempts = 0
    for level, budget in ((2, 4000), (1, 3000)):
        while len(sets) < n_sets and attempts < budget:
            attempts += 1
            cand = tuple(weighted_sample(top, weights, k, rng))
            if cand in seen or not combo_filter.ok(cand, level):
                continue
            seen.add(cand)
            sets.append(list(cand))
        if len(sets) >= n_sets:
            break
    while len(sets) < n_sets:  # 兜底：完全放宽
        cand = tuple(weighted_sample(top, weights, k, rng))
        if cand not in seen:
            seen.add(cand)
            sets.append(list(cand))
    return sets[:n_sets]


def _generate_sub_sets(game, scores, n_sets, rng):
    """副池组合：单号轮换 Top-3；双号抽样 Top-6"""
    sub = game.sub_pool
    ranked = sorted(sub.numbers, key=lambda n: -scores[n - 1])
    if sub.count == 1:
        top = ranked[:3]
        return [[n] for n in (top[i % len(top)] for i in range(n_sets))]
    top = ranked[:6]
    weights = [max(scores[n - 1], 1e-6) ** 3 for n in top]
    pairs, seen = [], set()
    first = tuple(sorted(top[:2]))
    pairs.append(list(first))
    seen.add(first)
    attempts = 0
    while len(pairs) < n_sets and attempts < 3000:
        attempts += 1
        p = tuple(weighted_sample(top, weights, 2, rng))
        if p in seen:
            continue
        seen.add(p)
        pairs.append(list(p))
    while len(pairs) < n_sets:
        pairs.append(list(first))
    return pairs[:n_sets]


_backtest_cache = {}


def _core_predict(draws, game, algo, params, n_sets, rng, cache_key=None):
    if cache_key:
        hit = _backtest_cache.get(cache_key)
        if hit is not None:
            return hit
    scores = algo.predict(draws, game, params)
    combo_filter = filters.ComboFilter(draws, game)
    # 自定义出票的算法直接拿到主池评分列表
    custom = algo.generate_sets(draws, game, params, scores[game.main_pool.key], combo_filter, n_sets, rng)
    if custom is not None:
        main_sets, sub_sets = custom
        if sub_sets is None:
            sub_sets = _generate_sub_sets(game, scores[game.sub_pool.key], len(main_sets), rng)
    else:
        main_sets = _generate_main_sets(game, scores[game.main_pool.key], combo_filter, n_sets, rng)
        sub_sets = _generate_sub_sets(game, scores[game.sub_pool.key], n_sets, rng)
    result = (scores, main_sets, sub_sets)
    if cache_key:
        if len(_backtest_cache) > 800:
            _backtest_cache.clear()
        _backtest_cache[cache_key] = result
    return result


def _number_detail(pool, nums, scores, freq, omission_list):
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    out = []
    for n in sorted(nums):
        heat = round((scores[n - 1] - lo) / span * 100)
        out.append({"n": n, "score": round(scores[n - 1], 4), "heat": heat,
                    "freq": freq[n - 1], "omission": omission_list[n - 1]})
    return out


def predict(game_key, algo_key, window=100, n_sets=5, nonce=0, algo_params=None):
    game = get_game(game_key)
    algo = get_algorithm(algo_key)
    algo_params = algo_params or {}
    draws_all = storage.get_draws(game_key)
    if len(draws_all) < 20:
        raise ValueError("历史数据不足（少于 20 期），请先点击“更新数据”")
    draws = draws_all[-int(window):]
    rng = _make_rng(game_key, algo_key, window, nonce, draws[-1]["issue"])
    algo.generation_note = ""
    scores, main_sets, sub_sets = _core_predict(draws, game, algo, algo_params, n_sets, rng)

    n = len(main_sets)
    freq = {p.key: stats.frequency(draws, p) for p in game.pools}
    omiss = {p.key: stats.omission(draws, p) for p in game.pools}

    tickets = []
    for i in range(n):
        numbers = {game.main_pool.key: sorted(main_sets[i]), game.sub_pool.key: sorted(sub_sets[i])}
        detail, heat_sum, heat_cnt = {}, 0, 0
        for pool in game.pools:
            d = _number_detail(pool, numbers[pool.key], scores[pool.key], freq[pool.key], omiss[pool.key])
            detail[pool.key] = d
            heat_sum += sum(x["heat"] for x in d)
            heat_cnt += len(d)
        tickets.append({"numbers": numbers, "score": round(heat_sum / max(heat_cnt, 1), 1), "detail": detail})

    ranking = {}
    for pool in game.pools:
        s = scores[pool.key]
        order = sorted(pool.numbers, key=lambda x: -s[x - 1])[:10]
        ranking[pool.key] = [{"n": x, "score": round(s[x - 1], 4),
                              "freq": freq[pool.key][x - 1], "omission": omiss[pool.key][x - 1]} for x in order]

    summary = algo.summarize(draws, game, algo_params, scores)
    if getattr(algo, "generation_note", ""):
        summary += " " + algo.generation_note
    return {
        "game": game_key, "algo": algo_key, "algo_name": algo.name,
        "window": len(draws),
        "based_on": {"from": draws[0]["issue"], "to": draws[-1]["issue"], "count": len(draws)},
        "summary": summary,
        "tickets": tickets,
        "ranking": ranking,
    }


def backtest(game_key, algo_key, window=100, count=30, algo_params=None):
    """滚动回测：对最近 count 期逐期“只用之前的数据”预测一注，统计命中情况"""
    game = get_game(game_key)
    algo = get_algorithm(algo_key)
    algo_params = algo_params or {}
    params_json = json.dumps(algo_params, sort_keys=True)
    draws_all = storage.get_draws(game_key)
    if len(draws_all) < window + count + 5:
        count = max(len(draws_all) - window - 5, 5)
    m, s = game.main_pool, game.sub_pool
    baseline_main = m.count * m.count / m.max_number
    baseline_sub = s.count * s.count / s.max_number

    rows = []
    for i in range(len(draws_all) - count, len(draws_all)):
        train = draws_all[max(0, i - int(window)):i]
        if len(train) < 20:
            continue
        actual = draws_all[i]
        rng = _make_rng(game_key, algo_key, window, 0, train[-1]["issue"])
        cache_key = f"{game_key}|{algo_key}|{window}|{params_json}|{len(train)}|{train[-1]['issue']}"
        _, main_sets, sub_sets = _core_predict(train, game, algo, algo_params, 1, rng, cache_key=cache_key)
        pred = {m.key: main_sets[0], s.key: sub_sets[0]}
        main_hits = len(set(pred[m.key]) & set(actual["numbers"][m.key]))
        sub_hits = len(set(pred[s.key]) & set(actual["numbers"][s.key]))
        rows.append({"issue": actual["issue"], "date": actual["date"],
                     "predicted": pred, "actual": actual["numbers"],
                     "main_hits": main_hits, "sub_hits": sub_hits})

    n = max(len(rows), 1)
    threshold = 3 if m.count >= 5 else 2
    summary = {
        "count": len(rows),
        "avg_main_hits": round(sum(r["main_hits"] for r in rows) / n, 3),
        "baseline_main_hits": round(baseline_main, 3),
        "main_hit_rate_ge": round(sum(1 for r in rows if r["main_hits"] >= threshold) / n, 3),
        "main_hit_threshold": threshold,
        "avg_sub_hits": round(sum(r["sub_hits"] for r in rows) / n, 3),
        "baseline_sub_hits": round(baseline_sub, 3),
        "sub_full_rate": round(sum(1 for r in rows if r["sub_hits"] == s.count) / n, 3),
    }
    return {"game": game_key, "algo": algo_key, "algo_name": algo.name,
            "window": int(window), "baseline_note": "随机选号的理论平均命中",
            "summary": summary, "rows": rows}


# ---------- 预测跟踪 ----------

def save_tracking(game_key, algo_key, params, issue_target, tickets):
    """把一次预测的每注保存入库，待下期开奖后结算（tickets 元素为号码 dict 或含 numbers 的 ticket）"""
    params_json = json.dumps(params or {}, sort_keys=True)
    ids = []
    for t in tickets:
        numbers = t["numbers"] if isinstance(t, dict) and "numbers" in t else t
        ids.append(storage.save_prediction(game_key, algo_key, params_json, issue_target, json.dumps(numbers)))
    return ids


def settle_pending(game_key):
    """对已到开奖期的未结算预测计算命中并落库，返回结算条数"""
    game = get_game(game_key)
    m, s = game.main_pool, game.sub_pool
    n = 0
    for p in storage.unsettled_predictions(game_key):
        draw = storage.get_draw_by_issue(game_key, p["issue_target"])
        if not draw:
            continue
        pred = p["numbers"]
        mh = len(set(pred.get(m.key, [])) & set(draw["numbers"][m.key]))
        sh = len(set(pred.get(s.key, [])) & set(draw["numbers"][s.key]))
        storage.settle_prediction(p["id"], mh, sh)
        n += 1
    return n


def tracking_summary(game_key, limit=60):
    settle_pending(game_key)
    return {"stats": storage.prediction_stats(game_key), "rows": storage.get_predictions(game_key, limit)}


# ---------- 算法对比 ----------

def compare(game_key, algo_keys, window=100, sets=1, backtest_count=8, params_by_algo=None):
    """多算法并排：各出一注 + 快速回测摘要 + 已跟踪的历史命中"""
    params_by_algo = params_by_algo or {}
    results = []
    for key in algo_keys[:8]:
        algo = get_algorithm(key)
        params = params_by_algo.get(key, {})
        algo.generation_note = ""
        r = predict(game_key, key, window, sets, nonce=0, algo_params=params)
        bt = None
        if backtest_count:
            try:
                bt = backtest(game_key, key, window, backtest_count, params)["summary"]
            except Exception:
                bt = None
        results.append({"algo": key, "name": algo.name, "category": algo.category,
                        "ticket": r["tickets"][0], "bt": bt,
                        "summary": r["summary"]})
    track = {s["algo"]: s for s in storage.prediction_stats(game_key)}
    return {"game": game_key, "window": int(window), "backtest_count": backtest_count,
            "results": results, "tracking": track}
