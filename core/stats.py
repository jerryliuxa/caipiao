"""统计分析：频率 / 遗漏 / 和值 / 奇偶等"""
import math


def pool_series(draws, pool):
    """每期该池开出的号码列表（旧→新）"""
    return [d["numbers"][pool.key] for d in draws]


def frequency(draws, pool):
    """窗口内各号码出现次数，列表下标 = 号码-1"""
    maxn = pool.max_number
    freq = [0] * maxn
    for combo in pool_series(draws, pool):
        for n in combo:
            freq[n - 1] += 1
    return freq


def weighted_frequency(draws, pool, decay=0.97):
    """时间衰减加权频率：最新一期权重 1，往前每期乘 decay"""
    maxn = pool.max_number
    w = [0.0] * maxn
    n_draws = len(draws)
    for i, combo in enumerate(pool_series(draws, pool)):
        age = n_draws - 1 - i
        weight = decay ** age
        for num in combo:
            w[num - 1] += weight
    return w


def omission(draws, pool):
    """当前遗漏：距上次出现的期数（最新一期开出则为 0）"""
    maxn = pool.max_number
    gap = [len(draws)] * maxn  # 未在窗口内出现过的号码，遗漏 = 窗口长度
    for i, combo in enumerate(pool_series(draws, pool)):
        for num in combo:  # 从旧到新扫，最后一次出现覆盖为最新遗漏
            gap[num - 1] = len(draws) - 1 - i
    return gap


def avg_gaps(draws, pool):
    """各号码的平均遗漏（期望回补间隔）= (N+1)/(出现次数+1)"""
    n = len(draws)
    freq = frequency(draws, pool)
    return [(n + 1) / (c + 1) for c in freq]


def sum_series(draws, pool):
    """主池每期和值（旧→新）"""
    return [sum(c) for c in pool_series(draws, pool)]


def odd_count_hist(draws, pool):
    """奇数个数分布：{奇数个数: 期数}"""
    hist = {}
    for combo in pool_series(draws, pool):
        o = sum(1 for x in combo if x % 2)
        hist[o] = hist.get(o, 0) + 1
    return hist


def percentile(values, p):
    s = sorted(values)
    idx = (len(s) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def normalize(scores, lo=0.05, hi=1.0):
    """min-max 归一到 [lo, hi]，极差为 0 时全部取中间值"""
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-12:
        return [(lo + hi) / 2] * len(scores)
    return [lo + (v - mn) / (mx - mn) * (hi - lo) for v in scores]


def next_draw_date(last_date_str, weekdays):
    """按开奖星期推算下一期开奖日期"""
    import datetime
    try:
        d = datetime.date.fromisoformat(last_date_str)
    except (ValueError, TypeError):
        return ""
    for i in range(1, 15):
        nd = d + datetime.timedelta(days=i)
        if nd.weekday() in weekdays:
            return nd.isoformat()
    return ""


def zone_of(num, zone_size):
    return (num - 1) // zone_size


def zones(n_zones, pool):
    size = math.ceil(pool.max_number / n_zones)
    return size
