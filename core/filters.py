"""组合过滤器：和值 / 奇偶 / 区间 / 012路 / AC值 / 跨度 / 尾数 / 大小比 / 质合比 / 连号 / 重号
所有阈值均由窗口内历史分布驱动；ok() 支持分级（2=全部约束，1=核心约束，0=不过滤），
供出票管线在抽样失败时逐级放宽。
"""
import math

from . import stats


def _ac_value(combo):
    """AC 值：两两差值去重个数 - (K-1)，衡量组合离散度"""
    diffs = {abs(a - b) for i, a in enumerate(combo) for b in combo[i + 1:]}
    return len(diffs) - (len(combo) - 1)


class ComboFilter:
    def __init__(self, draws, game):
        main = game.main_pool
        combos = stats.pool_series(draws, main)
        n = max(len(combos), 1)
        k = main.count
        self.k = k

        # 和值：P08~P92
        sums = [sum(c) for c in combos]
        self.sum_lo = round(stats.percentile(sums, 0.08))
        self.sum_hi = round(stats.percentile(sums, 0.92))

        def allowed_by(hist, ratio):
            return {v for v, c in hist.items() if c / n >= ratio}

        # 奇偶比 / 大小比 / 质合比：出现率 ≥6%
        odd_hist = {}
        big_hist = {}
        prime_hist = {}
        median = (main.max_number + 1) / 2
        primes = {x for x in range(2, main.max_number + 1)
                  if all(x % d for d in range(2, int(math.sqrt(x)) + 1))}
        for c in combos:
            odd_hist[sum(1 for x in c if x % 2)] = odd_hist.get(sum(1 for x in c if x % 2), 0) + 1
            big = sum(1 for x in c if x >= median)
            big_hist[big] = big_hist.get(big, 0) + 1
            pr = sum(1 for x in c if x in primes)
            prime_hist[pr] = prime_hist.get(pr, 0) + 1
        self.allowed_odd = allowed_by(odd_hist, 0.06)
        self.allowed_big = allowed_by(big_hist, 0.06)
        self.allowed_prime = allowed_by(prime_hist, 0.06)
        self.median = median
        self.primes = primes

        # 三分区 / 012路：出现率 ≥3%
        self.zone_size = math.ceil(main.max_number / 3)
        zone_hist, road_hist = {}, {}
        for c in combos:
            zt = tuple(sum(1 for x in c if stats.zone_of(x, self.zone_size) == z) for z in range(3))
            zone_hist[zt] = zone_hist.get(zt, 0) + 1
            rt = tuple(sum(1 for x in c if x % 3 == r) for r in range(3))
            road_hist[rt] = road_hist.get(rt, 0) + 1
        self.allowed_zones = allowed_by(zone_hist, 0.03)
        self.allowed_roads = allowed_by(road_hist, 0.03)

        # AC 值 / 跨度 / 最大同尾数：出现率 ≥3%
        ac_hist, span_hist, tail_hist = {}, {}, {}
        for c in combos:
            ac = _ac_value(c)
            ac_hist[ac] = ac_hist.get(ac, 0) + 1
            span = c[-1] - c[0]
            span_hist[span] = span_hist.get(span, 0) + 1
            tails = {}
            for x in c:
                tails[x % 10] = tails.get(x % 10, 0) + 1
            mt = max(tails.values())
            tail_hist[mt] = tail_hist.get(mt, 0) + 1
        self.allowed_ac = allowed_by(ac_hist, 0.03)
        span_list = [s for s, cnt in span_hist.items() for _ in range(cnt)]
        self.span_lo = stats.percentile(span_list, 0.03)
        self.span_hi = stats.percentile(span_list, 0.97)
        self.allowed_tail = allowed_by(tail_hist, 0.03)

        # 连号 / 重号上限
        self.last = set(combos[-1]) if combos else set()

    def _zones(self, combo):
        return tuple(sum(1 for x in combo if stats.zone_of(x, self.zone_size) == z) for z in range(3))

    def _roads(self, combo):
        return tuple(sum(1 for x in combo if x % 3 == r) for r in range(3))

    def ok(self, combo, level=2):
        """level 2=全部约束；1=核心约束（和值/奇偶/重号/连号）；0=不过滤"""
        if level <= 0:
            return True
        combo = sorted(combo)
        k = self.k
        if not (self.sum_lo <= sum(combo) <= self.sum_hi):
            return False
        if self.allowed_odd and sum(1 for x in combo if x % 2) not in self.allowed_odd:
            return False
        # 连号：不允许三连，最多两组二连
        run, pairs = 1, 0
        for a, b in zip(combo, combo[1:]):
            if b - a == 1:
                run += 1
            else:
                if run >= 3:
                    return False
                if run == 2:
                    pairs += 1
                run = 1
        if run >= 3 or pairs > 2:
            return False
        if len(set(combo) & self.last) > 2:
            return False
        if level < 2:
            return True
        # —— 以下为扩展形态约束 ——
        if self.allowed_big and sum(1 for x in combo if x >= self.median) not in self.allowed_big:
            return False
        if self.allowed_prime and sum(1 for x in combo if x in self.primes) not in self.allowed_prime:
            return False
        if self.allowed_zones and self._zones(combo) not in self.allowed_zones:
            return False
        if self.allowed_roads and self._roads(combo) not in self.allowed_roads:
            return False
        if self.allowed_ac and _ac_value(combo) not in self.allowed_ac:
            return False
        if not (self.span_lo <= combo[-1] - combo[0] <= self.span_hi):
            return False
        tails = {}
        for x in combo:
            tails[x % 10] = tails.get(x % 10, 0) + 1
        if self.allowed_tail and max(tails.values()) not in self.allowed_tail:
            return False
        return True
