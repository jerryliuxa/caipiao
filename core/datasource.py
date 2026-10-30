"""历史开奖数据抓取与解析。

双色球 / 大乐透：datachart.500.com 历史图表页。
美国强力球 / 超级百万：jbaranski/jeffs-lottery-utils 自动更新 CSV（GitHub raw 直连）。
欧洲百万 / 欧洲大奖：dev-baris/lottery-archive 自动更新 CSV（GitHub Actions 每期后更新，
  只保留现行号码矩阵：欧洲百万 2016-09-27 幸运星 1-12 起、欧洲大奖 2022-03-25 Euro号 1-12 起）。
欧洲梦想：euro-millions.com 年度归档页（2023 年开售）。
维京乐透：Norsk Tipping 官方 API（单次查询窗口 ≤15 周，自动分块回溯）。
所有数据源在抓取失败时回退到 data/seed/ 内置快照。
"""
import datetime
import json
import os
import re
import ssl
import time
import urllib.request

from .games import get_game

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_URL_500 = "https://datachart.500.com/{game}/history/newinc/history.php?limit={limit}"
_URL_USCSV = {
    "pb": "https://raw.githubusercontent.com/jbaranski/jeffs-lottery-utils/main/numbers/powerball.csv",
    "mm": "https://raw.githubusercontent.com/jbaranski/jeffs-lottery-utils/main/numbers/megamillions.csv",
}
# lottery-archive：date,n1..n5,x1,x2（x 为幸运星 / Euro 号），按现行矩阵起点过滤
_URL_EUCSV = {
    "em": ("https://raw.githubusercontent.com/dev-baris/lottery-archive/main/eu/euromillions/results.csv", "2016-09-27"),
    "ej": ("https://raw.githubusercontent.com/dev-baris/lottery-archive/main/eu/eurojackpot/results.csv", "2022-03-25"),
}
_URL_ED_YEAR = "https://www.euro-millions.com/eurodreams/results-history-{year}"
_URL_VL_API = "https://api.norsk-tipping.no/LotteryGameInfo/v2/api/results/vikinglotto?fromDate={f}&toDate={t}"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.500.com/",
    "Accept": "text/html,application/json,*/*",
}
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE


def fetch_history(game_key, limit=300):
    game = get_game(game_key)
    if game_key in ("ssq", "dlt"):
        return _fetch_500(game, limit)
    if game_key in ("pb", "mm"):
        return _fetch_uscsv(game, limit)
    if game_key in ("em", "ej"):
        return _fetch_eucsv(game, limit)
    if game_key == "ed":
        return _fetch_ed(game, limit)
    if game_key == "vl":
        return _fetch_vl(game, limit)
    raise ValueError(f"未知游戏: {game_key}")


def _http_get(url, timeout=25, retries=2):
    """带重试的 GET（raw.githubusercontent 等站点偶发连接失败）"""
    last = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as resp:
                return resp.read()
        except Exception as exc:
            last = exc
            time.sleep(0.8)
    raise last


def _seed_text(game_key):
    """网络不可用时回退内置快照（normalized CSV: issue,date,main...,sub...）"""
    seed = os.path.join(_ROOT, "data", "seed", f"{game_key}.csv")
    with open(seed, "r", encoding="utf-8") as f:
        return f.read()


def _seed_rows(game, limit):
    rows = []
    for line in _seed_text(game.key).splitlines()[1:]:
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3:
            continue
        main = [int(x) for x in p[2].split() if x.isdigit()]
        sub = [int(x) for x in p[3].split() if x.isdigit()] if len(p) > 3 else []
        numbers = {game.main_pool.key: main, game.sub_pool.key: sub}
        if not _valid(game, numbers):
            continue
        rows.append({"issue": p[0], "date": p[1], "numbers": numbers})
    return rows[-int(limit):]


# ---------- 500.com（双色球 / 大乐透） ----------

def _fetch_500(game, limit=300):
    url = _URL_500.format(game=game.key, limit=limit)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=25, context=_SSL) as resp:
        raw = resp.read()
    html = raw.decode("utf-8", "ignore")
    if "双色球" not in html and "大乐透" not in html:
        html = raw.decode("gbk", "ignore")

    m = re.search(r'<tbody[^>]*id="tdata"[^>]*>(.*?)</tbody>', html, re.S)
    if not m:
        raise ValueError("500.com 页面结构变化，未找到开奖数据表")

    n_main, n_sub = game.main_pool.count, game.sub_pool.count
    draws = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S):
        tds = [re.sub(r"<[^>]+>", "", t).strip() for t in re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)]
        issue_i = next((i for i, t in enumerate(tds) if re.fullmatch(r"\d{4,7}", t)), None)
        if issue_i is None:
            continue
        nums = []
        j = issue_i + 1
        while j < len(tds) and len(nums) < n_main + n_sub:
            if re.fullmatch(r"\d{1,2}", tds[j]):
                nums.append(int(tds[j]))
            j += 1
        if len(nums) < n_main + n_sub:
            continue
        date_m = next((t for t in tds if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t)), "")
        numbers = {
            game.main_pool.key: sorted(nums[:n_main]),
            game.sub_pool.key: sorted(nums[n_main:]),
        }
        if not _valid(game, numbers):
            continue
        draws.append({"issue": tds[issue_i], "date": date_m, "numbers": numbers})

    if not draws:
        raise ValueError("未解析到任何开奖记录")
    draws.reverse()  # 页面为新→旧，统一为旧→新
    return draws


# ---------- 美国 CSV（强力球 / 超级百万） ----------

def _fetch_uscsv(game, limit=300):
    try:
        text = _http_get(_URL_USCSV[game.key]).decode("utf-8", "ignore")
        rows = _parse_uscsv(game, text)
    except Exception:
        rows = _seed_rows(game, limit)  # 回退内置快照
    if not rows:
        raise ValueError("未解析到任何开奖记录")
    return rows[-int(limit):]


def _parse_uscsv(game, text):
    rows = []
    for line in text.splitlines()[1:]:  # 跳过表头
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        date_iso = _parse_date(parts[0])
        if not date_iso:
            continue
        main = [int(x) for x in parts[1].split("|") if x.strip().isdigit()]
        sub = [int(x) for x in parts[2].split("|") if x.strip().isdigit()]
        numbers = {game.main_pool.key: sorted(main), game.sub_pool.key: sorted(sub)}
        if not _valid(game, numbers):
            continue
        rows.append({"issue": date_iso.replace("-", ""), "date": date_iso, "numbers": numbers})
    return rows


# ---------- 欧洲 CSV（欧洲百万 / 欧洲大奖） ----------

def _fetch_eucsv(game, limit=300):
    url, matrix_start = _URL_EUCSV[game.key]
    rows = []
    try:
        text = _http_get(url).decode("utf-8", "ignore")
        for line in text.splitlines()[1:]:
            p = [x.strip() for x in line.split(",")]
            if len(p) != 8 or p[0] < matrix_start:
                continue
            try:
                nums = [int(x) for x in p[1:]]
            except ValueError:
                continue
            main, sub = nums[:5], nums[5:]
            if sorted(main) != main or sorted(sub) != sub or len(set(nums)) != 7:
                continue
            numbers = {game.main_pool.key: main, game.sub_pool.key: sub}
            if not _valid(game, numbers):
                continue
            rows.append({"issue": p[0].replace("-", ""), "date": p[0], "numbers": numbers})
    except Exception:
        rows = _seed_rows(game, limit)  # 回退内置快照
    if not rows:
        raise ValueError("未解析到任何开奖记录")
    return rows[-int(limit):]


# ---------- 欧洲梦想（euro-millions.com 年度归档） ----------

def _fetch_ed(game, limit=300):
    rows, year = [], datetime.date.today().year
    while year >= 2023 and len(rows) < int(limit):
        try:
            raw = _http_get(_URL_ED_YEAR.format(year=year))
            rows.extend(_parse_ed_html(game, raw.decode("utf-8", "ignore")))
        except Exception:
            break
        year -= 1
        time.sleep(0.4)
    if not rows:
        rows = _seed_rows(game, limit)
    if not rows:
        raise ValueError("未解析到任何开奖记录")
    rows.sort(key=lambda d: d["issue"])
    seen, uniq = set(), []
    for d in rows:
        if d["issue"] not in seen:
            seen.add(d["issue"])
            uniq.append(d)
    return uniq[-int(limit):]


def _parse_ed_html(game, html):
    rows = []
    for tr in re.findall(r'<tr class="resultRow".*?</tr>', html, re.S):
        dm = re.search(r'href="/[^"]*results/(\d{2})-(\d{2})-(\d{4})"', tr)
        if not dm:
            continue
        d, mo, y = dm.groups()
        mains = [int(x) for x in re.findall(r'class="resultBall eurodreams ball small">(\d+)<', tr)]
        sub = [int(x) for x in re.findall(r'class="resultBall eurodreams dream-number small">(\d+)<', tr)]
        numbers = {game.main_pool.key: sorted(mains), game.sub_pool.key: sorted(sub)}
        if not _valid(game, numbers):
            continue
        date_iso = f"{y}-{mo}-{d}"
        rows.append({"issue": date_iso.replace("-", ""), "date": date_iso, "numbers": numbers})
    return rows


# ---------- 维京乐透（Norsk Tipping 官方 API，15 周窗口分块） ----------

def _fetch_vl(game, limit=300):
    rows, cursor = {}, datetime.date.today()
    floor = cursor - datetime.timedelta(days=730)  # 最多回溯两年
    while cursor > floor and len(rows) < int(limit):
        start = max(cursor - datetime.timedelta(weeks=15), floor)
        try:
            raw = _http_get(_URL_VL_API.format(f=start.isoformat(), t=cursor.isoformat()))
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            break
        for rec in data.get("gameResult", []):
            wn = rec.get("winnerNumber") or []
            mains = sorted(int(x["number"]) for x in wn if x.get("type") == 1)
            subs = sorted(int(x["number"]) for x in wn if x.get("type") == 2)
            if not mains or not subs:
                continue
            numbers = {game.main_pool.key: mains, game.sub_pool.key: subs}
            if not _valid(game, numbers):
                continue
            date_iso = rec["drawDate"][:10]
            rows[date_iso] = {"issue": date_iso.replace("-", ""), "date": date_iso, "numbers": numbers}
        if start <= floor:
            break
        cursor = start - datetime.timedelta(days=1)
        time.sleep(0.3)
    if not rows:
        rows = {d["issue"]: d for d in _seed_rows(game, limit)}
    if not rows:
        raise ValueError("未解析到任何开奖记录")
    out = sorted(rows.values(), key=lambda d: d["issue"])
    return out[-int(limit):]


# ---------- 公共 ----------

def _parse_date(us):
    """M/D/YYYY → YYYY-MM-DD，无法解析返回空串"""
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", us.strip())
    if not m:
        return ""
    mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return ""
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _valid(game, numbers):
    for pool in game.pools:
        vals = numbers.get(pool.key, [])
        if len(vals) != pool.count:
            return False
        if len(set(vals)) != pool.count:
            return False
        if any(not (1 <= v <= pool.max_number) for v in vals):
            return False
    return True
