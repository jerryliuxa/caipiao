"""历史开奖数据抓取与解析。

国内游戏（双色球/大乐透）：datachart.500.com 历史图表页。
美国游戏（强力球/超级百万）：jbaranski/jeffs-lottery-utils 的自动更新 CSV
（GitHub raw 直连，每期开奖后数小时内更新，覆盖现行号码矩阵全部历史）；
网络不可用时回退到仓库内置的 seed 快照。
"""
import os
import re
import ssl
import urllib.request

from .games import get_game

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_URL_500 = "https://datachart.500.com/{game}/history/newinc/history.php?limit={limit}"
_URL_USCSV = {
    "pb": "https://raw.githubusercontent.com/jbaranski/jeffs-lottery-utils/main/numbers/powerball.csv",
    "mm": "https://raw.githubusercontent.com/jbaranski/jeffs-lottery-utils/main/numbers/megamillions.csv",
}
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.500.com/",
}
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE


def fetch_history(game_key, limit=300):
    game = get_game(game_key)
    if game.date_issue:
        return _fetch_uscsv(game, limit)
    return _fetch_500(game, limit)


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
    text = _download_csv(game.key)
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

    if not rows:
        raise ValueError("未解析到任何开奖记录")
    return rows[-int(limit):]  # CSV 为旧→新


def _download_csv(game_key, timeout=25):
    try:
        req = urllib.request.Request(_URL_USCSV[game_key], headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as resp:
            return resp.read().decode("utf-8", "ignore")
    except Exception:
        seed = os.path.join(_ROOT, "data", "seed", f"{game_key}.csv")
        with open(seed, "r", encoding="utf-8") as f:
            return f.read()  # 网络不可用时回退内置快照（可能略旧）


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
