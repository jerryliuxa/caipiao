"""500.com 历史开奖数据抓取与解析（datachart.500.com 历史图表页）"""
import re
import ssl
import urllib.request

from .games import get_game

_URL = "https://datachart.500.com/{game}/history/newinc/history.php?limit={limit}"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.500.com/",
}
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE


def fetch_history(game_key, limit=300):
    """抓取最近 limit 期开奖，返回旧→新排列的 draw 列表。

    draw 结构: {"issue": "26114", "date": "2026-10-06", "numbers": {"red": [...], "blue": [...]}}
    """
    game = get_game(game_key)
    url = _URL.format(game=game_key, limit=limit)
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
