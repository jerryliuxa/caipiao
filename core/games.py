"""游戏定义：双色球 / 大乐透 / 美国强力球 / 超级百万 / 欧洲三种乐透 / 维京乐透"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Pool:
    key: str        # 池标识，如 red / blue
    name: str       # 显示名，如 红球
    max_number: int # 号码上限
    count: int      # 每期开出个数
    color: str      # 前端配色 red / blue

    @property
    def numbers(self):
        return range(1, self.max_number + 1)


@dataclass(frozen=True)
class Game:
    key: str
    name: str
    pools: tuple            # (主池, 副池)
    draw_weekdays: tuple    # 开奖星期，0=周一 … 6=周日
    date_issue: bool = False  # True=无官方期号，以开奖日期 YYYYMMDD 作为期号

    @property
    def main_pool(self):
        return self.pools[0]

    @property
    def sub_pool(self):
        return self.pools[1]


GAMES = {
    "ssq": Game("ssq", "双色球",
                (Pool("red", "红球", 33, 6, "red"), Pool("blue", "蓝球", 16, 1, "blue")),
                (1, 3, 6)),
    "dlt": Game("dlt", "大乐透",
                (Pool("front", "前区", 35, 5, "red"), Pool("back", "后区", 12, 2, "blue")),
                (0, 2, 5)),
    "pb": Game("pb", "美国强力球",
               (Pool("white", "白球", 69, 5, "red"), Pool("powerball", "强力球", 26, 1, "blue")),
               (0, 2, 5), date_issue=True),
    "mm": Game("mm", "超级百万",
               (Pool("white", "白球", 70, 5, "red"), Pool("megaball", "Mega Ball", 24, 1, "blue")),
               (1, 4), date_issue=True),
    "em": Game("em", "欧洲百万",
               (Pool("main", "主号", 50, 5, "red"), Pool("star", "幸运星", 12, 2, "blue")),
               (1, 4), date_issue=True),
    "ej": Game("ej", "欧洲大奖",
               (Pool("main", "主号", 50, 5, "red"), Pool("euro", "Euro号", 12, 2, "blue")),
               (1, 4), date_issue=True),
    "ed": Game("ed", "欧洲梦想",
               (Pool("main", "主号", 40, 6, "red"), Pool("dream", "梦想号", 5, 1, "blue")),
               (0, 3), date_issue=True),
    "vl": Game("vl", "维京乐透",
               (Pool("main", "主号", 48, 6, "red"), Pool("viking", "维京号", 8, 1, "blue")),
               (2,), date_issue=True),
}


def get_game(key):
    if key not in GAMES:
        raise ValueError(f"未知游戏: {key}")
    return GAMES[key]
