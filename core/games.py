"""游戏定义：双色球 / 大乐透"""
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
}


def get_game(key):
    if key not in GAMES:
        raise ValueError(f"未知游戏: {key}")
    return GAMES[key]
