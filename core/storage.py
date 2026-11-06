"""SQLite 本地存储"""
import json
import os
import sqlite3
import threading

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_DB_PATH = os.path.join(_DATA_DIR, "lottery.db")
_lock = threading.Lock()


def _conn():
    os.makedirs(_DATA_DIR, exist_ok=True)
    c = sqlite3.connect(_DB_PATH)
    c.execute(
        "CREATE TABLE IF NOT EXISTS draws ("
        " game TEXT NOT NULL, issue TEXT NOT NULL, draw_date TEXT NOT NULL,"
        " numbers TEXT NOT NULL, PRIMARY KEY (game, issue))"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS predictions ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " game TEXT NOT NULL, algo TEXT NOT NULL, params TEXT NOT NULL DEFAULT '{}',"
        " issue_target TEXT NOT NULL, numbers TEXT NOT NULL, created_at TEXT NOT NULL,"
        " settled INTEGER NOT NULL DEFAULT 0, main_hits INTEGER, sub_hits INTEGER)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS backtest_runs ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " fingerprint TEXT NOT NULL UNIQUE,"
        " game TEXT NOT NULL, algo TEXT NOT NULL, params TEXT NOT NULL DEFAULT '{}',"
        " window INTEGER NOT NULL, count INTEGER NOT NULL,"
        " result TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    return c


def upsert_draws(game_key, draws):
    """写入/更新开奖记录，返回新增期数"""
    with _lock, _conn() as c:
        before = c.execute("SELECT COUNT(*) FROM draws WHERE game=?", (game_key,)).fetchone()[0]
        c.executemany(
            "INSERT OR REPLACE INTO draws (game, issue, draw_date, numbers) VALUES (?,?,?,?)",
            [(game_key, d["issue"], d["date"], json.dumps(d["numbers"], ensure_ascii=False)) for d in draws],
        )
        after = c.execute("SELECT COUNT(*) FROM draws WHERE game=?", (game_key,)).fetchone()[0]
    return after - before


def get_draws(game_key, limit=None):
    """读取开奖记录，统一旧→新排列；limit 取最近 N 期"""
    base = "SELECT issue, draw_date, numbers FROM draws WHERE game=?"
    if limit:
        # 先取最新 N 期（DESC），再翻转为旧→新
        sql, params = base + " ORDER BY draw_date DESC, CAST(issue AS INTEGER) DESC LIMIT ?", [game_key, int(limit)]
    else:
        sql, params = base + " ORDER BY draw_date ASC, CAST(issue AS INTEGER) ASC", [game_key]
    with _conn() as c:
        rows = c.execute(sql, params).fetchall()
    if limit:
        rows.reverse()
    return [{"issue": r[0], "date": r[1], "numbers": json.loads(r[2])} for r in rows]


def count(game_key):
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM draws WHERE game=?", (game_key,)).fetchone()[0]


def latest_issue(game_key):
    """该游戏最新一期的期号（无数据返回空串）"""
    with _conn() as c:
        row = c.execute(
            "SELECT issue FROM draws WHERE game=?"
            " ORDER BY draw_date DESC, CAST(issue AS INTEGER) DESC LIMIT 1",
            (game_key,),
        ).fetchone()
    return row[0] if row else ""


# ---------- 预测跟踪 ----------

def save_prediction(game_key, algo_key, params_json, issue_target, numbers_json):
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO predictions (game, algo, params, issue_target, numbers, created_at)"
            " VALUES (?,?,?,?,?, datetime('now','localtime'))",
            (game_key, algo_key, params_json, issue_target, numbers_json),
        )
        return cur.lastrowid


def get_draw_by_issue(game_key, issue):
    with _conn() as c:
        row = c.execute("SELECT issue, draw_date, numbers FROM draws WHERE game=? AND issue=?",
                        (game_key, issue)).fetchone()
    return {"issue": row[0], "date": row[1], "numbers": json.loads(row[2])} if row else None


def unsettled_predictions(game_key):
    with _conn() as c:
        rows = c.execute(
            "SELECT id, algo, issue_target, numbers FROM predictions WHERE game=? AND settled=0",
            (game_key,),
        ).fetchall()
    return [{"id": r[0], "algo": r[1], "issue_target": r[2], "numbers": json.loads(r[3])} for r in rows]


def settle_prediction(pid, main_hits, sub_hits):
    with _lock, _conn() as c:
        c.execute("UPDATE predictions SET settled=1, main_hits=?, sub_hits=? WHERE id=?",
                  (main_hits, sub_hits, pid))


def get_predictions(game_key, limit=60):
    with _conn() as c:
        rows = c.execute(
            "SELECT id, algo, issue_target, numbers, created_at, settled, main_hits, sub_hits"
            " FROM predictions WHERE game=? ORDER BY id DESC LIMIT ?",
            (game_key, int(limit)),
        ).fetchall()
    return [{"id": r[0], "algo": r[1], "issue_target": r[2], "numbers": json.loads(r[3]),
             "created_at": r[4], "settled": bool(r[5]), "main_hits": r[6], "sub_hits": r[7]}
            for r in rows]


# ---------- 回测存档 ----------

def save_backtest_run(game_key, algo_key, params_json, window, count, result_json, last_issue):
    """存档一次回测运行（同指纹=同算法+参数+数据期，仅刷新时间不重复入库）"""
    import hashlib

    fp = hashlib.md5(
        f"{game_key}|{algo_key}|{params_json}|{window}|{count}|{last_issue}".encode("utf-8")
    ).hexdigest()
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO backtest_runs (fingerprint, game, algo, params, window, count, result, created_at)"
            " VALUES (?,?,?,?,?,?,?, datetime('now','localtime'))"
            " ON CONFLICT(fingerprint) DO UPDATE SET created_at=datetime('now','localtime')",
            (fp, game_key, algo_key, params_json, int(window), int(count), result_json),
        )
        c.execute(
            "DELETE FROM backtest_runs WHERE id NOT IN"
            " (SELECT id FROM backtest_runs ORDER BY id DESC LIMIT 500)"
        )


def list_backtest_runs(game_key, limit=30):
    """回测存档列表（新→旧，不含逐期明细）"""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, algo, window, count, result, created_at FROM backtest_runs"
            " WHERE game=? ORDER BY id DESC LIMIT ?",
            (game_key, int(limit)),
        ).fetchall()
    out = []
    for r in rows:
        try:
            summary = json.loads(r[4]).get("summary", {})
        except Exception:
            summary = {}
        out.append({"id": r[0], "algo": r[1], "window": r[2], "count": r[3],
                    "summary": summary, "created_at": r[5]})
    return out


def get_backtest_run(run_id):
    """取一次回测存档的完整结果（结构与 /api/backtest 返回一致）"""
    with _conn() as c:
        row = c.execute("SELECT result FROM backtest_runs WHERE id=?", (int(run_id),)).fetchone()
    return json.loads(row[0]) if row else None


def prediction_stats(game_key):
    with _conn() as c:
        rows = c.execute(
            "SELECT algo, COUNT(*), SUM(settled),"
            " AVG(CASE WHEN settled=1 THEN main_hits END),"
            " AVG(CASE WHEN settled=1 THEN sub_hits END),"
            " SUM(CASE WHEN settled=1 AND sub_hits>0 THEN 1 ELSE 0 END)"
            " FROM predictions WHERE game=? GROUP BY algo ORDER BY COUNT(*) DESC",
            (game_key,),
        ).fetchall()
    return [{"algo": r[0], "total": r[1], "settled": r[2] or 0,
             "avg_main": round(r[3], 3) if r[3] is not None else None,
             "avg_sub": round(r[4], 3) if r[4] is not None else None,
             "sub_hit_count": r[5] or 0} for r in rows]
