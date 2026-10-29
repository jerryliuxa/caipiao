"""彩票分析预测平台 — Web 服务（仅 Python 标准库，无第三方依赖）"""
import json
import os
import socket
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from core import datasource, service, storage
from core.algorithms import ALGORITHMS, availability
from core.games import GAMES

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
        ".ico": "image/x-icon"}


def ensure_data():
    """首次启动时自动抓取近 300 期数据"""
    for gk in GAMES:
        try:
            if storage.count(gk) < 50:
                draws = datasource.fetch_history(gk, 300)
                added = storage.upsert_draws(gk, draws)
                print(f"[data] {GAMES[gk].name} 已载入 {len(draws)} 期（新增 {added}）")
        except Exception as exc:
            print(f"[data] {gk} 初始化抓取失败: {exc}（可稍后在页面点击“更新数据”）")


class Handler(BaseHTTPRequestHandler):
    server_version = "LotteryLab/1.0"

    def log_message(self, fmt, *args):
        pass  # 静默访问日志，保持控制台清爽

    # ---------- 基础 ----------
    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    def _static(self, rel_path):
        path = os.path.normpath(os.path.join(WEB_DIR, rel_path.lstrip("/")))
        if not path.startswith(WEB_DIR) or not os.path.isfile(path):
            self._json({"error": "not found"}, 404)
            return
        ext = os.path.splitext(path)[1]
        with open(path, "rb") as f:
            self._send(200, MIME.get(ext, "application/octet-stream"), f.read())

    # ---------- 路由 ----------
    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            route = parsed.path
            q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if route.startswith("/api/"):
                self._api(route, q)
            elif route in ("/", "/index.html"):
                self._static("/index.html")
            else:
                self._static(route)
        except BrokenPipeError:
            pass
        except Exception:
            traceback.print_exc()
            try:
                self._json({"error": "服务器内部错误，详情见控制台"}, 500)
            except Exception:
                pass

    def do_POST(self):
        """写操作：保存预测跟踪"""
        try:
            route = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if route == "/api/track":
                ids = service.save_tracking(body.get("game", "ssq"), body.get("algo", ""),
                                            body.get("params", {}), body.get("issue_target", ""),
                                            body.get("tickets", []))
                self._json({"ok": True, "ids": ids})
            else:
                self._json({"error": f"未知接口 {route}"}, 404)
        except BrokenPipeError:
            pass
        except Exception:
            traceback.print_exc()
            try:
                self._json({"error": "服务器内部错误，详情见控制台"}, 500)
            except Exception:
                pass

    def _api(self, route, q):
        if route == "/api/games":
            self._json({k: {"key": g.key, "name": g.name,
                            "pools": [p.__dict__ for p in g.pools]} for k, g in GAMES.items()})

        elif route == "/api/algorithms":
            self._json({"algorithms": [{"key": a.key, "name": a.name, "desc": a.desc,
                                        "category": a.category, "heavy": a.heavy,
                                        "params": a.params} for a in ALGORITHMS],
                        "availability": availability()})

        elif route == "/api/history":
            gk = q.get("game", "ssq")
            limit = min(int(q.get("limit", 30)), 1000)
            draws = storage.get_draws(gk, limit)
            self._json({"game": gk, "total": storage.count(gk), "draws": list(reversed(draws))})

        elif route == "/api/update":
            gk = q.get("game", "all")
            limit = min(int(q.get("limit", 120)), 2000)
            games = list(GAMES) if gk == "all" else [gk]
            result = {}
            for g in games:
                fetched = datasource.fetch_history(g, limit)
                added = storage.upsert_draws(g, fetched)
                settled = service.settle_pending(g)
                result[g] = {"fetched": len(fetched), "added": added, "total": storage.count(g),
                             "settled": settled}
            self._json({"ok": True, "result": result})

        elif route == "/api/tracking":
            self._json(service.tracking_summary(q.get("game", "ssq"),
                                                limit=min(int(q.get("limit", 60)), 300)))

        elif route == "/api/compare":
            algos = [a for a in q.get("algos", "").split(",") if a][:8]
            if not algos:
                self._json({"error": "请通过 algos 参数指定至少一个算法"}, 400)
                return
            params_by_algo = json.loads(q.get("params", "{}"))
            self._json(service.compare(
                q.get("game", "ssq"), algos,
                window=int(q.get("window", 100)), sets=1,
                backtest_count=min(int(q.get("backtest", 8)), 15),
                params_by_algo=params_by_algo))

        elif route == "/api/stats":
            self._json(self._stats(q.get("game", "ssq"), int(q.get("window", 100))))

        elif route == "/api/predict":
            params = json.loads(q.get("params", "{}"))
            self._json(service.predict(
                q.get("game", "ssq"), q.get("algo", "ensemble"),
                window=int(q.get("window", 100)), n_sets=int(q.get("sets", 5)),
                nonce=q.get("nonce", "0"), algo_params=params))

        elif route == "/api/backtest":
            params = json.loads(q.get("params", "{}"))
            self._json(service.backtest(
                q.get("game", "ssq"), q.get("algo", "ensemble"),
                window=int(q.get("window", 100)), count=int(q.get("count", 30)),
                algo_params=params))

        else:
            self._json({"error": f"未知接口 {route}"}, 404)

    # ---------- 统计 ----------
    def _stats(self, game_key, window):
        from core import stats as st

        game = GAMES[game_key]
        draws_all = storage.get_draws(game_key)
        if not draws_all:
            return {"game": game_key, "empty": True, "total": 0}
        draws = draws_all[-window:]
        latest = draws[-1]
        pools = {}
        for pool in game.pools:
            freq = st.frequency(draws, pool)
            omiss = st.omission(draws, pool)
            lo, hi = min(freq), max(freq)
            span = (hi - lo) or 1
            pools[pool.key] = {
                "name": pool.name, "max": pool.max_number, "count": pool.count, "color": pool.color,
                "freq": freq, "omission": omiss,
                "avg_omission": [round(v, 2) for v in st.avg_gaps(draws, pool)],
                "heat": [round((v - lo) / span, 3) for v in freq],
            }
        sums = st.sum_series(draws, game.main_pool)
        m = game.main_pool
        odd_labels = [f"{o}奇{m.count - o}偶" for o in range(m.count + 1)]
        odd_values = [st.odd_count_hist(draws, m).get(o, 0) for o in range(m.count + 1)]
        # 主池号码共现矩阵（对称，对角线为该号码出现次数）
        N = m.max_number
        pairs = [[0] * N for _ in range(N)]
        for combo in st.pool_series(draws, m):
            for i in combo:
                for j in combo:
                    pairs[i - 1][j - 1] += 1
        if game.date_issue:  # 美国游戏无官方期号，以下一开奖日期为跟踪目标
            next_issue = st.next_draw_date(latest["date"], game.draw_weekdays).replace("-", "")
        else:
            next_issue = str(int(latest["issue"]) + 1).zfill(len(latest["issue"]))
        return {
            "game": game_key, "total": len(draws_all), "window": len(draws),
            "latest": latest,
            "next_issue": next_issue,
            "next_date": st.next_draw_date(latest["date"], game.draw_weekdays),
            "pools": pools,
            "sums": {"series": sums, "avg": round(sum(sums) / len(sums), 1),
                     "p10": round(st.percentile(sums, 0.10)), "p90": round(st.percentile(sums, 0.90))},
            "parity": {"labels": odd_labels, "values": odd_values},
            "pairs": pairs,
        }


def find_port(start=8787):
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("8787~8806 端口均被占用")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    port = find_port()
    threading.Thread(target=ensure_data, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"彩票分析预测平台已启动: {url}  (Ctrl+C 退出)")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
