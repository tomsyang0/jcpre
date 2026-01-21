#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fbref 全模块批量下载器（按建模字段清洗版）
author: You
"""
from DrissionPage import ChromiumPage
import pandas as pd
import io, re, time, random, pathlib

# 1. 联赛配置
########################################
LEAGUES = [
    [9, "Premier-League", "Premier-League"],  #英超
    [10,  "Championship", "Championship"],    #英冠
    [11, "Serie-A", "Serie-A"],               #意甲
    [12, "La-Liga", "La-Liga"],               #西甲
    [13, "Ligue-1", "Ligue-1"],               #法甲
    [60, "Ligue-2", "Ligue-2"],               #法乙
    [15, "League-One", "League-One"],         #英甲
    [20, "Bundesliga", "Bundesliga"],         #德甲
    [23, "Eredivisie", "Eredivisie"],         #荷甲
    [25, "J1-League", "J1-League"],           #日职联
    [32, "Primeira-Liga", "Primeira-Liga"],   #葡超
    [33,  "2-Bundesliga", "2-Bundesliga"],    #德乙
    [51,  "Eredivisie-2", "Eerste-Divisie"],  #荷乙
    [65, "A-League", "A-League"],             #澳超
    [70, "Saudi-Pro-League", "Saudi-Pro-League"],  #沙特职业联赛    
]

# -------------------- 2. 模块配置 --------------------
# key=保存文件夹名, value=(url 后缀或 schedule, 锚点, 表 ID, 列头兜底关键字, 是否输出中文)
TABLES = {
    "standard":   ("",          "",                  "stats_squads_standard",  "Squad",        True),
    "shooting":   ("",          "#stats_shooting",   "stats_squads_shooting",  "Shots",        True),
    "passing":    ("",          "#stats_passing",    "stats_squads_passing",   "KP",           True),
    "defense":    ("",          "#stats_defense",    "stats_squads_defense",   "Tkl",          True),
    "possession": ("",          "#stats_possession", "stats_squads_possession","Poss",         True),
    "home_away":  ("",          "#stats_home_away",  "stats_squads_home_away", "Home",         True),
    "set_pieces": ("",          "#stats_set_pieces", "stats_squads_set_pieces","Set",          True),
    "fixtures":   ("schedule",  "#fixtures",         "matchlogs_all",          "Date",         False),
}

# -------------------- 3. 中文映射（暂未使用，可后续扩展） --------------------
CN_MAP = {
    "Squad":"球队", "MP":"场次", "GF":"进球", "GA":"失球", "GD":"净胜球", "Pts":"积分",
    "xG":"预期进球", "xGA":"预期失球", "xGDiff":"xG差", "xGDiff/90":"xG差/90",
    "Shots":"射门", "SoT":"射正", "SoT%":"射正率", "G-xG":"进球-xG",
    "KP":"关键传球", "Cmp%":"传球成功率", "Prog":"推进带球",
    "Tkl":"抢断", "TklW":"抢断成功", "Blocks":"封堵", "Int":"拦截", "Clr":"解围",
    "Poss":"控球", "Press":"压迫", "PPDA":"PPDA",
}

# -------------------- 4. 工具函数 --------------------
def rand_sleep(a=4, b=8):
    time.sleep(random.uniform(a, b))

def clean_html(html: str) -> str:
    return re.sub(r"<!--|-->", "", html)

def save_df(df: pd.DataFrame, save_dir: pathlib.Path, filename: str, do_cn: bool):
    """
    保存DataFrame到指定目录，按赛季子目录存放
    """
    season_dir = save_dir / "2025-2026"
    season_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(season_dir / f"{filename}.csv", index=False, encoding="utf-8-sig")

def parse_table(html: str, *, table_id: str = "", wanted_text: str = "") -> pd.DataFrame | None:
    """优先用 table_id，没有再按列头关键字兜底"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(clean_html(html), "lxml")
    if table_id:
        tb = soup.select_one(f"table#{table_id}")
        if tb:
            return pd.read_html(io.StringIO(str(tb)))[0]
    if wanted_text:
        dfs = pd.read_html(io.StringIO(clean_html(html)))
        for df in dfs:
            if wanted_text.lower() in str(df.columns).lower():
                return df
    return None

def parse_the_only_stats_table(html: str) -> pd.DataFrame | None:
    """fbref 每页只有一张真正的 stats_table，直接抓它"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(clean_html(html), "lxml")
    tb = soup.select_one("table.stats_table")
    return pd.read_html(io.StringIO(str(tb)))[0] if tb else None

def post_process_df(df: pd.DataFrame, tbl_id: str) -> pd.DataFrame:
    """
    按你给的需求，对各表做字段筛选和类型转换
    """

    # 1）处理多重表头
    if isinstance(df.columns, pd.MultiIndex):
        if tbl_id == "stats_squads_home_away":
            # home_away 需要保留 Home/Away 信息
            new_cols = []
            for col in df.columns:
                parts = [
                    str(c).strip() for c in col
                    if pd.notna(c) and not str(c).startswith("Unnamed")
                ]
                new_cols.append(".".join(parts) if parts else "")
            df.columns = new_cols
        else:
            # 其他表直接用第二行表头（最后一层）
            df.columns = df.columns.get_level_values(-1)

    # 去掉完全空的列名
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]

    # 2）按具体表做字段筛选 & 类型处理
    if tbl_id == "matchlogs_all":
        # ========== 赛程 & 赛果 ==========
        needed = ["Date", "Home", "Away", "Score", "xG", "xG.1"]
        existing = [c for c in needed if c in df.columns]
        df = df[existing].copy()

        # 日期转 datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Score -> HomeGoals, AwayGoals
        if "Score" in df.columns:
            def _split_score(s):
                if not isinstance(s, str):
                    return pd.NA, pd.NA
                # 兼容 “1-0”、“1–0”、“1 : 0”等
                m = re.search(r"(\d+)\D+(\d+)", s)
                if m:
                    return int(m.group(1)), int(m.group(2))
                return pd.NA, pd.NA

            tmp = df["Score"].apply(_split_score)
            df["HomeGoals"] = tmp.apply(lambda x: x[0])
            df["AwayGoals"] = tmp.apply(lambda x: x[1])

        # xG 转 float
        for col in ["xG", "xG.1"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    elif tbl_id == "stats_squads_standard":
        # ========== 标准数据：GF, GA, xG, xGA, Pts ==========
        needed = ["Squad", "GF", "GA", "xG", "xGA", "Pts"]
        existing = [c for c in needed if c in df.columns]
        df = df[existing].copy()

    elif tbl_id == "stats_squads_home_away":
        # ========== 主客场 xG / xGA ==========
        # 此时列名形如：'Rk','Squad','Home.MP',...,'Home.xG','Home.xGA',...,'Away.xG','Away.xGA',...
        needed = ["Squad", "Home.xG", "Home.xGA", "Away.xG", "Away.xGA"]
        existing = [c for c in needed if c in df.columns]
        df = df[existing].copy()

    elif tbl_id == "stats_squads_defense":
        # ========== 防守：TklW, Lost, Blocks, Clr ==========
        # 多重表头第二行是：Squad, # Pl, 90s, Tkl, TklW, Def 3rd, ... Lost, Blocks, ... Clr, Err
        needed = ["Squad", "TklW", "Lost", "Blocks", "Clr"]
        existing = [c for c in needed if c in df.columns]
        df = df[existing].copy()

    elif tbl_id == "stats_squads_passing":
        # ========== 传球：xAG, PrgP, KP ==========
        needed = ["Squad", "xAG", "PrgP", "KP"]
        existing = [c for c in needed if c in df.columns]
        df = df[existing].copy()

    elif tbl_id == "stats_squads_possession":
        # ========== 控球：Poss, PrgC, G+A-PK ==========
        needed = ["Squad", "Poss", "PrgC", "G+A-PK"]
        existing = [c for c in needed if c in df.columns]
        df = df[existing].copy()

    # 其他表（shooting / set_pieces）保留原始字段，只做了多重表头展平
    return df

# -------------------- 5. 抓取函数 --------------------
def fetch_league(lid: int, folder: str, url_name: str, base: pathlib.Path):
    print(f"\n===== {folder} (id={lid}) =====")
    league_dir = base / folder
    league_dir.mkdir(parents=True, exist_ok=True)

    page = ChromiumPage()
    try:
        for mod, (sched, anchor, tbl_id, want, do_cn) in TABLES.items():
            print(f"  → {mod} …", end="")
            # 修复 URL 空格
            if sched == "schedule":
                url = f"https://fbref.com/en/comps/{lid}/schedule/{url_name}-Scores-and-Fixtures{anchor}"
            else:
                url = f"https://fbref.com/en/comps/{lid}/{url_name}-Stats{anchor}"

            ok = False
            for attempt in range(1, 4):
                try:
                    page.get(url, retry=2, timeout=30)
                    page.wait(2, 3)
                    page.wait.ele_displayed("table.stats_table", timeout=20)
                    page.wait(1, 2)

                    # 抓表
                    if mod in ("shooting", "set_pieces"):
                        df = parse_the_only_stats_table(page.html)
                    else:
                        df = parse_table(page.html, table_id=tbl_id, wanted_text=want)

                    if df is not None and not df.empty:
                        # 核心：按建模需求做字段清洗
                        df = post_process_df(df, tbl_id)
                        save_df(df, league_dir, tbl_id, do_cn)
                        print(f" {len(df)} 行 ✔")
                        ok = True
                        break
                    else:
                        print(f" 空表重试 {attempt}", end="")
                except Exception as e:
                    print(f" 异常重试 {attempt}: {e}", end="")
                rand_sleep(5, 8)

            if not ok:
                (league_dir / f"{tbl_id}.failed").touch()
                print("  失败 ❌")

            rand_sleep(5, 8)
    finally:
        page.quit()
    print(f"===== {folder} 完成 =====\n")

# -------------------- 6. 主入口 --------------------
if __name__ == "__main__":
    base = pathlib.Path("Global_Football_Stats")
    for lid, folder, url_name in LEAGUES:
        fetch_league(lid, folder, url_name, base)
        time.sleep(random.uniform(5, 10))