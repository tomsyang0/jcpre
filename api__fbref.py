#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
竞彩 API + fbref 队伍统计 → 预测用综合表

依赖：
    pip install requests pandas numpy scikit-learn

新增依赖（并发需要）：
    pip install tqdm

说明：
1. 使用统一接口 getUniformMatchResultV1.qry 获取比赛列表 + 当前让球盘赔率（h,d,a,goalLine）。
2. 使用 getFixedBonusV1.qry 获取：
   - 胜平负 hadList（开盘/收盘 SP）
   - 让球胜平负 hhadList（开盘/收盘 SP + 盘口）
   - 比分盘 crsList（推算主/客/总进球期望）
   - 总进球盘 ttgList（推算总进球期望）
   - sectionsNo999（最终正确比分）
3. 使用你已有的 team_map备份.csv 将中文队名映射到 fbref 英文队名。
4. 从 Global_Football_Stats/**/2025-2026/ 下读取 fbref 各 squads CSV，构造球队实力特征：
   - xG / xGA / Pts
   - Home/Away xG / xGA
   - 防守强度因子 def_strength（TklW, Lost, Blocks, Clr PCA）
   - 进攻组织因子 off_buildup（xAG, PrgP, KP PCA）
   - 节奏/控场因子 tempo（Poss, PrgC, G+A-PK PCA）
5. 输出综合表 CSV：datasets/jc_fbref_bonus_{start}_to_{end}.csv
"""

import requests
import pandas as pd
import numpy as np
import pathlib
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # 进度条

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================
# 1. 配置区
# =========================

# 竞彩统一赛果+赔率 API
JC_UNIFORM_URL = "https://webapi.sporttery.cn/gateway/uniform/football/getUniformMatchResultV1.qry"
# 竞彩固定奖金 API
JC_BONUS_URL = "https://webapi.sporttery.cn/gateway/uniform/football/getFixedBonusV1.qry"

# fbref 数据根目录 & 赛季子目录（对应你前面爬虫脚本的输出）
FBREF_BASE_DIR = pathlib.Path("Global_Football_Stats")
FBREF_SEASON_DIR = "2025-2026"

# 竞彩中文队名 ↔ fbref 英文队名 映射文件
TEAM_MAP_PATH = pathlib.Path("team_map备份.csv")

# 输出综合表目录
OUTPUT_DIR = pathlib.Path("datasets")

# 抓取的日期范围（修改：2025年1月1日至今）
START_DATE = "2025-01-01"
END_DATE = "2026-01-15"  # 或 datetime.now().strftime("%Y-%m-%d")

# 并发配置
MAX_WORKERS = 5 # bonus接口并发数（根据网络环境调整，建议5-20）
RATE_LIMIT_DELAY = 0.5  # 每个请求的延迟（秒），避免触发限流

# =========================
# 2. 通用工具函数
# =========================

def get_json(url: str,
             params: Optional[Dict[str, Any]] = None,
             timeout: int = 10) -> Any:
    """通用 GET JSON 封装"""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.sporttery.cn/",
        "Origin": "https://www.sporttery.cn/",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def safe_float(x: Any) -> float:
    """安全转 float，失败返回 NaN"""
    try:
        if x is None or x == "":
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def parse_score(score: Any) -> (Optional[int], Optional[int]):
    """把 '2:1' / '3-0' 之类比分解析成 (主, 客)"""
    if not isinstance(score, str):
        return None, None
    m = re.search(r"(\d+)\D+(\d+)", score)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def probs_from_sp(sp_home: float,
                  sp_draw: float,
                  sp_away: float) -> (float, float, float):
    """由三项 SP 算隐含概率（简单 1/SP 归一化），无效返回 NaN"""
    if any(np.isnan([sp_home, sp_draw, sp_away])):
        return np.nan, np.nan, np.nan
    inv = np.array([1.0 / sp_home, 1.0 / sp_draw, 1.0 / sp_away])
    s = inv.sum()
    if s <= 0:
        return np.nan, np.nan, np.nan
    p = inv / s
    return float(p[0]), float(p[1]), float(p[2])


def generate_date_chunks(start_date: str, end_date: str, chunk_days: int = 30) -> List[tuple]:
    """
    将日期范围拆分为多个块，每个块最多chunk_days天
    返回: [(chunk_start, chunk_end), ...]
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    chunks = []
    current = start
    
    while current <= end:
        chunk_end = min(current + timedelta(days=chunk_days-1), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    
    return chunks


# =========================
# 3. 竞彩统一接口：拉取 & 解析
# =========================

def extract_match_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    从统一赛果接口的单场比赛 JSON 里抽取一行数据。
    已按你提供的 matchResult 结构适配：
      - 顶层有 a/d/h + goalLine（让球胜平负盘口）
      - 队名为 homeTeam/allHomeTeam/awayTeam/allAwayTeam
      - 联赛为 leagueId / leagueName / leagueNameAbbr
      - 日期为 matchDate
      - 比分字段暂用 sectionsNo999（若比赛已结束）
    """
    row: Dict[str, Any] = {}

    # ---- 基本信息 ----
    row["match_id"] = rec.get("matchId")
    row["match_num"] = rec.get("matchNum")
    row["match_num_str"] = rec.get("matchNumStr")

    # 开赛日期（目前接口只有日期，没有具体时间）
    row["jc_date"] = rec.get("matchDate")
    row["match_time"] = rec.get("matchDate")  # 如后续发现有具体时间字段可替换

    # 联赛信息
    row["league_id_jc"] = rec.get("leagueId")
    row["league_name_jc"] = rec.get("leagueName")
    row["league_name_abbr_jc"] = rec.get("leagueNameAbbr")

    # 队名（中文）：优先用全称 allHomeTeam/allAwayTeam，其次简称
    row["home_team_cn"] = rec.get("allHomeTeam") or rec.get("homeTeam")
    row["away_team_cn"] = rec.get("allAwayTeam") or rec.get("awayTeam")

    # ---- 比分（当前接口示例里是空字符串，历史完场比赛应有值）----
    # sectionsNo999 是最终比分（例如 "1:2"）
    row["full_score_raw"] = (
        rec.get("sectionsNo999")
        or rec.get("finalScore")
        or rec.get("fullScore")
        or rec.get("score")
    )
    row["half_score_raw"] = rec.get("sectionsNo1") or rec.get("halfScore")

    # ---- 当前让球盘赔率（统一接口中 a/d/h + goalLine）----
    row["hcp_sp_home_now"] = safe_float(rec.get("h"))
    row["hcp_sp_draw_now"] = safe_float(rec.get("d"))
    row["hcp_sp_away_now"] = safe_float(rec.get("a"))
    row["hcp_line_now"] = safe_float(rec.get("goalLine"))

    # 不让球胜平负 SP：统一接口结构里暂未看到，后续从 bonus 接口拿
    row["spf_sp_home_now"] = np.nan
    row["spf_sp_draw_now"] = np.nan
    row["spf_sp_away_now"] = np.nan

    # 其他标志位
    row["bettingSingle"] = rec.get("bettingSingle")
    row["poolStatus"] = rec.get("poolStatus")
    row["matchResultStatus"] = rec.get("matchResultStatus")

    return row


def fetch_jc_matches(start_date: str, end_date: str,
                     page_size: int = 50,
                     max_pages: int = 50) -> pd.DataFrame:
    """
    从统一赛果接口抓取[start_date, end_date]之间的所有比赛。
    已适配结构：
      顶层: errorCode / value / ...
      比赛列表: value.matchResult
    """
    all_rows: List[Dict[str, Any]] = []
    
    # 将日期范围拆分为30天的块
    date_chunks = generate_date_chunks(start_date, end_date, chunk_days=30)
    print(f"  日期范围已拆分为 {len(date_chunks)} 个块（每块最多30天）")
    
    for chunk_start, chunk_end in date_chunks:
        print(f"  正在拉取 {chunk_start} ~ {chunk_end} ...")
        page_no = 1
        
        while page_no <= max_pages:
            params = {
                "matchBeginDate": chunk_start,
                "matchEndDate": chunk_end,
                "leagueId": "",
                "pageSize": page_size,
                "pageNo": page_no,
                "isFix": 0,
                "matchPage": 1,
                "pcOrWap": 1,
            }
            js = get_json(JC_UNIFORM_URL, params=params)

            value = js.get("value") or js.get("data") or js
            match_list = value.get("matchResult")
            if not match_list:
                break

            for rec in match_list:
                all_rows.append(extract_match_row(rec))

            if len(match_list) < page_size:
                break
            page_no += 1
    
    df = pd.DataFrame(all_rows)
    return df


def add_market_probs_uniform(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于统一接口中的当前让球盘 h/d/a + goalLine 计算隐含概率：
    - hcp_p_home_now / hcp_p_draw_now / hcp_p_away_now
    空 DataFrame 或缺少相关列时直接原样返回。
    """
    if df is None or df.empty:
        return df

    needed_cols = {"hcp_sp_home_now", "hcp_sp_draw_now", "hcp_sp_away_now"}
    if not needed_cols.issubset(df.columns):
        # 统一接口结构变了或 extract_match_row 没有生成这些列时，先不计算
        return df

    df = df.copy()
    sp_h = df["hcp_sp_home_now"].astype(float)
    sp_d = df["hcp_sp_draw_now"].astype(float)
    sp_a = df["hcp_sp_away_now"].astype(float)

    p_home, p_draw, p_away = [], [], []
    for h, d, a in zip(sp_h, sp_d, sp_a):
        ph, pd_, pa = probs_from_sp(h, d, a)
        p_home.append(ph)
        p_draw.append(pd_)
        p_away.append(pa)

    df["hcp_p_home_now"] = p_home
    df["hcp_p_draw_now"] = p_draw
    df["hcp_p_away_now"] = p_away
    df["hcp_p_home_minus_away_now"] = df["hcp_p_home_now"] - df["hcp_p_away_now"]

    return df


def add_labels_from_score(df: pd.DataFrame,
                          score_col: str = "full_score") -> pd.DataFrame:
    """
    根据比分列（score_col）生成标签：
    - home_goals / away_goals
    - goal_diff / total_goals
    - result_1x2 (H/D/A)
    - ou25_result (大球>2.5=1, 其他=0)
    """
    df = df.copy()
    if score_col not in df.columns:
        return df

    home_goals, away_goals = [], []
    for s in df[score_col]:
        h, a = parse_score(s)
        home_goals.append(h)
        away_goals.append(a)

    df["home_goals"] = home_goals
    df["away_goals"] = away_goals
    df["goal_diff"] = df["home_goals"] - df["away_goals"]
    df["total_goals"] = df["home_goals"] + df["away_goals"]

    conds = [
        df["home_goals"] > df["away_goals"],
        df["home_goals"] == df["away_goals"],
        df["home_goals"] < df["away_goals"],
    ]
    choices = ["H", "D", "A"]
    df["result_1x2"] = np.select(conds, choices, default=None)

    df["ou25_result"] = np.where(df["total_goals"] > 2.5, 1, 0)
    df.loc[df["home_goals"].isna() | df["away_goals"].isna(), "ou25_result"] = np.nan

    return df


# =========================
# 4. 竞彩固定奖金接口（bonus）：赔率历史 & 期望进球
# =========================

def compute_expectation_from_crs(crs_item: Dict[str, Any]) -> Dict[str, float]:
    """
    从 crsList 的一个元素（最后一档比分盘）中，计算：
    - exp_goals_home_cs: 主队期望进球
    - exp_goals_away_cs: 客队期望进球
    - exp_goals_total_cs: 总进球期望
    - var_goals_total_cs: 总进球方差

    仅使用明确比分键：形如 's01s00', 's02s01' 等，忽略 's-1s*' 及 '*f' 标志。
    """
    scores: List[tuple[int, int]] = []
    weights: List[float] = []

    for k, v in crs_item.items():
        if k.endswith("f"):
            continue
        m = re.match(r"^s(\d{2})s(\d{2})$", k)
        if not m:
            continue
        h_goals = int(m.group(1))
        a_goals = int(m.group(2))
        sp = safe_float(v)
        if np.isnan(sp) or sp <= 0:
            continue
        w = 1.0 / sp
        scores.append((h_goals, a_goals))
        weights.append(w)

    if not weights:
        return {
            "exp_goals_home_cs": np.nan,
            "exp_goals_away_cs": np.nan,
            "exp_goals_total_cs": np.nan,
            "var_goals_total_cs": np.nan,
        }

    w = np.array(weights)
    s = np.array(scores)  # N x 2
    w_norm = w / w.sum()

    E = (w_norm[:, None] * s).sum(axis=0)   # [E_H, E_A]
    E_H, E_A = float(E[0]), float(E[1])
    total = s.sum(axis=1)                  # 每个比分的总进球
    E_tot = float((w_norm * total).sum())
    var_tot = float((w_norm * (total - E_tot) ** 2).sum())

    return {
        "exp_goals_home_cs": E_H,
        "exp_goals_away_cs": E_A,
        "exp_goals_total_cs": E_tot,
        "var_goals_total_cs": var_tot,
    }


def compute_expectation_from_ttg(ttg_item: Dict[str, Any]) -> Dict[str, float]:
    """
    从 ttgList 的一个元素（最后一档总进球盘）中，计算：
    - exp_goals_total_ttg: 总进球期望
    - var_goals_total_ttg: 总进球方差

    使用键 s0..s7 的 SP，忽略 sXf 标志。
    """
    totals: List[int] = []
    weights: List[float] = []

    for k, v in ttg_item.items():
        if k.endswith("f"):
            continue
        m = re.match(r"^s(\d)$", k)
        if not m:
            continue
        g = int(m.group(1))
        sp = safe_float(v)
        if np.isnan(sp) or sp <= 0:
            continue
        w = 1.0 / sp
        totals.append(g)
        weights.append(w)

    if not weights:
        return {
            "exp_goals_total_ttg": np.nan,
            "var_goals_total_ttg": np.nan,
        }

    w = np.array(weights)
    g = np.array(totals)
    w_norm = w / w.sum()

    E = float((w_norm * g).sum())
    var = float((w_norm * (g - E) ** 2).sum())
    return {
        "exp_goals_total_ttg": E,
        "var_goals_total_ttg": var,
    }


def fetch_fixed_bonus_for_match(match_id: int) -> Dict[str, Any]:
    """
    调用 getFixedBonusV1.qry，解析出：
    - 胜平负开盘/收盘 SP + 隐含概率
    - 让球胜平负开盘/收盘 SP + 盘口 + 隐含概率
    - 比分盘隐含期望进球（exp_goals_home_cs / away_cs / total_cs / var_total_cs）
    - 总进球盘隐含期望进球（exp_goals_total_ttg / var_goals_total_ttg）
    - 最终比分 bonus_score (sectionsNo999)
    """
    params = {"clientCode": "3001", "matchId": match_id}
    js = get_json(JC_BONUS_URL, params=params)
    val = js.get("value") or {}

    out: Dict[str, Any] = {"match_id": match_id}
    out["bonus_isCancel"] = val.get("isCancel")

    odds_hist = val.get("oddsHistory") or {}

    # ---------- 胜平负 hadList ----------
    had_list = odds_hist.get("hadList") or []
    if had_list:
        open_had = had_list[0]
        close_had = had_list[-1]

        # 开盘 SP
        out["spf_sp_home_open"] = safe_float(open_had.get("h"))
        out["spf_sp_draw_open"] = safe_float(open_had.get("d"))
        out["spf_sp_away_open"] = safe_float(open_had.get("a"))
        # 收盘 SP
        out["spf_sp_home_close"] = safe_float(close_had.get("h"))
        out["spf_sp_draw_close"] = safe_float(close_had.get("d"))
        out["spf_sp_away_close"] = safe_float(close_had.get("a"))

        # 开盘隐含概率
        ph, pd_, pa = probs_from_sp(
            out["spf_sp_home_open"],
            out["spf_sp_draw_open"],
            out["spf_sp_away_open"],
        )
        out["spf_p_home_open"] = ph
        out["spf_p_draw_open"] = pd_
        out["spf_p_away_open"] = pa

        # 收盘隐含概率
        ph, pd_, pa = probs_from_sp(
            out["spf_sp_home_close"],
            out["spf_sp_draw_close"],
            out["spf_sp_away_close"],
        )
        out["spf_p_home_close"] = ph
        out["spf_p_draw_close"] = pd_
        out["spf_p_away_close"] = pa
        out["spf_p_home_minus_away_close"] = (
            out["spf_p_home_close"] - out["spf_p_away_close"]
        )
    else:
        for k in [
            "spf_sp_home_open", "spf_sp_draw_open", "spf_sp_away_open",
            "spf_sp_home_close", "spf_sp_draw_close", "spf_sp_away_close",
            "spf_p_home_open", "spf_p_draw_open", "spf_p_away_open",
            "spf_p_home_close", "spf_p_draw_close", "spf_p_away_close",
            "spf_p_home_minus_away_close",
        ]:
            out[k] = np.nan

    # ---------- 让球胜平负 hhadList ----------
    hhad_list = odds_hist.get("hhadList") or []
    if hhad_list:
        open_hh = hhad_list[0]
        close_hh = hhad_list[-1]

        out["hcp_sp_home_open"] = safe_float(open_hh.get("h"))
        out["hcp_sp_draw_open"] = safe_float(open_hh.get("d"))
        out["hcp_sp_away_open"] = safe_float(open_hh.get("a"))
        out["hcp_line_open"] = safe_float(open_hh.get("goalLine"))

        out["hcp_sp_home_close"] = safe_float(close_hh.get("h"))
        out["hcp_sp_draw_close"] = safe_float(close_hh.get("d"))
        out["hcp_sp_away_close"] = safe_float(close_hh.get("a"))
        out["hcp_line_close"] = safe_float(close_hh.get("goalLine"))

        # 开盘隐含概率
        ph, pd_, pa = probs_from_sp(
            out["hcp_sp_home_open"],
            out["hcp_sp_draw_open"],
            out["hcp_sp_away_open"],
        )
        out["hcp_p_home_open"] = ph
        out["hcp_p_draw_open"] = pd_
        out["hcp_p_away_open"] = pa

        # 收盘隐含概率
        ph, pd_, pa = probs_from_sp(
            out["hcp_sp_home_close"],
            out["hcp_sp_draw_close"],
            out["hcp_sp_away_close"],
        )
        out["hcp_p_home_close"] = ph
        out["hcp_p_draw_close"] = pd_
        out["hcp_p_away_close"] = pa
        out["hcp_p_home_minus_away_close"] = (
            out["hcp_p_home_close"] - out["hcp_p_away_close"]
        )
    else:
        for k in [
            "hcp_sp_home_open", "hcp_sp_draw_open", "hcp_sp_away_open",
            "hcp_line_open",
            "hcp_sp_home_close", "hcp_sp_draw_close", "hcp_sp_away_close",
            "hcp_line_close",
            "hcp_p_home_open", "hcp_p_draw_open", "hcp_p_away_open",
            "hcp_p_home_close", "hcp_p_draw_close", "hcp_p_away_close",
            "hcp_p_home_minus_away_close",
        ]:
            out[k] = np.nan

    # ---------- 比分盘 crsList（用最后一档） ----------
    crs_list = odds_hist.get("crsList") or []
    if crs_list:
        crs_last = crs_list[-1]
        out.update(compute_expectation_from_crs(crs_last))
    else:
        out.update({
            "exp_goals_home_cs": np.nan,
            "exp_goals_away_cs": np.nan,
            "exp_goals_total_cs": np.nan,
            "var_goals_total_cs": np.nan,
        })

    # ---------- 总进球盘 ttgList（用最后一档） ----------
    ttg_list = odds_hist.get("ttgList") or []
    if ttg_list:
        ttg_last = ttg_list[-1]
        out.update(compute_expectation_from_ttg(ttg_last))
    else:
        out.update({
            "exp_goals_total_ttg": np.nan,
            "var_goals_total_ttg": np.nan,
        })

    # ---------- 最终比分（正确比分） ----------
    out["bonus_score"] = val.get("sectionsNo999")

    return out


def fetch_fixed_bonus_for_matches(match_ids: List[int], max_workers: int = 10) -> pd.DataFrame:
    """
    批量为多场比赛拉取 bonus 特征。
    使用并发提高下载速度。
    """
    rows: List[Dict[str, Any]] = []
    
    # 使用 ThreadPoolExecutor 实现并发
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_mid = {executor.submit(fetch_fixed_bonus_for_match, mid): mid for mid in match_ids}
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_mid), total=len(match_ids), desc="  下载赔率详情"):
            mid = future_to_mid[future]
            try:
                row = future.result()
                rows.append(row)
            except Exception as e:
                rows.append({"match_id": mid, "bonus_error": str(e)})
    
    return pd.DataFrame(rows)


# =========================
# 5. fbref：加载 & 构造球队特征
# =========================

def load_fbref_csvs(base_dir: pathlib.Path,
                    season_dir: str) -> Dict[str, pd.DataFrame]:
    """
    从 Global_Football_Stats 下所有联赛目录中，汇总各类 squads_* CSV。
    返回 dict: {name: DataFrame}
    """
    leagues = [d for d in base_dir.iterdir() if d.is_dir()]
    dfs_standard, dfs_home_away, dfs_defense, dfs_passing, dfs_possession = [], [], [], [], []

    for league_dir in leagues:
        sdir = league_dir / season_dir
        if not sdir.exists():
            continue

        f_standard = sdir / "stats_squads_standard.csv"
        f_home_away = sdir / "stats_squads_home_away.csv"
        f_defense = sdir / "stats_squads_defense.csv"
        f_passing = sdir / "stats_squads_passing.csv"
        f_possession = sdir / "stats_squads_possession.csv"

        if f_standard.exists():
            dfs_standard.append(pd.read_csv(f_standard))

        if f_home_away.exists():
            dfs_home_away.append(pd.read_csv(f_home_away))

        if f_defense.exists():
            dfs_defense.append(pd.read_csv(f_defense))

        if f_passing.exists():
            dfs_passing.append(pd.read_csv(f_passing))

        if f_possession.exists():
            dfs_possession.append(pd.read_csv(f_possession))

    out: Dict[str, pd.DataFrame] = {}
    if dfs_standard:
        out["standard"] = pd.concat(dfs_standard, ignore_index=True)
    if dfs_home_away:
        out["home_away"] = pd.concat(dfs_home_away, ignore_index=True)
    if dfs_defense:
        out["defense"] = pd.concat(dfs_defense, ignore_index=True)
    if dfs_passing:
        out["passing"] = pd.concat(dfs_passing, ignore_index=True)
    if dfs_possession:
        out["possession"] = pd.concat(dfs_possession, ignore_index=True)
    return out


def build_pca_factor(df: pd.DataFrame,
                     cols: List[str],
                     factor_name: str) -> pd.DataFrame:
    """
    对指定列做：
    1）按 Squad 分组（求均值），避免一个队多行导致 merge 爆炸
    2）标准化 + PCA(1)，生成一个因子列 factor_name
    返回：两列 [Squad, factor_name]
    """
    if "Squad" not in df.columns:
        raise ValueError("DataFrame 中不存在 'Squad' 列，无法构建因子。")

    use_cols = ["Squad"] + [c for c in cols if c in df.columns]
    sub = df[use_cols].copy()

    # 数值化
    num_cols = [c for c in use_cols if c != "Squad"]
    for c in num_cols:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    # 先按 Squad 聚合成一行
    sub = sub.groupby("Squad", as_index=False).mean(numeric_only=True)

    if not num_cols:
        # 没有可用数值列时，返回 NaN 因子
        sub[factor_name] = np.nan
        return sub[["Squad", factor_name]]

    X = sub[num_cols].values
    # 用列均值填补少量缺失
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=1)
    comp = pca.fit_transform(X_scaled).ravel()
    sub[factor_name] = comp

    return sub[["Squad", factor_name]]


def build_fbref_team_features() -> pd.DataFrame:
    """
    汇总并构造每支球队的 fbref 特征（每个 Squad 只保留一行）：
    - 赛季 xG / xGA / Pts
    - 主客场 xG / xGA
    - 防守强度因子 def_strength
    - 进攻组织因子 off_buildup
    - 节奏/控场因子 tempo
    """
    all_dfs = load_fbref_csvs(FBREF_BASE_DIR, FBREF_SEASON_DIR)

    # ---- 标准：xG, xGA, Pts ----
    if "standard" in all_dfs:
        std = all_dfs["standard"].copy()
        if "Squad" not in std.columns:
            raise ValueError("stats_squads_standard 中找不到 'Squad' 列")

        cols = [c for c in ["GF", "GA", "xG", "xGA", "Pts"] if c in std.columns]
        use_cols = ["Squad"] + cols
        std = std[use_cols].copy()
        for c in cols:
            std[c] = pd.to_numeric(std[c], errors="coerce")

        # 按 Squad 分组聚合，防止多个联赛/重复行
        fb = std.groupby("Squad", as_index=False).mean(numeric_only=True)
    else:
        fb = pd.DataFrame(columns=["Squad"])

    # ---- 主客场 xG / xGA ----
    if "home_away" in all_dfs:
        ha = all_dfs["home_away"].copy()
        if "Squad" in ha.columns:
            cols = [c for c in ["Home.xG", "Home.xGA", "Away.xG", "Away.xGA"] if c in ha.columns]
            use_cols = ["Squad"] + cols
            ha = ha[use_cols].copy()
            for c in cols:
                ha[c] = pd.to_numeric(ha[c], errors="coerce")
            ha = ha.groupby("Squad", as_index=False).mean(numeric_only=True)
            fb = fb.merge(ha, on="Squad", how="left")

    # ---- 防守因子 ----
    if "defense" in all_dfs:
        df_def = all_dfs["defense"].copy()
        if "Squad" in df_def.columns:
            def_fac = build_pca_factor(df_def, ["TklW", "Lost", "Blocks", "Clr"], "def_strength")
            fb = fb.merge(def_fac, on="Squad", how="left")

    # ---- 进攻组织因子 ----
    if "passing" in all_dfs:
        df_pas = all_dfs["passing"].copy()
        if "Squad" in df_pas.columns:
            off_fac = build_pca_factor(df_pas, ["xAG", "PrgP", "KP"], "off_buildup")
            fb = fb.merge(off_fac, on="Squad", how="left")

    # ---- 节奏/控场因子 ----
    if "possession" in all_dfs:
        df_pos = all_dfs["possession"].copy()
        if "Squad" in df_pos.columns:
            tempo_fac = build_pca_factor(df_pos, ["Poss", "PrgC", "G+A-PK"], "tempo")
            fb = fb.merge(tempo_fac, on="Squad", how="left")

    # 最终再去一次重，确保每个 Squad 一行
    fb = fb.drop_duplicates(subset=["Squad"]).reset_index(drop=True)
    return fb


# =========================
# 6. 合并：竞彩 + team_map + fbref
# =========================

def load_team_map(map_path: pathlib.Path) -> Dict[str, str]:
    """
    读取 team_map备份.csv，返回 中文 → 英文fbref 的映射 dict。
    文件中应包含逻辑列：team_cn_sporttery, team_en_fbref
    实际文件列名可能带空格或反斜杠，这里统一做标准化。
    """
    if not map_path.exists():
        raise FileNotFoundError(f"team_map 文件不存在: {map_path}")

    # 用前面已经试验成功的编码（你那次打印列名时的编码）
    tm = pd.read_csv(map_path, encoding="gbk")

    # 关键一步：标准化列名，去掉前后空格和反斜杠
    tm.columns = [c.strip().replace("\\", "") for c in tm.columns]
    print("标准化后的 team_map 列名：", list(tm.columns))

    cn_col = "team_cn_sporttery"
    en_col = "team_en_fbref"

    if cn_col not in tm.columns or en_col not in tm.columns:
        raise ValueError(
            f"team_map 文件中找不到列: {cn_col}, {en_col}，当前列名为: {list(tm.columns)}"
        )

    return dict(zip(tm[cn_col], tm[en_col]))


def merge_jc_fbref(jc_df: pd.DataFrame,
                   fbref_df: pd.DataFrame,
                   team_map_cn2en: Dict[str, str]) -> pd.DataFrame:
    """
    把 竞彩比赛 jc_df 与 fbref 队伍特征 fbref_df 合到一起。
    """
    jc_df = jc_df.copy()
    jc_df["home_team_fbref"] = jc_df["home_team_cn"].map(team_map_cn2en)
    jc_df["away_team_fbref"] = jc_df["away_team_cn"].map(team_map_cn2en)

    # fbref 特征分别加主队/客队前缀
    fb_home = fbref_df.add_prefix("home_")
    fb_home = fb_home.rename(columns={"home_Squad": "home_team_fbref"})

    fb_away = fbref_df.add_prefix("away_")
    fb_away = fb_away.rename(columns={"away_Squad": "away_team_fbref"})

    df = jc_df.merge(fb_home, on="home_team_fbref", how="left")
    df = df.merge(fb_away, on="away_team_fbref", how="left")

    # 构造一些差值特征
    if "home_xG" in df.columns and "away_xG" in df.columns:
        df["strength_xg_diff"] = df["home_xG"] - df["away_xG"]
    if "home_def_strength" in df.columns and "away_def_strength" in df.columns:
        df["strength_def_diff"] = df["home_def_strength"] - df["away_def_strength"]
    if "home_off_buildup" in df.columns and "away_off_buildup" in df.columns:
        df["strength_offbuild_diff"] = df["home_off_buildup"] - df["away_off_buildup"]
    if "home_tempo" in df.columns and "away_tempo" in df.columns:
        df["tempo_diff"] = df["home_tempo"] - df["away_tempo"]

    return df


def build_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    # 1) 竞彩统一接口：比赛列表 + 当前盘口
    print(f"拉取竞彩比赛 {start_date} ~ {end_date} ...")
    jc = fetch_jc_matches(start_date, end_date)
    print(f"  共 {len(jc)} 场（行）")

    if jc.empty:
        print("当前日期范围内无竞彩比赛数据，直接返回空 DataFrame。")
        return jc  # 后面 fbref、bonus 就不再跑了

    jc = add_market_probs_uniform(jc)

    # 2) fbref：球队特征
    print("加载 fbref 球队特征 ...")
    fb = build_fbref_team_features()
    print(f"  共 {len(fb)} 支球队")

    # 3) team_map 映射并合并
    print("读取 team_map 并合表(竞彩 + fbref) ...")
    cn2en = load_team_map(TEAM_MAP_PATH)
    base = merge_jc_fbref(jc, fb, cn2en)

    # 4) bonus 接口：开盘/收盘赔率 + 期望进球
    match_ids = base["match_id"].dropna().unique().tolist()
    print(f"调用 bonus 接口获取 {len(match_ids)} 场比赛的赔率历史 ...")
    
    # 使用并发下载，可调整 MAX_WORKERS
    bonus_df = fetch_fixed_bonus_for_matches(match_ids, max_workers=MAX_WORKERS)

    merged = base.merge(bonus_df, on="match_id", how="left")

    # 5) 统一比分字段 + 标签
    # 优先使用 bonus 接口返回的最终比分
    merged["full_score"] = merged["bonus_score"].fillna(merged["full_score_raw"])
    merged = add_labels_from_score(merged, score_col="full_score")
    
    return merged


# =========================
# 7. 主入口
# =========================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = build_dataset(START_DATE, END_DATE)

    out_file = OUTPUT_DIR / f"jc_fbref_bonus_{START_DATE}_to_{END_DATE}.csv"
    df_all.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"已保存综合表(含 fbref + bonus 特征)到: {out_file}")
    print(f"数据概览：{len(df_all)} 行，{len(df_all.columns)} 列")
    print(df_all.head())