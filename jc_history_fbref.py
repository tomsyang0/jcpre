#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
竞彩 API + fbref 球队统计 + 支持率 + 比赛特征 → 预测用综合表
2025-01-01 至今，30 天分片 + 并发详情 + 支持率 + 比赛特征
"""
import os, sys, time, datetime as dt, pathlib, requests, json, re
from requests.adapters import HTTPAdapter
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------- 0. 路径/常量 ----------
JC_UNIFORM_URL = "https://webapi.sporttery.cn/gateway/uniform/football/getUniformMatchResultV1.qry"
JC_BONUS_URL   = "https://webapi.sporttery.cn/gateway/uniform/football/getFixedBonusV1.qry"
JC_SUPPORT_URL = "https://webapi.sporttery.cn/gateway/jc/common/getSupportRateV1.qry"
JC_FEATURE_URL = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchFeatureV1.qry"  # 新增特征接口

FBREF_BASE_DIR   = pathlib.Path("Global_Football_Stats")
FBREF_SEASON_DIR = "2025-2026"
TEAM_MAP_PATH    = pathlib.Path("team_map备份.csv")
OUTPUT_DIR       = pathlib.Path("datasets")

START_DATE = "2025-01-01"
END_DATE   = dt.date.today().strftime("%Y-%m-%d")

BONUS_WORKERS   = 5
SUPPORT_WORKERS = 8
FEATURE_WORKERS = 5  # 新增特征接口并发数
RETRY           = 3

# ---------- 1. 通用工具 ----------
def safe_float(x: Any) -> float:
    try:
        return float(x) if x not in (None, "") else np.nan
    except Exception:
        return np.nan

def parse_score(score: Any):
    if not isinstance(score, str):
        return None, None
    m = re.search(r"(\d+)\D+(\d+)", score)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)

def probs_from_sp(h, d, a):
    if any(np.isnan([h, d, a])):
        return np.nan, np.nan, np.nan
    inv = np.array([1.0 / h, 1.0 / d, 1.0 / a])
    s = inv.sum()
    return tuple(inv / s) if s > 0 else (np.nan, np.nan, np.nan)

# ---------- 2. 会话 ----------
def make_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://www.sporttery.cn/",
        "Origin": "https://www.sporttery.cn/",
    })
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=128, max_retries=RETRY)
    sess.mount("https://", adapter)
    return sess

SESSION = make_session()

def get_json(url, params=None, timeout=10):
    resp = SESSION.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# ---------- 3. 比赛列表 ----------
def extract_match_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "match_id": rec.get("matchId"),
        "match_num": rec.get("matchNum"),
        "jc_date": rec.get("matchDate"),
        "league_name_jc": rec.get("leagueName"),
        "home_team_cn": rec.get("allHomeTeam") or rec.get("homeTeam"),
        "away_team_cn": rec.get("allAwayTeam") or rec.get("awayTeam"),
        "full_score_raw": rec.get("sectionsNo999") or rec.get("finalScore") or "",
        "half_score_raw": rec.get("sectionsNo1") or "",
        "hcp_sp_home_now": safe_float(rec.get("h")),
        "hcp_sp_draw_now": safe_float(rec.get("d")),
        "hcp_sp_away_now": safe_float(rec.get("a")),
        "hcp_line_now": safe_float(rec.get("goalLine")),
        "bettingSingle": rec.get("bettingSingle"),
        "poolStatus": rec.get("poolStatus"),
    }

def fetch_jc_matches_slice(start: str, end: str, page_size=50, max_pages=50) -> List[Dict[str, Any]]:
    rows, page = [], 1
    while page <= max_pages:
        params = {
            "matchBeginDate": start,
            "matchEndDate": end,
            "leagueId": "",
            "pageSize": page_size,
            "pageNo": page,
            "isFix": 0,
            "matchPage": 1,
            "pcOrWap": 1,
        }
        js = get_json(JC_UNIFORM_URL, params=params)
        lst = (js.get("value") or {}).get("matchResult") or []
        if not lst:
            break
        rows.extend([extract_match_row(r) for r in lst])
        if len(lst) < page_size:
            break
        page += 1
    return rows

def fetch_all_matches(start_date: str, end_date: str) -> pd.DataFrame:
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end   = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    delta = dt.timedelta(days=29)
    all_rows = []
    while start <= end:
        slice_end = min(start + delta, end)
        print(f"拉取比赛列表 {start} ~ {slice_end}")
        rows = fetch_jc_matches_slice(start.strftime("%Y-%m-%d"), slice_end.strftime("%Y-%m-%d"))
        all_rows.extend(rows)
        start = slice_end + dt.timedelta(days=1)
    df = pd.DataFrame(all_rows)
    if not df.empty and {"hcp_sp_home_now", "hcp_sp_draw_now", "hcp_sp_away_now"}.issubset(df.columns):
        sp = df[["hcp_sp_home_now", "hcp_sp_draw_now", "hcp_sp_away_now"]].astype(float)
        probs = np.stack([probs_from_sp(h, d, a) for h, d, a in sp.itertuples(index=False)])
        df["hcp_p_home_now"], df["hcp_p_draw_now"], df["hcp_p_away_now"] = probs.T
        df["hcp_p_home_minus_away_now"] = df["hcp_p_home_now"] - df["hcp_p_away_now"]
    return df

# ---------- 4. bonus ----------
def fetch_bonus_single(match_id: int) -> Dict[str, Any]:
    for _ in range(RETRY):
        try:
            params = {"clientCode": "3001", "matchId": match_id}
            js = get_json(JC_BONUS_URL, params=params)
            val = js.get("value") or {}
            out = {"match_id": match_id, "bonus_isCancel": val.get("isCancel")}
            # hadList
            had = (val.get("oddsHistory") or {}).get("hadList") or []
            if had:
                open_h, close_h = had[0], had[-1]
                out.update({
                    "spf_sp_home_open":  safe_float(open_h.get("h")),
                    "spf_sp_draw_open":  safe_float(open_h.get("d")),
                    "spf_sp_away_open":  safe_float(open_h.get("a")),
                    "spf_sp_home_close": safe_float(close_h.get("h")),
                    "spf_sp_draw_close": safe_float(close_h.get("d")),
                    "spf_sp_away_close": safe_float(close_h.get("a")),
                })
                ph, pd, pa = probs_from_sp(out["spf_sp_home_close"], out["spf_sp_draw_close"], out["spf_sp_away_close"])
                out.update({"spf_p_home_close": ph, "spf_p_draw_close": pd, "spf_p_away_close": pa,
                            "spf_p_home_minus_away_close": ph - pa})
            # hhadList
            hhad = (val.get("oddsHistory") or {}).get("hhadList") or []
            if hhad:
                open_hh, close_hh = hhad[0], hhad[-1]
                out.update({
                    "hcp_sp_home_open":  safe_float(open_hh.get("h")),
                    "hcp_sp_draw_open":  safe_float(open_hh.get("d")),
                    "hcp_sp_away_open":  safe_float(open_hh.get("a")),
                    "hcp_line_open":     safe_float(open_hh.get("goalLine")),
                    "hcp_sp_home_close": safe_float(close_hh.get("h")),
                    "hcp_sp_draw_close": safe_float(close_hh.get("d")),
                    "hcp_sp_away_close": safe_float(close_hh.get("a")),
                    "hcp_line_close":    safe_float(close_hh.get("goalLine")),
                })
                ph, pd, pa = probs_from_sp(out["hcp_sp_home_close"], out["hcp_sp_draw_close"], out["hcp_sp_away_close"])
                out.update({"hcp_p_home_close": ph, "hcp_p_draw_close": pd, "hcp_p_away_close": pa,
                            "hcp_p_home_minus_away_close": ph - pa})
            out["bonus_score"] = val.get("sectionsNo999")
            return out
        except Exception as e:
            time.sleep(0.5)
    return {"match_id": match_id, "bonus_error": "max_retry"}

def fetch_bonus_concurrent(match_ids: List[int]) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=BONUS_WORKERS) as exe:
        fut_map = {exe.submit(fetch_bonus_single, mid): mid for mid in match_ids}
        for fut in as_completed(fut_map):
            rows.append(fut.result())
            time.sleep(0.05)
    return pd.DataFrame(rows)

# ---------- 5. 支持率 ----------
def parse_support_item(d: Dict[str, Any]) -> Dict[str, float]:
    """把 HAD/HHAD 里的支持率/概率/误差 拆成平铺字段"""
    out = {}
    for tp in ["HAD", "HHAD"]:
        if tp not in d:
            continue
        item = d[tp]
        prefix = "had_" if tp == "HAD" else "hhad_"
        out.update({
            f"{prefix}support_h": safe_float(item.get("hSupportRate").strip("%")),
            f"{prefix}support_d": safe_float(item.get("dSupportRate").strip("%")),
            f"{prefix}support_a": safe_float(item.get("aSupportRate").strip("%")),
            f"{prefix}prob_h": safe_float(item.get("hProbability").strip("%")),
            f"{prefix}prob_d": safe_float(item.get("dProbability").strip("%")),
            f"{prefix}prob_a": safe_float(item.get("aProbability").strip("%")),
            f"{prefix}psy_err": int(item.get("psyError") or 0),
        })
    return out

def fetch_support_single(match_id: int) -> Dict[str, Any]:
    """单场比赛支持率"""
    for _ in range(RETRY):
        try:
            params = {"matchIds": match_id}
            js = get_json(JC_SUPPORT_URL, params=params)
            val = js.get("value") or {}
            key = f"_{match_id}"
            if key in val:
                out = parse_support_item(val[key])
                out["match_id"] = match_id
                return out
            return {"match_id": match_id}
        except Exception as e:
            time.sleep(0.5)
    return {"match_id": match_id, "support_error": "max_retry"}

def fetch_support_concurrent(match_ids: List[int]) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=SUPPORT_WORKERS) as exe:
        fut_map = {exe.submit(fetch_support_single, mid): mid for mid in match_ids}
        for fut in as_completed(fut_map):
            rows.append(fut.result())
            time.sleep(0.02)
    return pd.DataFrame(rows)

# ---------- 6. 比赛特征（新增） ----------
def safe_float_parse(x: Any) -> float:
    """增强版安全浮点转换，处理字符串格式的数值"""
    if x is None or x == "":
        return np.nan
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan

def fetch_feature_single(match_id: int) -> Dict[str, Any]:
    """单场比赛特征数据（适配实际接口返回格式）"""
    for _ in range(RETRY):
        try:
            params = {
                "termLimits": 10,  # 获取最近10场比赛数据
                "sportteryMatchId": match_id
            }
            js = get_json(JC_FEATURE_URL, params=params)
            
            # 检查接口返回是否成功
            if not js.get("success") or js.get("errorCode") != "0":
                raise ValueError(f"接口返回错误: {js.get('errorMessage')}")
            
            val = js.get("value") or {}
            out = {"match_id": match_id}
            
            # 提取基础信息
            out["home_team_short_name"] = val.get("homeTeamShortName", "")
            out["away_team_short_name"] = val.get("awayTeamShortName", "")
            
            # 1. 进球平均值特征
            goal_avg = val.get("goalAvg", {})
            out.update({
                "home_goal_avg_cnt": safe_float_parse(goal_avg.get("homeGoalAvgCnt")),
                "away_goal_avg_cnt": safe_float_parse(goal_avg.get("awayGoalAvgCnt")),
                "home_goal_avg_ratio": safe_float_parse(goal_avg.get("homeGoalAvgCntRatio")),
                "away_goal_avg_ratio": safe_float_parse(goal_avg.get("awayGoalAvgCntRatio")),
            })
            
            # 2. 失球平均值特征
            loss_goal_avg = val.get("lossGoalAvg", {})
            out.update({
                "home_loss_goal_avg_cnt": safe_float_parse(loss_goal_avg.get("homeLossGoalAvgCnt")),
                "away_loss_goal_avg_cnt": safe_float_parse(loss_goal_avg.get("awayLossGoalAvgCnt")),
                "home_loss_goal_avg_ratio": safe_float_parse(loss_goal_avg.get("homeLossGoalAvgCntRatio")),
                "away_loss_goal_avg_ratio": safe_float_parse(loss_goal_avg.get("awayLossGoalAvgCntRatio")),
            })
            
            # 3. 主客场胜负平统计（eachHomeAway）
            each_home_away = val.get("eachHomeAway", {})
            out.update({
                "home_win_cnt": each_home_away.get("homeWinGoalMatchCnt", 0),
                "home_draw_cnt": each_home_away.get("homeDrawMatchCnt", 0),
                "home_loss_cnt": each_home_away.get("homeLossGoalMatchCnt", 0),
                "away_win_cnt": each_home_away.get("awayWinGoalMatchCnt", 0),
                "away_draw_cnt": each_home_away.get("awayDrawMatchCnt", 0),
                "away_loss_cnt": each_home_away.get("awayLossGoalMatchCnt", 0),
                "home_score_ratio": safe_float_parse(each_home_away.get("homeScoreRatio")),
                "away_score_ratio": safe_float_parse(each_home_away.get("awayScoreRatio")),
                "total_leg_cnt": each_home_away.get("totalLegCnt", 0),
            })
            
            # 4. 相同主客场统计（eachSameHomeAway）
            each_same_home_away = val.get("eachSameHomeAway", {})
            out.update({
                "same_home_win_cnt": each_same_home_away.get("homeWinGoalMatchCnt", 0),
                "same_home_draw_cnt": each_same_home_away.get("homeDrawMatchCnt", 0),
                "same_home_loss_cnt": each_same_home_away.get("homeLossGoalMatchCnt", 0),
                "same_away_win_cnt": each_same_home_away.get("awayWinGoalMatchCnt", 0),
                "same_away_draw_cnt": each_same_home_away.get("awayDrawMatchCnt", 0),
                "same_away_loss_cnt": each_same_home_away.get("awayLossGoalMatchCnt", 0),
                "same_home_score_ratio": safe_float_parse(each_same_home_away.get("homeScoreRatio")),
                "same_away_score_ratio": safe_float_parse(each_same_home_away.get("awayScoreRatio")),
            })
            
            # 5. 计算衍生特征
            # 主队近期胜率
            home_total = out["home_win_cnt"] + out["home_draw_cnt"] + out["home_loss_cnt"]
            out["home_win_rate"] = out["home_win_cnt"] / home_total if home_total > 0 else 0.0
            
            # 客队近期胜率
            away_total = out["away_win_cnt"] + out["away_draw_cnt"] + out["away_loss_cnt"]
            out["away_win_rate"] = out["away_win_cnt"] / away_total if away_total > 0 else 0.0
            
            # 主客场进球差
            out["home_away_goal_diff"] = out["home_goal_avg_cnt"] - out["away_goal_avg_cnt"]
            
            # 主客场失球差
            out["home_away_loss_goal_diff"] = out["home_loss_goal_avg_cnt"] - out["away_loss_goal_avg_cnt"]
            
            return out
        
        except Exception as e:
            print(f"获取特征失败 match_id={match_id}: {str(e)}")
            time.sleep(0.5)
    
    # 返回带错误标记的空数据
    return {
        "match_id": match_id,
        "feature_error": "max_retry",
        "home_team_short_name": "",
        "away_team_short_name": "",
        "home_goal_avg_cnt": np.nan,
        "away_goal_avg_cnt": np.nan,
        "home_goal_avg_ratio": np.nan,
        "away_goal_avg_ratio": np.nan,
        "home_loss_goal_avg_cnt": np.nan,
        "away_loss_goal_avg_cnt": np.nan,
        "home_loss_goal_avg_ratio": np.nan,
        "away_loss_goal_avg_ratio": np.nan,
        "home_win_cnt": 0,
        "home_draw_cnt": 0,
        "home_loss_cnt": 0,
        "away_win_cnt": 0,
        "away_draw_cnt": 0,
        "away_loss_cnt": 0,
        "home_score_ratio": np.nan,
        "away_score_ratio": np.nan,
        "total_leg_cnt": 0,
        "same_home_win_cnt": 0,
        "same_home_draw_cnt": 0,
        "same_home_loss_cnt": 0,
        "same_away_win_cnt": 0,
        "same_away_draw_cnt": 0,
        "same_away_loss_cnt": 0,
        "same_home_score_ratio": np.nan,
        "same_away_score_ratio": np.nan,
        "home_win_rate": 0.0,
        "away_win_rate": 0.0,
        "home_away_goal_diff": np.nan,
        "home_away_loss_goal_diff": np.nan,
    }

def fetch_feature_concurrent(match_ids: List[int]) -> pd.DataFrame:
    """并发获取比赛特征"""
    rows = []
    with ThreadPoolExecutor(max_workers=FEATURE_WORKERS) as exe:
        fut_map = {exe.submit(fetch_feature_single, mid): mid for mid in match_ids}
        for fut in as_completed(fut_map):
            rows.append(fut.result())
            time.sleep(0.05)  # 轻微延迟避免请求过快
    return pd.DataFrame(rows)

# ---------- 7. fbref 特征 ----------
def load_fbref_csvs(base_dir: pathlib.Path, season_dir: str) -> Dict[str, pd.DataFrame]:
    leagues = [d for d in base_dir.iterdir() if d.is_dir()]
    buckets = {"standard": [], "home_away": [], "defense": [], "passing": [], "possession": []}
    for league_dir in leagues:
        sdir = league_dir / season_dir
        if not sdir.exists():
            continue
        for k, f in [("standard", "stats_squads_standard.csv"),
                     ("home_away", "stats_squads_home_away.csv"),
                     ("defense", "stats_squads_defense.csv"),
                     ("passing", "stats_squads_passing.csv"),
                     ("possession", "stats_squads_possession.csv")]:
            fp = sdir / f
            if fp.exists():
                buckets[k].append(pd.read_csv(fp))
    return {k: pd.concat(v, ignore_index=True) for k, v in buckets.items() if v}

def build_pca_factor(df: pd.DataFrame, cols: List[str], name: str) -> pd.DataFrame:
    use = ["Squad"] + [c for c in cols if c in df.columns]
    sub = df[use].copy()
    for c in use[1:]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub = sub.groupby("Squad", as_index=False).mean(numeric_only=True)
    if not use[1:]:
        sub[name] = np.nan
        return sub[["Squad", name]]
    X = sub[use[1:]].values
    col_means = np.nanmean(X, axis=0)
    X[np.isnan(X)] = np.take(col_means, np.where(np.isnan(X))[1])
    X = StandardScaler().fit_transform(X)
    comp = PCA(n_components=1).fit_transform(X).ravel()
    sub[name] = comp
    return sub[["Squad", name]]

def build_fbref_team_features() -> pd.DataFrame:
    all_dfs = load_fbref_csvs(FBREF_BASE_DIR, FBREF_SEASON_DIR)
    std = all_dfs.get("standard")
    if std is None or "Squad" not in std.columns:
        raise ValueError("standard 表缺失或不含 Squad 列")
    fb = std.groupby("Squad", as_index=False)[["GF", "GA", "xG", "xGA", "Pts"]].mean()
    if "home_away" in all_dfs:
        ha = all_dfs["home_away"]
        if "Squad" in ha.columns:
            ha = ha.groupby("Squad", as_index=False)[["Home.xG", "Home.xGA", "Away.xG", "Away.xGA"]].mean()
            fb = fb.merge(ha, on="Squad", how="left")
    for tbl, cols, name in [("defense", ["TklW", "Lost", "Blocks", "Clr"], "def_strength"),
                            ("passing", ["xAG", "PrgP", "KP"], "off_buildup"),
                            ("possession", ["Poss", "PrgC", "G+A-PK"], "tempo")]:
        if tbl in all_dfs:
            df = all_dfs[tbl]
            if "Squad" in df.columns:
                fac = build_pca_factor(df, cols, name)
                fb = fb.merge(fac, on="Squad", how="left")
    return fb.drop_duplicates(subset=["Squad"])

# ---------- 8. 队名映射 ----------
def load_team_map(path: pathlib.Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    tm = pd.read_csv(path, encoding="gbk")
    tm.columns = [c.strip().replace("\\", "") for c in tm.columns]
    return dict(zip(tm["team_cn_sporttery"], tm["team_en_fbref"]))

def merge_jc_fbref(jc: pd.DataFrame, fb: pd.DataFrame, mp: Dict[str, str]) -> pd.DataFrame:
    jc = jc.copy()
    jc["home_team_fbref"] = jc["home_team_cn"].map(mp)
    jc["away_team_fbref"] = jc["away_team_cn"].map(mp)
    fb_home = fb.add_prefix("home_").rename(columns={"home_Squad": "home_team_fbref"})
    fb_away = fb.add_prefix("away_").rename(columns={"away_Squad": "away_team_fbref"})
    df = jc.merge(fb_home, on="home_team_fbref", how="left").merge(fb_away, on="away_team_fbref", how="left")
    # 差值特征
    diff_cols = ["strength_xg_diff", "strength_def_diff", "strength_offbuild_diff", "tempo_diff"]
    for col in diff_cols:
        suffix = col.split("_diff")[0]
        field = suffix.split("_", 1)[-1]
        home_col, away_col = f"home_{field}", f"away_{field}"
        if home_col in df.columns and away_col in df.columns:
            df[col] = df[home_col] - df[away_col]
    return df

# ---------- 9. 标签 ----------
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score_col = df["bonus_score"].fillna(df["full_score_raw"])
    home_goals, away_goals = zip(*[parse_score(s) for s in score_col])
    df["home_goals"] = home_goals
    df["away_goals"] = away_goals
    df["goal_diff"]  = df["home_goals"] - df["away_goals"]
    df["total_goals"]= df["home_goals"] + df["away_goals"]
    df["result_1x2"] = np.select(
        [df["goal_diff"] > 0, df["goal_diff"] == 0, df["goal_diff"] < 0],
        ["H", "D", "A"], default=""
    )
    df["ou25_result"] = np.where(df["total_goals"] > 2.5, 1, 0)
    miss = df["home_goals"].isna() | df["away_goals"].isna()
    df.loc[miss, ["result_1x2", "ou25_result"]] = np.nan
    return df

# ---------- 10. 主流程 ----------
def build_dataset(start: str, end: str) -> pd.DataFrame:
    print("【1】拉取竞彩比赛列表（30 天分片）...")
    jc = fetch_all_matches(start, end)
    if jc.empty:
        print("无比赛，流程结束。")
        return jc
    print(f"  共 {len(jc)} 场")

    print("【2】加载 fbref 球队特征...")
    fb = build_fbref_team_features()
    print(f"  共 {len(fb)} 队")

    print("【3】队名映射并合并...")
    mp = load_team_map(TEAM_MAP_PATH)
    base = merge_jc_fbref(jc, fb, mp)

    print("【4】并发拉取 bonus 详情...")
    mids = base["match_id"].dropna().astype(int).unique().tolist()
    bonus_df = fetch_bonus_concurrent(mids)
    base = base.merge(bonus_df, on="match_id", how="left")

    print("【5】并发拉取支持率...")
    support_df = fetch_support_concurrent(mids)
    base = base.merge(support_df, on="match_id", how="left")

    print("【6】并发拉取比赛特征...")  # 新增步骤
    feature_df = fetch_feature_concurrent(mids)
    base = base.merge(feature_df, on="match_id", how="left")

    print("【7】生成标签...")
    base = add_labels(base)
    return base

# ---------- 11. 入口 ----------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset(START_DATE, END_DATE)
    out_file = OUTPUT_DIR / f"jc_fbref_bonus_support_feature_{START_DATE}_to_{END_DATE}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"✅ 已保存 -> {out_file}")