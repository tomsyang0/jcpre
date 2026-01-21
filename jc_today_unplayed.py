#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
竞彩当日未开赛比赛数据获取脚本
功能：获取今日未开赛比赛数据，用于预测分析
适配接口：https://webapi.sporttery.cn/gateway/uniform/football/getMatchListV1.qry
"""
import os, sys, time, datetime as dt, pathlib, requests, json, re
from requests.adapters import HTTPAdapter
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 0. 路径/常量 ----------
# 未开赛比赛专用接口
JC_UNPLAYED_URL  = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchListV1.qry"
JC_BONUS_URL     = "https://webapi.sporttery.cn/gateway/uniform/football/getFixedBonusV1.qry"
JC_SUPPORT_URL   = "https://webapi.sporttery.cn/gateway/jc/common/getSupportRateV1.qry"
JC_FEATURE_URL   = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchFeatureV1.qry"

FBREF_BASE_DIR   = pathlib.Path("Global_Football_Stats")
FBREF_SEASON_DIR = "2025-2026"
TEAM_MAP_PATH    = pathlib.Path("team_map备份.csv")
OUTPUT_DIR       = pathlib.Path("predict_data")

# 当日日期
TODAY_DATE = dt.date.today().strftime("%Y-%m-%d")

BONUS_WORKERS   = 5
SUPPORT_WORKERS = 8
FEATURE_WORKERS = 5
RETRY           = 3

# ---------- 1. 通用工具 ----------
def safe_float(x: Any) -> float:
    try:
        return float(x) if x not in (None, "") else np.nan
    except Exception:
        return np.nan

def safe_float_parse(x: Any) -> float:
    """增强版安全浮点转换"""
    if x is None or x == "":
        return np.nan
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan

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

# ---------- 3. 未开赛比赛列表获取（适配新接口） ----------
def extract_odds_info(odds_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从oddsList中提取HHAD和HAD赔率信息"""
    odds_info = {
        "hcp_sp_home_now": np.nan,
        "hcp_sp_draw_now": np.nan,
        "hcp_sp_away_now": np.nan,
        "hcp_line_now": np.nan,
        "spf_sp_home_now": np.nan,
        "spf_sp_draw_now": np.nan,
        "spf_sp_away_now": np.nan,
    }
    
    # 遍历赔率列表，提取HHAD（让球胜平负）和HAD（胜平负）
    for odds in odds_list:
        pool_code = odds.get("poolCode")
        if pool_code == "HHAD":  # 让球胜平负
            odds_info["hcp_sp_home_now"] = safe_float(odds.get("h"))
            odds_info["hcp_sp_draw_now"] = safe_float(odds.get("d"))
            odds_info["hcp_sp_away_now"] = safe_float(odds.get("a"))
            odds_info["hcp_line_now"] = safe_float(odds.get("goalLine"))
        elif pool_code == "HAD":  # 胜平负
            odds_info["spf_sp_home_now"] = safe_float(odds.get("h"))
            odds_info["spf_sp_draw_now"] = safe_float(odds.get("d"))
            odds_info["spf_sp_away_now"] = safe_float(odds.get("a"))
    
    return odds_info

def extract_unplayed_match_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    """提取未开赛比赛数据（适配新接口数据结构）"""
    # 提取赔率信息
    odds_info = extract_odds_info(rec.get("oddsList", []))
    
    # 构建基础字段（保持原有关键字段）
    return {
        "match_id": rec.get("matchId"),
        "match_num": rec.get("matchNum"),
        "match_num_str": rec.get("matchNumStr"),  # 新增：周三001
        "jc_date": rec.get("matchDate"),  # 比赛日期（可能是次日）
        "business_date": rec.get("businessDate"),  # 开售日期
        "match_time": rec.get("matchTime"),  # 比赛时间
        "league_name_jc": rec.get("leagueAllName"),  # 联赛全称
        "league_abb_name": rec.get("leagueAbbName"),  # 联赛简称
        "league_id": rec.get("leagueId"),
        "home_team_cn": rec.get("homeTeamAllName"),  # 主队全称
        "home_team_abb": rec.get("homeTeamAbbName"),  # 主队简称
        "home_team_id": rec.get("homeTeamId"),
        "away_team_cn": rec.get("awayTeamAllName"),  # 客队全称
        "away_team_abb": rec.get("awayTeamAbbName"),  # 客队简称
        "away_team_id": rec.get("awayTeamId"),
        # 未开赛比赛暂无比分
        "full_score_raw": "",
        "half_score_raw": "",
        # 赔率字段（保持原有关键字）
        "hcp_sp_home_now": odds_info["hcp_sp_home_now"],
        "hcp_sp_draw_now": odds_info["hcp_sp_draw_now"],
        "hcp_sp_away_now": odds_info["hcp_sp_away_now"],
        "hcp_line_now": odds_info["hcp_line_now"],
        "spf_sp_home_now": odds_info["spf_sp_home_now"],
        "spf_sp_draw_now": odds_info["spf_sp_draw_now"],
        "spf_sp_away_now": odds_info["spf_sp_away_now"],
        # 状态字段
        "match_status": rec.get("matchStatus"),  # Define/Selling
        "sell_status": rec.get("sellStatus"),
        "weekday": rec.get("weekday"),
        "back_color": rec.get("backColor"),
        "bettingSingle": None,  # 兼容原有字段
        "poolStatus": rec.get("poolStatus"),
    }

def fetch_today_unplayed_matches() -> pd.DataFrame:
    """获取当日未开赛比赛（使用新接口）"""
    rows = []
    
    try:
        # 新接口参数
        params = {
            "clientCode": "3001"
        }
        
        # 调用未开赛专用接口
        js = get_json(JC_UNPLAYED_URL, params=params)
        
        # 检查接口返回状态
        if not js.get("success") or js.get("errorCode") != "0":
            print(f"接口调用失败: {js.get('errorMessage')}")
            return pd.DataFrame()
        
        # 解析比赛数据
        value = js.get("value", {})
        match_info_list = value.get("matchInfoList", [])
        
        # 遍历比赛日期分组
        for match_info in match_info_list:
            # 遍历具体比赛列表
            sub_match_list = match_info.get("subMatchList", [])
            for rec in sub_match_list:
                # 过滤未开赛比赛（matchStatus为Define/Selling都是未开赛状态）
                match_status = rec.get("matchStatus")
                if match_status in ["FINISHED", "结束", "已完赛"]:
                    continue
                
                # 提取比赛数据
                rows.append(extract_unplayed_match_row(rec))
        
        # 构建DataFrame
        df = pd.DataFrame(rows)
        
        # 计算概率特征（保持原有逻辑）
        if not df.empty:
            # 计算让球胜平负概率
            if {"hcp_sp_home_now", "hcp_sp_draw_now", "hcp_sp_away_now"}.issubset(df.columns):
                sp = df[["hcp_sp_home_now", "hcp_sp_draw_now", "hcp_sp_away_now"]].astype(float)
                probs = np.stack([probs_from_sp(h, d, a) for h, d, a in sp.itertuples(index=False)])
                df["hcp_p_home_now"], df["hcp_p_draw_now"], df["hcp_p_away_now"] = probs.T
                df["hcp_p_home_minus_away_now"] = df["hcp_p_home_now"] - df["hcp_p_away_now"]
            
            # 计算胜平负概率（新增）
            if {"spf_sp_home_now", "spf_sp_draw_now", "spf_sp_away_now"}.issubset(df.columns):
                sp = df[["spf_sp_home_now", "spf_sp_draw_now", "spf_sp_away_now"]].astype(float)
                probs = np.stack([probs_from_sp(h, d, a) for h, d, a in sp.itertuples(index=False)])
                df["spf_p_home_now"], df["spf_p_draw_now"], df["spf_p_away_now"] = probs.T
                df["spf_p_home_minus_away_now"] = df["spf_p_home_now"] - df["spf_p_away_now"]
        
        return df
    
    except Exception as e:
        print(f"获取未开赛比赛失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# ---------- 4. bonus 数据（未开赛） ----------
def fetch_bonus_single_unplayed(match_id: int) -> Dict[str, Any]:
    """获取未开赛比赛的bonus数据"""
    for _ in range(RETRY):
        try:
            params = {"clientCode": "3001", "matchId": match_id}
            js = get_json(JC_BONUS_URL, params=params)
            val = js.get("value") or {}
            out = {"match_id": match_id, "bonus_isCancel": val.get("isCancel")}
            
            # hadList（未开赛只有当前赔率）
            had = (val.get("oddsHistory") or {}).get("hadList") or []
            if had:
                latest = had[-1]  # 最新赔率
                out.update({
                    "spf_sp_home_open":  safe_float(latest.get("h")),
                    "spf_sp_draw_open":  safe_float(latest.get("d")),
                    "spf_sp_away_open":  safe_float(latest.get("a")),
                    "spf_sp_home_close": safe_float(latest.get("h")),
                    "spf_sp_draw_close": safe_float(latest.get("d")),
                    "spf_sp_away_close": safe_float(latest.get("a")),
                })
                ph, pd, pa = probs_from_sp(out["spf_sp_home_close"], out["spf_sp_draw_close"], out["spf_sp_away_close"])
                out.update({"spf_p_home_close": ph, "spf_p_draw_close": pd, "spf_p_away_close": pa,
                            "spf_p_home_minus_away_close": ph - pa})
            
            # hhadList
            hhad = (val.get("oddsHistory") or {}).get("hhadList") or []
            if hhad:
                latest = hhad[-1]
                out.update({
                    "hcp_sp_home_open":  safe_float(latest.get("h")),
                    "hcp_sp_draw_open":  safe_float(latest.get("d")),
                    "hcp_sp_away_open":  safe_float(latest.get("h")),
                    "hcp_line_open":     safe_float(latest.get("goalLine")),
                    "hcp_sp_home_close": safe_float(latest.get("h")),
                    "hcp_sp_draw_close": safe_float(latest.get("d")),
                    "hcp_sp_away_close": safe_float(latest.get("a")),
                    "hcp_line_close":    safe_float(latest.get("goalLine")),
                })
                ph, pd, pa = probs_from_sp(out["hcp_sp_home_close"], out["hcp_sp_draw_close"], out["hcp_sp_away_close"])
                out.update({"hcp_p_home_close": ph, "hcp_p_draw_close": pd, "hcp_p_away_close": pa,
                            "hcp_p_home_minus_away_close": ph - pa})
            
            out["bonus_score"] = ""  # 未开赛无比分
            return out
        except Exception as e:
            time.sleep(0.5)
    return {"match_id": match_id, "bonus_error": "max_retry"}

def fetch_bonus_concurrent_unplayed(match_ids: List[int]) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=BONUS_WORKERS) as exe:
        fut_map = {exe.submit(fetch_bonus_single_unplayed, mid): mid for mid in match_ids}
        for fut in as_completed(fut_map):
            rows.append(fut.result())
            time.sleep(0.05)
    return pd.DataFrame(rows)

# ---------- 5. 支持率（未开赛） ----------
def parse_support_item(d: Dict[str, Any]) -> Dict[str, float]:
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

def fetch_support_single_unplayed(match_id: int) -> Dict[str, Any]:
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

def fetch_support_concurrent_unplayed(match_ids: List[int]) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=SUPPORT_WORKERS) as exe:
        fut_map = {exe.submit(fetch_support_single_unplayed, mid): mid for mid in match_ids}
        for fut in as_completed(fut_map):
            rows.append(fut.result())
            time.sleep(0.02)
    return pd.DataFrame(rows)

# ---------- 6. 比赛特征（未开赛） ----------
def fetch_feature_single_unplayed(match_id: int) -> Dict[str, Any]:
    for _ in range(RETRY):
        try:
            params = {
                "termLimits": 10,
                "sportteryMatchId": match_id
            }
            js = get_json(JC_FEATURE_URL, params=params)
            
            if not js.get("success") or js.get("errorCode") != "0":
                raise ValueError(f"接口返回错误: {js.get('errorMessage')}")
            
            val = js.get("value") or {}
            out = {"match_id": match_id}
            
            # 基础信息
            out["home_team_short_name"] = val.get("homeTeamShortName", "")
            out["away_team_short_name"] = val.get("awayTeamShortName", "")
            
            # 进球平均值
            goal_avg = val.get("goalAvg", {})
            out.update({
                "home_goal_avg_cnt": safe_float_parse(goal_avg.get("homeGoalAvgCnt")),
                "away_goal_avg_cnt": safe_float_parse(goal_avg.get("awayGoalAvgCnt")),
                "home_goal_avg_ratio": safe_float_parse(goal_avg.get("homeGoalAvgCntRatio")),
                "away_goal_avg_ratio": safe_float_parse(goal_avg.get("awayGoalAvgCntRatio")),
            })
            
            # 失球平均值
            loss_goal_avg = val.get("lossGoalAvg", {})
            out.update({
                "home_loss_goal_avg_cnt": safe_float_parse(loss_goal_avg.get("homeLossGoalAvgCnt")),
                "away_loss_goal_avg_cnt": safe_float_parse(loss_goal_avg.get("awayLossGoalAvgCnt")),
                "home_loss_goal_avg_ratio": safe_float_parse(loss_goal_avg.get("homeLossGoalAvgCntRatio")),
                "away_loss_goal_avg_ratio": safe_float_parse(loss_goal_avg.get("awayLossGoalAvgCntRatio")),
            })
            
            # 主客场胜负平
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
            
            # 相同主客场统计
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
            
            # 衍生特征
            home_total = out["home_win_cnt"] + out["home_draw_cnt"] + out["home_loss_cnt"]
            out["home_win_rate"] = out["home_win_cnt"] / home_total if home_total > 0 else 0.0
            
            away_total = out["away_win_cnt"] + out["away_draw_cnt"] + out["away_loss_cnt"]
            out["away_win_rate"] = out["away_win_cnt"] / away_total if away_total > 0 else 0.0
            
            out["home_away_goal_diff"] = out["home_goal_avg_cnt"] - out["away_goal_avg_cnt"]
            out["home_away_loss_goal_diff"] = out["home_loss_goal_avg_cnt"] - out["away_loss_goal_avg_cnt"]
            
            return out
        
        except Exception as e:
            print(f"获取特征失败 match_id={match_id}: {str(e)}")
            time.sleep(0.5)
    
    # 默认返回值
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

def fetch_feature_concurrent_unplayed(match_ids: List[int]) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=FEATURE_WORKERS) as exe:
        fut_map = {exe.submit(fetch_feature_single_unplayed, mid): mid for mid in match_ids}
        for fut in as_completed(fut_map):
            rows.append(fut.result())
            time.sleep(0.05)
    return pd.DataFrame(rows)

# ---------- 7. fbref 特征整合 ----------
def load_team_map(path: pathlib.Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    tm = pd.read_csv(path, encoding="gbk")
    tm.columns = [c.strip().replace("\\", "") for c in tm.columns]
    return dict(zip(tm["team_cn_sporttery"], tm["team_en_fbref"]))

def load_fbref_features() -> pd.DataFrame:
    """加载fbref特征（简化版）"""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
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
        
        all_dfs = load_fbref_csvs(FBREF_BASE_DIR, FBREF_SEASON_DIR)
        std = all_dfs.get("standard")
        if std is None or "Squad" not in std.columns:
            return pd.DataFrame()
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
    except Exception as e:
        print(f"加载fbref特征失败: {e}")
        return pd.DataFrame()

def merge_fbref_features(df: pd.DataFrame) -> pd.DataFrame:
    """合并fbref特征"""
    if df.empty:
        return df
    
    # 加载队名映射和fbref特征
    try:
        mp = load_team_map(TEAM_MAP_PATH)
        fb = load_fbref_features()
        
        if fb.empty:
            return df
        
        # 队名映射（优先使用全称，兼容简称）
        df["home_team_fbref"] = df["home_team_cn"].map(mp)
        df["away_team_fbref"] = df["away_team_cn"].map(mp)
        
        # 补充：如果全称匹配不到，尝试用简称匹配
        if df["home_team_fbref"].isna().any():
            print("部分主队全称匹配失败，尝试简称匹配")
            # 这里需要根据你的team_map调整，假设包含简称映射
            home_na_mask = df["home_team_fbref"].isna()
            df.loc[home_na_mask, "home_team_fbref"] = df.loc[home_na_mask, "home_team_abb"].map(mp)
        
        if df["away_team_fbref"].isna().any():
            print("部分客队全称匹配失败，尝试简称匹配")
            away_na_mask = df["away_team_fbref"].isna()
            df.loc[away_na_mask, "away_team_fbref"] = df.loc[away_na_mask, "away_team_abb"].map(mp)
        
        # 合并主客场特征
        fb_home = fb.add_prefix("home_").rename(columns={"home_Squad": "home_team_fbref"})
        fb_away = fb.add_prefix("away_").rename(columns={"away_Squad": "away_team_fbref"})
        
        df = df.merge(fb_home, on="home_team_fbref", how="left")
        df = df.merge(fb_away, on="away_team_fbref", how="left")
        
        # 差值特征
        diff_cols = ["strength_xg_diff", "strength_def_diff", "strength_offbuild_diff", "tempo_diff"]
        for col in diff_cols:
            suffix = col.split("_diff")[0]
            field = suffix.split("_", 1)[-1]
            home_col, away_col = f"home_{field}", f"away_{field}"
            if home_col in df.columns and away_col in df.columns:
                df[col] = df[home_col] - df[away_col]
                
    except Exception as e:
        print(f"合并fbref特征失败: {e}")
    
    return df

# ---------- 8. 主流程 ----------
def build_today_predict_data() -> pd.DataFrame:
    """构建当日预测用数据"""
    print(f"📅 获取 {TODAY_DATE} 未开赛比赛数据...")
    
    # 1. 获取未开赛比赛列表（使用新接口）
    print("【1】拉取未开赛比赛列表...")
    unplayed_df = fetch_today_unplayed_matches()
    if unplayed_df.empty:
        print("⚠️  暂无未开赛比赛")
        return unplayed_df
    print(f"  共 {len(unplayed_df)} 场未开赛比赛")
    
    # 2. 获取bonus数据
    print("【2】拉取赔率数据...")
    mids = unplayed_df["match_id"].dropna().astype(int).unique().tolist()
    bonus_df = fetch_bonus_concurrent_unplayed(mids)
    unplayed_df = unplayed_df.merge(bonus_df, on="match_id", how="left")
    
    # 3. 获取支持率数据
    print("【3】拉取支持率数据...")
    support_df = fetch_support_concurrent_unplayed(mids)
    unplayed_df = unplayed_df.merge(support_df, on="match_id", how="left")
    
    # 4. 获取比赛特征
    print("【4】拉取比赛特征数据...")
    feature_df = fetch_feature_concurrent_unplayed(mids)
    unplayed_df = unplayed_df.merge(feature_df, on="match_id", how="left")
    
    # 5. 合并fbref特征
    print("【5】合并球队统计特征...")
    unplayed_df = merge_fbref_features(unplayed_df)
    
    return unplayed_df

# ---------- 9. 入口 ----------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 构建预测数据
    predict_df = build_today_predict_data()
    
    if not predict_df.empty:
        # 保存预测数据
        out_file = OUTPUT_DIR / f"jc_today_unplayed_for_predict_{TODAY_DATE}.csv"
        predict_df.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"✅ 预测数据已保存至: {out_file}")
        print(f"📊 共 {len(predict_df)} 场比赛可用于预测")
        
        # 打印前5条数据预览
        print("\n📈 数据预览（前5条）:")
        preview_cols = ["match_num_str", "home_team_cn", "away_team_cn", "match_time", "hcp_sp_home_now", "hcp_sp_away_now"]
        preview_cols = [col for col in preview_cols if col in predict_df.columns]
        print(predict_df[preview_cols].head())
    else:
        print("✅ 处理完成，暂无未开赛比赛数据")