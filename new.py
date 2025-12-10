"""
LightGBM pipeline for horse-race rank1 prediction (IsWinner).
Features engineered per EDA conclusions + contextual race factors:
- Impute HPPNC by median (3.66% missing, skewed)
- Remove/soft-delete weak features (Brace2race; keep distance/race-condition signals)
- Collapse multicollinearity via composite scores (JockeyScore, HorseFormScore, MarketForm)
- Relative in-race features (group standardization/rank) by Race
- Target-encoded sparse Trainer with smoothing; win-rate signals for Trainer/Runners
- Contextual race features retained and encoded: distance, going, race class, course/venue bias

Target: per-race ranking to pick the champion (Rank==1). Objective: LambdaRank.
Primary metric: per-race Top-1 Accuracy; secondary: winner-in-Top3 recall.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")


SEED = 42
DATA_PATH = "Horse-Data_student.xlsx"
TRAIN_SHEET = "Training"
TEST_SHEET = "Test"
EXAM_SHEET = "Exam"
USE_GPU = True  # set False to force CPU
GPU_PARAMS = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0} if USE_GPU else {}

# Base variables from the EDA notebook
WANTED_VARIABLES = [
    "LastLogOdds",
    "HWinPer",
    "HPPNC",
    "NHPP",
    "JWinPer",
    "JPP",
    "TWinPer",
    "TPP",
    "HAge",
    "LifeHNoRace20",
    "Brace2race",
    "AveStdRank60",
    "JAveStdRank60",
    "TAveStdRank30",
    "AveSpeedRating",
    "LastSpeedRating",
    "WtCarriedChg",
    "DistTChg",
    "LWP60",
    "Runners",
    "Trainer",
    "Rank",
]


@dataclass
class FeatureMappings:
    hppnc_median: float
    overall_win_rate: float
    trainer_stats: pd.DataFrame
    runner_stats: pd.DataFrame
    trainer_te_map: Dict[str, float]
    going_te_map: Dict[str, float]
    raceclass_te_map: Dict[str, float]
    course_te_map: Dict[str, float]
    dist_te_map: Dict[str, float]


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/test/exam sheets."""
    train_df = pd.read_excel(DATA_PATH, sheet_name=TRAIN_SHEET)
    test_df = pd.read_excel(DATA_PATH, sheet_name=TEST_SHEET)
    exam_df = pd.read_excel(DATA_PATH, sheet_name=EXAM_SHEET)
    return train_df, test_df, exam_df


def compute_mappings(train_df: pd.DataFrame) -> FeatureMappings:
    """Compute global statistics used for feature engineering."""
    hppnc_median = float(train_df["HPPNC"].median())

    # Target
    is_winner = (train_df["Rank"] == 1).astype(int)
    overall_win_rate = is_winner.mean()

    def smooth_te(df: pd.DataFrame, col: str, prior: float = 20.0) -> Dict[str, float]:
        if col not in df.columns:
            return {}
        stats = df.groupby(col)["IsWinner"].agg(["sum", "count"]).reset_index()
        stats["te"] = (stats["sum"] + prior * overall_win_rate) / (stats["count"] + prior)
        return dict(zip(stats[col], stats["te"]))

    # Trainer statistics with Laplace smoothing to reduce noise for sparse trainers
    trainer_stats = (
        train_df.assign(IsWinner=is_winner)
        .groupby("Trainer")
        .agg(total_races=("IsWinner", "size"), wins=("IsWinner", "sum"))
    )
    prior = 20  # smoothing strength (Laplace)
    trainer_stats["win_rate"] = (
        trainer_stats["wins"] + prior * overall_win_rate
    ) / (trainer_stats["total_races"] + prior)
    trainer_te_map = trainer_stats["win_rate"].to_dict()

    # Contextual target encodings
    enriched = train_df.assign(IsWinner=is_winner)
    going_te_map = smooth_te(enriched, "Going")
    raceclass_te_map = smooth_te(enriched, "RaceClass")
    course_te_map = smooth_te(enriched, "Course")

    # Distance bucket TE
    if "Dist" in enriched.columns:
        dist_bins = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 9999]
        labels = ["<=1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000-2400", "2400+"]
        enriched["DistBucket"] = pd.cut(enriched["Dist"], bins=dist_bins, labels=labels, include_lowest=True)
        dist_te_map = smooth_te(enriched, "DistBucket")
    else:
        dist_te_map = {}

    # Runner-count win rates
    runner_stats = (
        train_df.assign(IsWinner=is_winner)
        .groupby("Runners")
        .agg(total_races=("IsWinner", "size"), wins=("IsWinner", "sum"))
    )
    runner_stats["win_rate"] = (
        runner_stats["wins"] + prior * overall_win_rate
    ) / (runner_stats["total_races"] + prior)

    # Label encoding for Trainer
    return FeatureMappings(
        hppnc_median=hppnc_median,
        overall_win_rate=overall_win_rate,
        trainer_stats=trainer_stats,
        runner_stats=runner_stats,
        trainer_te_map=trainer_te_map,
        going_te_map=going_te_map,
        raceclass_te_map=raceclass_te_map,
        course_te_map=course_te_map,
        dist_te_map=dist_te_map,
    )


def build_features(df: pd.DataFrame, mappings: FeatureMappings) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Create model features per FE plan, with defensive checks to avoid leakage.
    """
    data = df.copy()
    orig_idx = data["__orig_idx"] if "__orig_idx" in data.columns else pd.Series(np.arange(len(data)))

    # Drop target columns early to avoid leakage
    for tgt in ["Rank", "IsWinner"]:
        if tgt in data.columns:
            data = data.drop(columns=[tgt])

    # Convert categorical columns to string to avoid category reductions later
    for col in data.columns:
        if str(data[col].dtype) == "category":
            data[col] = data[col].astype(str)

    # --- Group identifier ---
    if "RaceID" not in data.columns:
        if all(c in data.columns for c in ["Date", "Raceno"]):
            data["RaceID"] = data["Date"].astype(str) + "_" + data["Raceno"].astype(str)
        else:
            raise ValueError("Race-level eval requires RaceID or Date+Raceno columns.")

    # --- 1) Missing value handling ---
    data["HPPNC_filled"] = data["HPPNC"].fillna(mappings.hppnc_median)

    # --- 2) Weak feature removal ---
    drop_cols = [c for c in ["Brace2race"] if c in data.columns]  # keep distance/going/raceclass
    if drop_cols:
        data = data.drop(columns=drop_cols)

    # --- 3) Composite scores to handle multicollinearity ---
    if all(c in data.columns for c in ["JWinPer", "JAveStdRank60", "JPP"]):
        data["JockeyScore"] = (
            0.4 * data["JWinPer"] - 0.3 * data["JAveStdRank60"] + 0.3 * data["JPP"]
        )
        data = data.drop(columns=["JWinPer", "JAveStdRank60", "JPP"])
    if all(c in data.columns for c in ["HPPNC_filled", "NHPP"]):
        data["HorseFormScore"] = 0.5 * data["HPPNC_filled"] + 0.5 * data["NHPP"]
        data = data.drop(columns=["HPPNC", "NHPP"], errors="ignore")
    if all(c in data.columns for c in ["LastLogOdds", "LWP60"]):
        data["MarketForm"] = -data["LastLogOdds"] + 0.5 * data["LWP60"]

    # --- 4) Trainer encodings (sparse category) ---
    trainer_map = mappings.trainer_te_map
    trainer_races_map = mappings.trainer_stats["total_races"].to_dict()
    data["Trainer_TE"] = data["Trainer"].map(trainer_map).fillna(
        mappings.overall_win_rate
    )
    data["Trainer_total_races"] = data["Trainer"].map(trainer_races_map).fillna(0)

    # --- 5) Runner-count win rate (quasi-categorical) ---
    runners_map = mappings.runner_stats["win_rate"].to_dict()
    data["Runners_win_rate"] = data["Runners"].map(runners_map).fillna(
        mappings.overall_win_rate
    )

    # --- 5b) Contextual encodings: Going, RaceClass, Course, DistBucket ---
    if mappings.going_te_map and "Going" in data.columns:
        data["GoingScore"] = data["Going"].map(mappings.going_te_map).fillna(
            mappings.overall_win_rate
        )
    if mappings.raceclass_te_map and "RaceClass" in data.columns:
        data["RaceClassScore"] = data["RaceClass"].map(mappings.raceclass_te_map).fillna(
            mappings.overall_win_rate
        )
    if mappings.course_te_map and "Course" in data.columns:
        data["CourseBias"] = data["Course"].map(mappings.course_te_map).fillna(
            mappings.overall_win_rate
        )
    if mappings.dist_te_map:
        if "Dist" in data.columns:
            dist_bins = [0, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 9999]
            labels = ["<=1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000-2400", "2400+"]
            data["DistBucket"] = pd.cut(
                data["Dist"], bins=dist_bins, labels=labels, include_lowest=True
            )
            data["DistanceProfile"] = data["DistBucket"].map(mappings.dist_te_map).fillna(
                mappings.overall_win_rate
            )
            data["DistanceProfile"] = pd.to_numeric(data["DistanceProfile"], errors="coerce")
        else:
            data["DistanceProfile"] = mappings.overall_win_rate

    # --- 6) Interaction features (optional, only if columns exist) ---
    if "WtCarried" in data.columns and "AveSpeedRating" in data.columns:
        data["Speed_Weight"] = data["AveSpeedRating"] / (data["WtCarried"] + 1e-3)
    if all(c in data.columns for c in ["Rating", "Drawing"]):
        data["Rating_Draw"] = data["Rating"] * data["Drawing"]
    if all(c in data.columns for c in ["JockeyScore", "HorseFormScore"]):
        data["JockeyHorse"] = data["JockeyScore"] * data["HorseFormScore"]
    if all(c in data.columns for c in ["MarketForm", "Rating"]):
        data["Market_Rating"] = data["MarketForm"] * data["Rating"]

    # --- 7) Race-level relative features ---
    group_cols: List[str] = ["RaceID"]
    rel_feats = [
        c
        for c in [
            "Rating",
            "JockeyScore",
            "HorseFormScore",
            "AveSpeedRating",
            "LastSpeedRating",
            "LastLogOdds",
            "Dist",
            "GoingScore",
            "RaceClassScore",
            "CourseBias",
            "DistanceProfile",
        ]
        if c in data.columns
    ]
    for col in rel_feats:
        # Ensure numeric dtype for aggregation
        data[col] = pd.to_numeric(data[col], errors="coerce")
        grp = data.groupby(group_cols)[col]
        data[col + "_rel"] = data[col] - grp.transform("mean")
        # For odds, lower is better -> ascending True; for others higher is better -> ascending False
        ascending_flag = True if col == "LastLogOdds" else False
        data[col + "_rank"] = grp.rank(ascending=ascending_flag, method="first")

    # Final feature selection: keep engineered plus core numerics that remain
    candidate_cols = [c for c in data.columns if c not in ["Rank", "IsWinner", "RaceID", "__orig_idx"]]
    # Ensure we only keep numeric columns (LightGBM can handle int/float)
    feature_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(data[c])]
    features = data[feature_cols].reset_index(drop=True)
    race_ids = data["RaceID"].reset_index(drop=True)
    return features, race_ids, orig_idx.reset_index(drop=True)


def make_group_list(race_ids: Iterable[str]) -> List[int]:
    """LightGBM Ranker requires group sizes in the order of the data."""
    sizes: List[int] = []
    last = None
    count = 0
    for rid in race_ids:
        if last is None:
            last = rid
            count = 1
        elif rid == last:
            count += 1
        else:
            sizes.append(count)
            last = rid
            count = 1
    if count:
        sizes.append(count)
    return sizes


def per_race_metrics(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    """Compute per-race Top1 accuracy and winner-in-Top3 recall."""
    df = df.copy()
    df["score"] = scores
    if "RaceID" not in df.columns:
        raise ValueError("RaceID is required for per-race metrics.")

    race_groups = df.groupby("RaceID")
    total_races = len(race_groups)
    top1_hits = 0
    top3_hits = 0
    for _, g in race_groups:
        g_sorted = g.sort_values("score", ascending=False)
        if g_sorted.iloc[0]["Rank"] == 1:
            top1_hits += 1
        if (g_sorted.head(3)["Rank"] == 1).any():
            top3_hits += 1
    return {
        "races": total_races,
        "top1_hits": top1_hits,
        "top1_accuracy": top1_hits / total_races if total_races else 0.0,
        "top3_hits": top3_hits,
        "top3_recall": top3_hits / total_races if total_races else 0.0,
    }


def train_ranker(
    features: pd.DataFrame,
    y: np.ndarray,
    race_ids: Iterable[str],
) -> LGBMRanker:
    group = make_group_list(race_ids)
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=SEED,
        n_jobs=-1,
        **GPU_PARAMS,
    )
    model.fit(
        features,
        y,
        group=group,
    )
    return model


def cross_validate_ranker(
    features: pd.DataFrame,
    y: np.ndarray,
    race_ids: pd.Series,
    raw_df: pd.DataFrame,
    folds: int = 5,
) -> List[Dict[str, float]]:
    """GroupKFold on RaceID to validate per-race metrics."""
    gkf = GroupKFold(n_splits=folds)
    metrics_list: List[Dict[str, float]] = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(features, y, groups=race_ids)):
        X_tr, X_va = features.iloc[tr_idx], features.iloc[va_idx]
        y_tr = y[tr_idx]
        rid_tr, rid_va = race_ids.iloc[tr_idx], race_ids.iloc[va_idx]

        model = train_ranker(X_tr, y_tr, rid_tr)
        va_scores = model.predict(X_va)

        va_df = raw_df.iloc[va_idx].copy()
        va_df["RaceID"] = rid_va.values
        va_df["score"] = va_scores
        m = per_race_metrics(va_df, va_scores)
        metrics_list.append(m)

        print(f"\nFold {fold+1}/{folds} - races: {m['races']}, top1_acc: {m['top1_accuracy']:.4f}, top3_recall: {m['top3_recall']:.4f}")

    # summary
    if metrics_list:
        mean_top1 = np.mean([m["top1_accuracy"] for m in metrics_list])
        mean_top3 = np.mean([m["top3_recall"] for m in metrics_list])
        print(f"\nCV summary ({folds}-fold GroupKFold by RaceID):")
        print(f"Avg Top1 accuracy: {mean_top1:.4f}")
        print(f"Avg Top3 recall:   {mean_top3:.4f}")
    return metrics_list


def main():
    # Load data
    train_df, test_df, exam_df = load_data()

    def ensure_race_and_sort(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "__orig_idx" not in df.columns:
            df["__orig_idx"] = np.arange(len(df))
        if "RaceID" not in df.columns and all(c in df.columns for c in ["Date", "Raceno"]):
            df["RaceID"] = df["Date"].astype(str) + "_" + df["Raceno"].astype(str)
        if "RaceID" in df.columns:
            df = df.sort_values("RaceID").reset_index(drop=True)
        return df

    train_df = ensure_race_and_sort(train_df)
    test_df = ensure_race_and_sort(test_df)
    exam_df = ensure_race_and_sort(exam_df)

    # Ensure required columns
    missing_cols = [c for c in WANTED_VARIABLES if c not in train_df.columns]
    if missing_cols:
        raise ValueError(f"Columns missing in training data: {missing_cols}")

    has_rank_test = "Rank" in test_df.columns

    # Build training pool (train + test if labels present)
    if has_rank_test:
        train_full_df = pd.concat([train_df, test_df], ignore_index=True)
        print(f"Using train+test for training: {len(train_df)} + {len(test_df)} rows")
    else:
        train_full_df = train_df.copy()
        print(f"Using train only for training: {len(train_df)} rows")

    # Target for ranking: relevance = 1 if Rank==1 else 0
    y = (train_full_df["Rank"] == 1).astype(int).values

    # Mappings & feature assembly (fit mappings on training pool only)
    mappings = compute_mappings(train_full_df)
    train_features, train_race_ids, train_orig_idx = build_features(train_full_df, mappings)
    test_features, test_race_ids, test_orig_idx = build_features(test_df, mappings)
    exam_features, exam_race_ids, exam_orig_idx = build_features(exam_df, mappings)

    # Train ranker on all training data (LambdaRank)
    # Optional: K-fold group CV for validation (only when labels available)
    if has_rank_test:
        cross_validate_ranker(train_features, y, train_race_ids, train_full_df, folds=5)

    ranker = train_ranker(train_features, y, train_race_ids)

    # Predict on test (if labels were present, still output predictions)
    test_scores = ranker.predict(test_features)
    if has_rank_test:
        test_eval_df = test_df.copy()
        test_eval_df["RaceID"] = test_race_ids.values
        test_eval_df["score"] = test_scores
        metrics = per_race_metrics(test_eval_df, test_scores)

        print("\nPer-race evaluation on Test (note: test was used in training pool):")
        print(f"Races: {metrics['races']}")
        print(f"Top1 hits: {metrics['top1_hits']}, Top1 accuracy: {metrics['top1_accuracy']:.4f}")
        print(f"Top3 hits: {metrics['top3_hits']}, Top3 recall: {metrics['top3_recall']:.4f}")
    else:
        print("\nTest set has no Rank; running inference only.")

    # Predict champions and full ranking for Test/Exam
    def predict_champions(df_src: pd.DataFrame, feats: pd.DataFrame, race_ids: pd.Series, orig_idx: pd.Series, label: str, save_path: str | None = None):
        scores = ranker.predict(feats)
        tmp = df_src.copy()
        tmp["RaceID"] = race_ids.values
        tmp["score"] = scores
        # per-race predicted rank (1 = highest score)
        tmp["PredRank"] = tmp.groupby("RaceID")["score"].rank(ascending=False, method="first")
        winners = (
            tmp.sort_values("score", ascending=False)
            .groupby("RaceID")
            .head(1)[["RaceID", "HorseNo", "score"]]
            .rename(columns={"HorseNo": "PredictedWinnerHorseNo"})
            .reset_index(drop=True)
        )
        print(f"\nPredicted champions for {label}: {len(winners)} races")
        print(winners.head(10))
        if save_path:
            tmp.rename(columns={"score": "PredScore"}, inplace=True)
            # restore original order
            tmp["__orig_idx"] = orig_idx.values
            tmp = tmp.sort_values("__orig_idx")
            tmp.to_csv(save_path, index=False)
            print(f"Saved full predictions to {save_path}")
        return winners

    predict_champions(test_df, test_features, test_race_ids, test_orig_idx, "Test", save_path="predictions_test.csv")
    predict_champions(exam_df, exam_features, exam_race_ids, exam_orig_idx, "Exam", save_path="predictions_exam.csv")


if __name__ == "__main__":
    main()

