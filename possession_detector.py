"""
Rule-based football possession detector.

Expected input:
  - ball_x, ball_y: normalized pitch coordinates (0..1)
  - player columns: <player_id>_x, <player_id>_y
  - optional <player_id>_s speed columns (not required)
  - optional timestamp and frame_id

Team mapping:
  - Home: player IDs 1..24
  - Away: player IDs 25..46

The model does not train. It calculates spatial/movement evidence and applies
centered temporal smoothing. Centered windows use future frames, so this is
intended for offline match analysis, not live inference.

Install:
  pip install pandas numpy

Run:
  python possession_detector.py "match.csv" --output "predictions.csv"

Optional plot:
  python possession_detector.py "match.csv" --plot "possession_timeline.png"
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Config:
    fps: float = 25.0
    pitch_length_m: float = 105.3
    pitch_width_m: float = 68.0

    home_ids: tuple = tuple(range(1, 25))
    away_ids: tuple = tuple(range(25, 47))

    # Spatial / possession features
    possession_radius_m: float = 1.8
    distance_sigma_m: float = 2.0
    team_support_radius_m: float = 5.0
    stability_window_s: float = 0.60
    velocity_error_scale_mps: float = 4.0

    # Score weights (individual-player score)
    weight_distance: float = 0.45
    weight_stability: float = 0.30
    weight_velocity: float = 0.25
    weight_team_support: float = 0.10

    possession_threshold: float = 0.48
    possession_margin: float = 0.08
    smoothing_window_s: float = 0.60
    switch_confirm_s: float = 0.24
    uncertain_hold_s: float = 1.0

    # Dead-ball detection: only a tracked ball clearly outside the pitch.
    out_of_bounds_margin: float = 0.005  # normalized coordinate units
    deadball_confirm_s: float = 0.08

    # X cannot be reliably inferred from positions alone. Keep disabled by
    # default; use inactive_intervals for known halftime/pre/post-match spans.
    detect_inactive_heuristically: bool = False
    inactive_confirm_s: float = 8.0
    inactive_ball_speed_mps: float = 0.25
    inactive_player_median_speed_mps: float = 0.35

    # Known inactive intervals in seconds relative to the first row.
    # Example: [(0, 15), (2700, 2850), (5400, 5500)]
    inactive_intervals: tuple = ()


def _base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove repeated duplicate column blocks such as period_id.1, ball_x.1, etc.
    Keep the first/original version of each column.
    """
    keep = []
    seen = set()
    for col in df.columns:
        base = re.sub(r"\.\d+$", "", str(col))
        if base not in seen:
            keep.append(col)
            seen.add(base)
    return df.loc[:, keep].copy()


def _get_time_seconds(df: pd.DataFrame, fps: float) -> np.ndarray:
    if "timestamp" in df.columns:
        s = df["timestamp"]
        # Common tracking format: "0 days 00:01:42.520000"
        td = pd.to_timedelta(s, errors="coerce")
        if td.notna().sum() >= max(2, int(0.5 * len(df))):
            seconds = td.dt.total_seconds().to_numpy(dtype=float)
            return seconds - seconds[0]

        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().sum() >= max(2, int(0.5 * len(df))):
            vals = numeric.to_numpy(dtype=float)
            return vals - vals[0]

    if "frame_id" in df.columns:
        frame = pd.to_numeric(df["frame_id"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(frame).sum() >= 2:
            return (frame - frame[0]) / fps

    return np.arange(len(df), dtype=float) / fps


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=max(1, window), center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def _run_length_mask(mask: np.ndarray, minimum_frames: int) -> np.ndarray:
    """Return True for every frame belonging to a True run of sufficient length."""
    out = np.zeros(len(mask), dtype=bool)
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(mask) - 1):
            end = i if not value else i + 1
            if end - start >= minimum_frames:
                out[start:end] = True
            start = None
    return out


def _player_velocity(xy: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Finite-difference x/y velocity in metres per second."""
    n = len(time_s)
    vel = np.full((n, 2), np.nan, dtype=float)
    if n < 2:
        return vel

    dt = np.diff(time_s)
    delta = np.diff(xy, axis=0)
    valid = (
        np.isfinite(delta).all(axis=1)
        & np.isfinite(dt)
        & (dt > 0)
    )
    step_vel = np.full((n - 1, 2), np.nan, dtype=float)
    step_vel[valid] = delta[valid] / dt[valid, None]

    # Use the adjacent finite difference; average previous/next when available.
    vel[0] = step_vel[0]
    vel[-1] = step_vel[-1]
    if n > 2:
        for i in range(1, n - 1):
            pair = np.vstack([step_vel[i - 1], step_vel[i]])
            if np.isfinite(pair).all(axis=1).any():
                vel[i] = np.nanmean(pair, axis=0)
    return vel


def _make_segments(df: pd.DataFrame):
    """Process each period separately to avoid differencing across halftime."""
    if "period_id" not in df.columns:
        return [np.arange(len(df))]
    period = df["period_id"].astype("string").fillna("__missing__")
    return [group.index.to_numpy() for _, group in df.groupby(period, sort=False)]


def predict_possession(
    input_csv: str | Path,
    output_csv: str | Path = "predictions.csv",
    config: Optional[Config] = None,
) -> pd.DataFrame:
    cfg = config or Config()
    df = pd.read_csv(input_csv)
    df = _base_columns(df)

    required = {"ball_x", "ball_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Numeric conversion and chronological ordering.
    for col in df.columns:
        if col.endswith(("_x", "_y", "_s")) or col in {"frame_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    time_s = _get_time_seconds(df, cfg.fps)
    df["_time_s"] = time_s
    if "frame_id" in df.columns:
        df = df.sort_values("frame_id", kind="stable").reset_index(drop=True)
        time_s = _get_time_seconds(df, cfg.fps)
        df["_time_s"] = time_s
    else:
        df = df.sort_values("_time_s", kind="stable").reset_index(drop=True)
        time_s = df["_time_s"].to_numpy()

    n = len(df)
    stability_frames = max(1, round(cfg.stability_window_s * cfg.fps))
    smooth_frames = max(1, round(cfg.smoothing_window_s * cfg.fps))
    switch_frames = max(1, round(cfg.switch_confirm_s * cfg.fps))
    hold_frames = max(1, round(cfg.uncertain_hold_s * cfg.fps))
    deadball_frames = max(1, round(cfg.deadball_confirm_s * cfg.fps))

    ball_xy = np.column_stack([
        df["ball_x"].to_numpy(dtype=float) * cfg.pitch_length_m,
        df["ball_y"].to_numpy(dtype=float) * cfg.pitch_width_m,
    ])
    ball_valid = np.isfinite(ball_xy).all(axis=1)
    ball_vel = _player_velocity(ball_xy, time_s)
    ball_speed = np.linalg.norm(ball_vel, axis=1)
    ball_speed[~np.isfinite(ball_speed)] = np.nan

    # Gather player positions in metres. A missing player ID is allowed so the
    # same script can run on the 20-player sample as well as the 46-player file.
    player_ids = []
    player_xy = {}
    for pid in list(cfg.home_ids) + list(cfg.away_ids):
        xcol, ycol = f"{pid}_x", f"{pid}_y"
        if xcol in df.columns and ycol in df.columns:
            xy = np.column_stack([
                df[xcol].to_numpy(dtype=float) * cfg.pitch_length_m,
                df[ycol].to_numpy(dtype=float) * cfg.pitch_width_m,
            ])
            player_ids.append(pid)
            player_xy[pid] = xy

    home_ids = [p for p in cfg.home_ids if p in player_xy]
    away_ids = [p for p in cfg.away_ids if p in player_xy]
    if not home_ids or not away_ids:
        warnings.warn(
            "This file does not contain players from both teams. "
            "The sample file may contain only IDs 1-20; meaningful H/A "
            "predictions require IDs from both 1-24 and 25-46."
        )

    # Frame-level player distances and velocity matching.
    dist_by_player = {}
    velocity_match_by_player = {}
    for pid in player_ids:
        xy = player_xy[pid]
        dist = np.linalg.norm(xy - ball_xy, axis=1)
        dist[~(np.isfinite(xy).all(axis=1) & ball_valid)] = np.nan
        dist_by_player[pid] = dist

        pvel = _player_velocity(xy, time_s)
        velocity_error = np.linalg.norm(pvel - ball_vel, axis=1)
        match = np.exp(-velocity_error / max(cfg.velocity_error_scale_mps, 1e-6))
        match[~np.isfinite(velocity_error)] = 0.0
        velocity_match_by_player[pid] = match

    # Team scores, built independently for each period.
    home_score = np.zeros(n, dtype=float)
    away_score = np.zeros(n, dtype=float)
    nearest_home = np.full(n, np.nan)
    nearest_away = np.full(n, np.nan)
    home_support = np.zeros(n, dtype=float)
    away_support = np.zeros(n, dtype=float)
    home_control_duration = np.zeros(n, dtype=float)
    away_control_duration = np.zeros(n, dtype=float)

    segments = _make_segments(df)
    for idx in segments:
        for team_ids, score_out, nearest_out, support_out, duration_out in [
            (home_ids, home_score, nearest_home, home_support, home_control_duration),
            (away_ids, away_score, nearest_away, away_support, away_control_duration),
        ]:
            if not team_ids:
                continue

            team_dist = np.column_stack([dist_by_player[p][idx] for p in team_ids])
            team_velmatch = np.column_stack([velocity_match_by_player[p][idx] for p in team_ids])

            # Nearest distance and support count.
            nearest = np.nanmin(
                np.where(np.isfinite(team_dist), team_dist, np.inf), axis=1
            )
            nearest[~np.isfinite(nearest) | (nearest == np.inf)] = np.nan
            support = np.sum(team_dist <= cfg.team_support_radius_m, axis=1)
            support_score = np.clip(support / 3.0, 0.0, 1.0)

            # Score each player using proximity, sustained proximity, and
            # similarity between player and ball velocity vectors.
            per_player_scores = []
            per_player_durations = []
            for j, pid in enumerate(team_ids):
                d = dist_by_player[pid][idx]
                d_score = np.exp(-d / max(cfg.distance_sigma_m, 1e-6))
                d_score[~np.isfinite(d)] = 0.0

                stable_d = _rolling_mean(d, stability_frames)
                stable_score = np.exp(-stable_d / max(cfg.possession_radius_m, 1e-6))
                stable_score[~np.isfinite(stable_d)] = 0.0

                v_score = velocity_match_by_player[pid][idx]
                individual = (
                    cfg.weight_distance * d_score
                    + cfg.weight_stability * stable_score
                    + cfg.weight_velocity * v_score
                )
                per_player_scores.append(individual)

                within = np.isfinite(d) & (d <= cfg.possession_radius_m)
                duration = np.zeros(len(idx), dtype=float)
                count = 0
                for k, yes in enumerate(within):
                    count = count + 1 if yes else 0
                    duration[k] = count / cfg.fps
                per_player_durations.append(duration)

            best_player_score = np.max(np.column_stack(per_player_scores), axis=1)
            team_score = (
                (1.0 - cfg.weight_team_support) * best_player_score
                + cfg.weight_team_support * support_score
            )
            score_out[idx] = team_score
            nearest_out[idx] = nearest
            support_out[idx] = support_score
            duration_out[idx] = np.max(np.column_stack(per_player_durations), axis=1)

    # Centered temporal smoothing of team evidence, separately per period.
    home_smooth = np.zeros(n)
    away_smooth = np.zeros(n)
    for idx in segments:
        home_smooth[idx] = _rolling_mean(home_score[idx], smooth_frames)
        away_smooth[idx] = _rolling_mean(away_score[idx], smooth_frames)

    # Initial possession candidate. U is internal uncertainty, not an output label.
    raw = np.full(n, "U", dtype=object)
    home_wins = (
        (home_smooth >= cfg.possession_threshold)
        & ((home_smooth - away_smooth) >= cfg.possession_margin)
    )
    away_wins = (
        (away_smooth >= cfg.possession_threshold)
        & ((away_smooth - home_smooth) >= cfg.possession_margin)
    )
    raw[home_wins] = "H"
    raw[away_wins] = "A"

    # Dead ball: require a tracked ball outside the normalized pitch boundary.
    bx = df["ball_x"].to_numpy(dtype=float)
    by = df["ball_y"].to_numpy(dtype=float)
    m = cfg.out_of_bounds_margin
    outside = ball_valid & (
        (bx < -m) | (bx > 1.0 + m) | (by < -m) | (by > 1.0 + m)
    )
    deadball = np.zeros(n, dtype=bool)
    for idx in segments:
        deadball[idx] = _run_length_mask(outside[idx], deadball_frames)

    # Explicit inactive intervals take priority. Intervals are seconds relative
    # to the first timestamp in the file.
    inactive = np.zeros(n, dtype=bool)
    relative_time = time_s - np.nanmin(time_s)
    for start_s, end_s in cfg.inactive_intervals:
        inactive |= (relative_time >= start_s) & (relative_time <= end_s)

    # Optional conservative inactivity heuristic. This is disabled by default
    # because positions alone cannot identify halftime/injury/penalty reliably.
    if cfg.detect_inactive_heuristically:
        all_speeds = []
        for pid in player_ids:
            pvel = _player_velocity(player_xy[pid], time_s)
            all_speeds.append(np.linalg.norm(pvel, axis=1))
        if all_speeds:
            median_player_speed = np.nanmedian(np.column_stack(all_speeds), axis=1)
        else:
            median_player_speed = np.full(n, np.nan)
        inactive_candidate = (
            (ball_speed <= cfg.inactive_ball_speed_mps)
            & (median_player_speed <= cfg.inactive_player_median_speed_mps)
        )
        for idx in segments:
            inactive[idx] |= _run_length_mask(
                inactive_candidate[idx],
                max(1, round(cfg.inactive_confirm_s * cfg.fps)),
            )

    # Temporal hysteresis: require a new H/A candidate to persist before
    # switching. U frames retain the last possession briefly.
    final_possession = np.full(n, "U", dtype=object)
    for idx in segments:
        current = "U"
        pending = "U"
        pending_count = 0
        uncertain_count = 0
        for local_i, global_i in enumerate(idx):
            candidate = raw[global_i]
            if candidate in ("H", "A"):
                uncertain_count = 0
                if candidate == current:
                    pending, pending_count = "U", 0
                else:
                    if candidate == pending:
                        pending_count += 1
                    else:
                        pending, pending_count = candidate, 1
                    if pending_count >= switch_frames:
                        current = candidate
                        pending, pending_count = "U", 0
            else:
                pending, pending_count = "U", 0
                uncertain_count += 1
                if uncertain_count > hold_frames:
                    current = "U"
            final_possession[global_i] = current

        # Fill remaining unresolved frames from nearest known state, without
        # crossing period boundaries. This only resolves short edge/uncertain
        # regions; long unresolved spans remain U for the fallback below.
        ser = pd.Series(final_possession[idx]).replace("U", np.nan)
        filled = ser.ffill().bfill()
        final_possession[idx] = filled.fillna("U").to_numpy()

    # Output must be H/A/D/X. For any remaining ambiguous U, choose the stronger
    # team score but preserve a low confidence score so uncertainty is visible.
    final = final_possession.copy()
    unresolved = final == "U"
    final[unresolved] = np.where(
        home_smooth[unresolved] >= away_smooth[unresolved], "H", "A"
    )
    final[deadball] = "D"
    final[inactive] = "X"

    # Confidence is a transparent score-strength indicator, not a calibrated
    # probability. Ambiguous score margins lower it.
    confidence = np.maximum(home_smooth, away_smooth) * (
        np.abs(home_smooth - away_smooth) / (np.maximum(home_smooth, away_smooth) + 1e-9)
    )
    confidence = np.clip(confidence, 0.0, 1.0)
    confidence[deadball | inactive] = 1.0

    result = df.copy()
    result["nearest_home_distance_m"] = nearest_home
    result["nearest_away_distance_m"] = nearest_away
    result["home_team_support_score"] = home_support
    result["away_team_support_score"] = away_support
    result["home_control_duration_s"] = home_control_duration
    result["away_control_duration_s"] = away_control_duration
    result["ball_speed_calculated_mps"] = ball_speed
    result["home_possession_score"] = home_score
    result["away_possession_score"] = away_score
    result["home_smoothed_score"] = home_smooth
    result["away_smoothed_score"] = away_smooth
    result["raw_prediction"] = raw
    result["prediction"] = final
    result["confidence_indicator"] = confidence

    result.to_csv(output_csv, index=False)
    return result


def plot_predictions(result: pd.DataFrame, output_path: str | Path):
    import matplotlib.pyplot as plt

    x = result["_time_s"].to_numpy()
    labels = result["prediction"].map({"H": 0, "A": 1, "D": 2, "X": 3}).to_numpy()
    fig, ax = plt.subplots(figsize=(15, 3.5))
    ax.step(x, labels, where="post")
    ax.set_yticks([0, 1, 2, 3], ["H (Home)", "A (Away)", "D (Dead ball)", "X (Inactive)"])
    ax.set_xlabel("Time from start (seconds)")
    ax.set_title("Rule-based ball possession prediction")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Rule-based football possession detector")
    parser.add_argument("input_csv", help="Tracking data CSV")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--plot", default=None, help="Optional possession timeline PNG")
    parser.add_argument("--detect-inactive", action="store_true",
                        help="Enable conservative low-motion X heuristic")
    args = parser.parse_args()

    cfg = Config(detect_inactive_heuristically=args.detect_inactive)
    result = predict_possession(args.input_csv, args.output, cfg)
    print(f"Saved predictions to: {args.output}")
    print("\nPrediction counts:")
    print(result["prediction"].value_counts(dropna=False).to_string())
    if args.plot:
        plot_predictions(result, args.plot)
        print(f"Saved plot to: {args.plot}")


if __name__ == "__main__":
    main()
