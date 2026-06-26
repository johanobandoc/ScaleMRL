"""
Download MRQ and NewT (tdmpc2) run data from WandB and plot them together
per task, with one curve per agent (mean ± std across seeds).

Both agents log to the same WandB project (mmbench_{task_set}), so a single
API call per task set fetches all agents at once.

Workflow
--------
Data is downloaded from WandB once and cached as CSVs in plots/data/{agent}/.
On subsequent runs use --no-download to skip the WandB API call and read
the cached CSVs directly — useful for iterating on plot settings quickly.

    # Use an existing env that already has all dependencies (wandb, pandas, matplotlib, numpy).
    # Activate with:
    #   source /cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/etc/profile.d/conda.sh
    #   conda activate fasttd3_hb
    # Alternatives: dynamo-repro, fasttd3_isaaclab
    #
    # Or call Python directly without activating:
    PYTHON=/home/mila/j/johan.ceron/.conda/envs/fasttd3_hb/bin/python

    # First run: download from WandB (slow), save CSVs, and plot
    $PYTHON download_and_plot_mrq.py

    # Re-plot from cached CSVs without hitting WandB again (fast)
    $PYTHON download_and_plot_mrq.py --no-download
    $PYTHON download_and_plot_mrq.py --no-download --max-steps 5e6
    $PYTHON download_and_plot_mrq.py --no-download --smooth 10

    # Only specific task sets
    $PYTHON download_and_plot_mrq.py --task-sets dmcontrol metaworld
    $PYTHON download_and_plot_mrq.py --no-download --task-sets atari --max-steps 8e6

    # Only specific agents
    $PYTHON download_and_plot_mrq.py --agents mrq
    $PYTHON download_and_plot_mrq.py --no-download --agents mrq tdmpc2

    # Include crashed/failed runs (excluded by default)
    $PYTHON download_and_plot_mrq.py --include-crashed

Robustness
----------
  - Crashed/failed runs are skipped by default (--include-crashed to keep them)
  - If a seed has multiple runs (restarts), keeps the one with the most steps
  - Incomplete/still-running runs are included and plotted up to their last step
  - Seeds with fewer steps than MIN_STEPS_FRACTION of expected total are flagged
  - Per-task metrics missing for some seeds are plotted with NaN gaps
  - Steps are aligned across seeds via reindexing before aggregating mean/std
"""

import argparse
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# ==============================================================================
# CONFIG
# ==============================================================================

ENTITY = "johan-ceron-obando"

# agent key (matches config.agent in WandB) -> display label, color
AGENTS: dict[str, dict] = {
    "mrq":    {"label": "MR.Q",   "color": "#1f77b4"},  # blue
    "tdmpc2": {"label": "NewT",   "color": "#ff7f0e"},  # orange
}

ALL_TASK_SETS = [
    "dmcontrol", "dmcontrol-ext", "metaworld", "mujoco",
    "box2d", "robodesk", "ogbench", "pygame", "atari", "maniskill",
]
SEEDS = [1, 2, 3, 4, 5]
EXPECTED_STEPS = 11_000_000
MIN_STEPS_FRACTION = 0.5

SCORE_KEY  = "eval/avg_score"
PERTASK_RE = re.compile(r"^eval/episode_score\+(.+)$")

GOOD_STATES = {"finished", "running"}

# These are set at runtime depending on --pixel flag
BASE_DIR = Path(__file__).parent
_OBS_MODE = "state"   # updated to "pixel" by --pixel flag


def _plot_dir() -> Path:
    p = BASE_DIR / f"comparison_{_OBS_MODE}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _data_dir(agent: str) -> Path:
    p = BASE_DIR / f"data_{_OBS_MODE}" / agent
    p.mkdir(parents=True, exist_ok=True)
    return p


def _project_name(task_set: str) -> str:
    if _OBS_MODE == "pixel":
        return f"mmbench_pixel_mmbench_{task_set}"
    return f"mmbench_{task_set}"


# ==============================================================================
# HELPERS
# ==============================================================================

def _is_nearly_all_nan(series: pd.Series, threshold: float = 0.95) -> bool:
    return series.isna().mean() > threshold


def _align_seeds(frames: dict[str, pd.Series]) -> pd.DataFrame:
    """Align seed series to a common step index via nearest-neighbor reindex."""
    if not frames:
        return pd.DataFrame()
    all_steps = sorted(set().union(*[s.index for s in frames.values()]))
    df = pd.DataFrame(index=pd.Index(all_steps, name="step"))
    for col, series in frames.items():
        df[col] = series.reindex(df.index, method="nearest", tolerance=1e9)
    return df


# ==============================================================================
# DOWNLOAD
# ==============================================================================

def _pick_best_run(runs_for_seed: list):
    if len(runs_for_seed) == 1:
        return runs_for_seed[0]
    def _max_step(run):
        try:
            return run.summary.get("_step", 0) or 0
        except Exception:
            return 0
    return max(runs_for_seed, key=_max_step)


def fetch_task_set(
    task_set: str,
    agents: list[str],
    include_crashed: bool = False,
    include_failed: bool = False,
) -> dict[str, tuple[dict[str, pd.DataFrame], dict]]:
    """
    Fetch all agents for a task set in one API call.

    Returns:
        {agent: (data, report)}
        data   : metric_short_name -> DataFrame(index=step, columns=[seed_1, …])
        report : per-seed status info
    """
    project = _project_name(task_set)
    print(f"\n[{project}] Fetching runs for agents: {agents} …")
    api = wandb.Api(timeout=180)

    try:
        runs = api.runs(
            f"{ENTITY}/{project}",
            filters={"config.agent": {"$in": agents}},
        )
    except Exception as e:
        print(f"  WARNING: could not fetch {project}: {e}")
        return {ag: ({}, {"project": project, "seeds": {}}) for ag in agents}

    # Group by agent then seed
    runs_by_agent_seed: dict[str, dict[int, list]] = {ag: defaultdict(list) for ag in agents}
    for run in runs:
        ag   = run.config.get("agent")
        seed = run.config.get("seed")
        if ag in agents and seed in SEEDS:
            runs_by_agent_seed[ag][seed].append(run)

    allowed_states = (GOOD_STATES
                      | ({"crashed"} if include_crashed else set())
                      | ({"failed"}  if include_failed  else set()))
    results = {}

    for ag in agents:
        print(f"\n  --- agent: {AGENTS[ag]['label']} ---")
        report: dict = {"project": project, "seeds": {}}
        seed_series: dict[int, dict[str, pd.Series]] = {}

        for seed in SEEDS:
            candidates = runs_by_agent_seed[ag].get(seed, [])
            if not candidates:
                print(f"  seed={seed}  MISSING")
                report["seeds"][seed] = {"status": "missing"}
                continue

            run = _pick_best_run(candidates)
            state = run.state

            if len(candidates) > 1:
                print(f"  seed={seed}  {len(candidates)} runs → keeping '{run.name}' (most steps)")

            if state not in allowed_states:
                print(f"  seed={seed}  state={state} → SKIPPED (use --include-crashed to keep)")
                report["seeds"][seed] = {"status": state, "skipped": True}
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pertask_keys = [k for k in (run.summary or {}).keys() if PERTASK_RE.match(k)]

            keys_to_fetch = [SCORE_KEY] + pertask_keys

            try:
                history = run.history(keys=keys_to_fetch, x_axis="_step", pandas=True)
            except Exception as e:
                print(f"  seed={seed}  ERROR fetching history: {e}")
                report["seeds"][seed] = {"status": "fetch_error", "error": str(e)}
                continue

            if history.empty:
                print(f"  seed={seed}  state={state}  EMPTY HISTORY → skipped")
                report["seeds"][seed] = {"status": "empty_history"}
                continue

            if "_step" in history.columns:
                history = history.rename(columns={"_step": "step"})

            max_step = history["step"].max() if "step" in history.columns else 0
            pct = 100 * max_step / EXPECTED_STEPS
            status_tag = ""
            if state == "running":
                status_tag = " [still running]"
            elif max_step < MIN_STEPS_FRACTION * EXPECTED_STEPS:
                status_tag = f" [INCOMPLETE: {pct:.0f}%]"

            print(f"  seed={seed}  state={state}  steps={int(max_step):,} ({pct:.0f}%){status_tag}")
            report["seeds"][seed] = {
                "status": state,
                "max_step": int(max_step),
                "pct_complete": round(pct, 1),
                "n_pertask_keys": len(pertask_keys),
            }

            metrics: dict[str, pd.Series] = {}
            for col in history.columns:
                if col == "step":
                    continue
                s = history.set_index("step")[col].dropna()
                if s.empty or _is_nearly_all_nan(history[col]):
                    continue
                metrics[col] = s

            if not metrics:
                print(f"    → no usable metrics, skipping seed")
                report["seeds"][seed]["status"] = "no_metrics"
                continue

            seed_series[seed] = metrics

        if not seed_series:
            print(f"  No usable seeds for {ag} in {project}")
            results[ag] = ({}, report)
            continue

        all_metrics: set[str] = set()
        for m in seed_series.values():
            all_metrics.update(m.keys())

        data: dict[str, pd.DataFrame] = {}
        for metric in all_metrics:
            frames = {
                f"seed_{seed}": seed_series[seed][metric]
                for seed in seed_series
                if metric in seed_series[seed]
            }
            if not frames:
                continue
            df = _align_seeds(frames)
            if df.empty:
                continue
            m = PERTASK_RE.match(metric)
            short = m.group(1) if m else "__avg__"
            data[short] = df

        n_metrics = len(data)
        print(f"  → {n_metrics} metrics ({n_metrics - (1 if '__avg__' in data else 0)} per-task)")
        results[ag] = (data, report)

    return results


def save_agent_task_set(agent: str, task_set: str, data: dict[str, pd.DataFrame]):
    out_dir = _data_dir(agent) / task_set
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        safe = re.sub(r"[^\w\-]", "_", name)
        df.to_csv(out_dir / f"{safe}.csv")
    print(f"  [{agent}] Saved {len(data)} CSVs → {out_dir}")


def load_agent_task_set(agent: str, task_set: str) -> dict[str, pd.DataFrame]:
    out_dir = _data_dir(agent) / task_set
    if not out_dir.exists():
        return {}
    result = {}
    for csv_file in sorted(out_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, index_col="step")
        result[csv_file.stem] = df
    return result


# ==============================================================================
# PLOT
# ==============================================================================

def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average; handles NaN by ignoring them."""
    if window <= 1:
        return values
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().values


def _seed_label(agent_label: str, seed_int: int, report: dict) -> str:
    info  = report.get("seeds", {}).get(seed_int, {})
    pct   = info.get("pct_complete")
    state = info.get("status", "?")
    suffix = f" ({pct:.0f}%{'*' if state == 'running' else '!'})" if pct is not None and pct < 100 else ""
    return f"{agent_label} s{seed_int}{suffix}"


def _plot_agent(
    ax,
    df: pd.DataFrame,
    agent: str,
    report: dict,
    smooth_window: int,
    show_seeds: bool,
    first_agent: bool,
):
    """Draw one agent's curves (seeds + mean ± std) onto ax."""
    cfg       = AGENTS[agent]
    color     = cfg["color"]
    label     = cfg["label"]
    seed_cols = [c for c in df.columns if c.startswith("seed_")]
    steps     = df.index.values.astype(float)

    if show_seeds:
        for col in seed_cols:
            seed_int  = int(col.split("_")[1])
            info      = report.get("seeds", {}).get(seed_int, {})
            pct       = info.get("pct_complete", 100)
            linestyle = "--" if pct < 100 else "-"
            ax.plot(
                steps / 1e6,
                smooth(df[col].values.astype(float), smooth_window),
                alpha=0.2, linewidth=0.8,
                color=color, linestyle=linestyle,
            )

    all_vals = df[seed_cols].values.astype(float)
    n_seeds  = np.sum(~np.isnan(all_vals), axis=1)
    mean_    = np.nanmean(all_vals, axis=1)
    std_     = np.nanstd(all_vals, axis=1)
    sm       = smooth(mean_, smooth_window)
    sstd     = smooth(std_,  smooth_window)

    ax.plot(steps / 1e6, sm, linewidth=2.0, color=color, label=label, zorder=5)
    ax.fill_between(steps / 1e6, sm - sstd, sm + sstd, alpha=0.15, color=color)

    # Red shade where fewer than all seeds contribute (only draw once, for first agent)
    full_seeds = len(seed_cols)
    if first_agent and np.any(n_seeds < full_seeds):
        ax.fill_between(
            steps / 1e6, *ax.get_ylim(),
            where=(n_seeds < full_seeds),
            alpha=0.06, color="red",
        )


def plot_task_set(
    task_set: str,
    data_by_agent: dict[str, dict[str, pd.DataFrame]],
    reports_by_agent: dict[str, dict],
    smooth_window: int = 5,
    max_steps: int | None = None,
    show_seeds: bool = True,
):
    # Collect all metric names across all agents
    all_metrics: set[str] = set()
    for data in data_by_agent.values():
        all_metrics.update(data.keys())

    if not all_metrics:
        print(f"  No data to plot for {task_set}")
        return

    if max_steps is not None:
        data_by_agent = {
            ag: {k: df[df.index <= max_steps] for k, df in data.items()}
            for ag, data in data_by_agent.items()
        }

    # __avg__ first, then tasks alphabetically
    tasks      = sorted(m for m in all_metrics if m != "__avg__")
    has_avg    = "__avg__" in all_metrics
    plot_items = (["__avg__"] if has_avg else []) + tasks

    total_plots = len(plot_items)
    n_cols_fig  = min(4, total_plots)
    n_rows_fig  = max(1, (total_plots + n_cols_fig - 1) // n_cols_fig)

    agent_labels = " vs ".join(AGENTS[ag]["label"] for ag in data_by_agent)
    fig, axes = plt.subplots(
        n_rows_fig, n_cols_fig,
        figsize=(5 * n_cols_fig, 4 * n_rows_fig),
        squeeze=False,
    )
    fig.suptitle(f"{agent_labels} — {task_set}", fontsize=14, fontweight="bold")

    for idx, name in enumerate(plot_items):
        ax = axes[idx // n_cols_fig][idx % n_cols_fig]

        for i, (agent, data) in enumerate(data_by_agent.items()):
            if name not in data:
                continue
            _plot_agent(
                ax, data[name], agent,
                report=reports_by_agent.get(agent, {}),
                smooth_window=smooth_window,
                show_seeds=show_seeds,
                first_agent=(i == 0),
            )

        title = "Overall avg_score" if name == "__avg__" else name
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Steps (M)", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)

        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    for idx in range(total_plots, n_rows_fig * n_cols_fig):
        axes[idx // n_cols_fig][idx % n_cols_fig].set_visible(False)

    plt.tight_layout()
    pdf_path = _plot_dir() / f"{task_set}.pdf"
    png_path = _plot_dir() / f"{task_set}.png"
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")   # vector, for paper
    fig.savefig(png_path, dpi=150, bbox_inches="tight")        # raster, for quick preview
    plt.close(fig)
    print(f"  Saved plot → {pdf_path} + {png_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download MRQ + NewT WandB runs and plot per-task scores."
    )
    parser.add_argument(
        "--task-sets", nargs="+", default=ALL_TASK_SETS,
        help="Task sets to process (default: all)"
    )
    parser.add_argument(
        "--agents", nargs="+", default=list(AGENTS.keys()),
        choices=list(AGENTS.keys()),
        help="Agents to include (default: mrq tdmpc2)"
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip WandB download; use previously saved CSVs"
    )
    parser.add_argument(
        "--include-crashed", action="store_true",
        help="Include crashed runs (killed by SLURM/OOM — likely have partial data)"
    )
    parser.add_argument(
        "--include-failed", action="store_true",
        help="Include failed runs (Python exception — likely have little/no data)"
    )
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="Smoothing window size in eval steps (default: 5)"
    )
    parser.add_argument(
        "--max-steps", type=float, default=None,
        help="Truncate plots at this many steps, e.g. 5e6 (default: plot all)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Download and save CSVs only, skip plotting"
    )
    parser.add_argument(
        "--no-seeds", action="store_true",
        help="Hide individual seed curves, show only mean ± std"
    )
    parser.add_argument(
        "--pixel", action="store_true",
        help="Use pixel observation projects (mmbench_pixel_mmbench_{task_set})"
    )
    args = parser.parse_args()

    global _OBS_MODE
    if args.pixel:
        _OBS_MODE = "pixel"

    global_reports: dict[str, dict[str, dict]] = {}  # task_set -> agent -> report

    for task_set in args.task_sets:
        print(f"\n{'='*60}")
        print(f"Task set: {task_set}")
        print(f"{'='*60}")

        data_by_agent: dict[str, dict[str, pd.DataFrame]] = {}
        reports_by_agent: dict[str, dict] = {}

        if args.no_download:
            for ag in args.agents:
                data = load_agent_task_set(ag, task_set)
                if data:
                    print(f"  [{ag}] Loaded {len(data)} cached metrics.")
                    data_by_agent[ag] = data
                else:
                    print(f"  [{ag}] No cached data found, skipping.")
            reports_by_agent = {ag: {} for ag in data_by_agent}
        else:
            fetched = fetch_task_set(
                task_set, args.agents,
                include_crashed=args.include_crashed,
                include_failed=args.include_failed,
            )
            for ag, (data, report) in fetched.items():
                reports_by_agent[ag] = report
                if data:
                    save_agent_task_set(ag, task_set, data)
                    data_by_agent[ag] = data

        global_reports[task_set] = reports_by_agent

        if not data_by_agent:
            print(f"  No data for any agent in {task_set}, skipping plot.")
            continue

        if args.no_plot:
            continue

        plot_task_set(
            task_set,
            data_by_agent,
            reports_by_agent,
            smooth_window=args.smooth,
            max_steps=int(args.max_steps) if args.max_steps else None,
            show_seeds=not args.no_seeds,
        )

    # Summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    for task_set, reports in global_reports.items():
        for ag, report in reports.items():
            seeds   = report.get("seeds", {})
            ok      = [s for s, i in seeds.items() if i.get("pct_complete", 0) >= 100]
            partial = [s for s, i in seeds.items() if 0 < i.get("pct_complete", 0) < 100]
            missing = [s for s, i in seeds.items() if i.get("status") == "missing"]
            skipped = [s for s, i in seeds.items() if i.get("skipped")]
            print(
                f"  {task_set:<18} [{AGENTS[ag]['label']:<5}]  "
                f"complete={ok}  partial={partial}  missing={missing}  skipped={skipped}"
            )

    print(f"\nDone. Plots saved to {_plot_dir()}")


if __name__ == "__main__":
    main()
