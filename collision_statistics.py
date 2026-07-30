"""
Calculate mean and standard deviation of collisions from episode_summary.csv
across multiple evaluation folders.
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---- Global plot style ----
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24

BASE_DIR = os.path.join(os.path.dirname(__file__), "eval_data")

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "collision_statistics.csv")

COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
folders = sorted(
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d))
)

def collision_statistics(folders = folders):
    rows = []
    spl_data = {}
   
    for folder in folders:
        csv_path = os.path.join(folder, "episode_summary.csv")
        if not os.path.exists(csv_path):
            print(f"[SKIP] {folder}: episode_summary.csv not found")
            continue

        df = pd.read_csv(csv_path)
        if "collisions" not in df.columns:
            print(f"[SKIP] {folder}: 'collisions' column not found")
            continue
            
        if "obstacle_in_line" in df.columns:
            df = df[df["obstacle_in_line"] == True]

        mean = df["collisions"].mean()
        std = df["collisions"].std()
        goal_mean = df["goal_reached"].mean() if "goal_reached" in df.columns else float("nan")
        
        avg_spl = df["SPL"].mean() if "SPL" in df.columns else float("nan")
        avg_spl_success = float("nan")
        std_spl_success = float("nan")
        avg_col_success = float("nan")
        std_col_success = float("nan")
        
        avg_speed_mean = df["average_speed"].mean() if "average_speed" in df.columns else float("nan")
        avg_speed_std = df["average_speed"].std() if "average_speed" in df.columns else float("nan")
        
        time_to_goal_mean = float("nan")
        time_to_goal_std = float("nan")
        if "steps" in df.columns and "goal_reached" in df.columns:
            steps_array = np.where(df["goal_reached"] == False, 600, df["steps"])
            time_to_goal = 0.02 * steps_array
            time_to_goal_mean = time_to_goal.mean()
            time_to_goal_std = time_to_goal.std()
        
        if "SPL" in df.columns:
            spl_data[folder] = df["SPL"].dropna().values

        if "goal_reached" in df.columns:
            success_df = df[df["goal_reached"] == True]
            if not success_df.empty:
                if "SPL" in success_df.columns:
                    avg_spl_success = success_df["SPL"].mean()
                    std_spl_success = success_df["SPL"].std()
                if "collisions" in success_df.columns:
                    avg_col_success = success_df["collisions"].mean()
                    std_col_success = success_df["collisions"].std()

        rows.append({
            "folder": folder,
            "collisions_mean": round(mean, 4),
            "collisions_std": round(std, 4),
            "success_rate": round(goal_mean, 4),
            "Average SPL": round(avg_spl, 4),
            "Average SPL | Success": round(avg_spl_success, 4),
            "Std SPL | Success": round(std_spl_success, 4),
            "Collisions | Success": round(avg_col_success, 4),
            "Collisions Std | Success": round(std_col_success, 4),
            "Average Speed Mean": round(avg_speed_mean, 4) if not math.isnan(avg_speed_mean) else float("nan"),
            "Average Speed Std": round(avg_speed_std, 4) if not math.isnan(avg_speed_std) else float("nan"),
            "Time to Goal Mean": round(time_to_goal_mean, 4) if not math.isnan(time_to_goal_mean) else float("nan"),
            "Time to Goal Std": round(time_to_goal_std, 4) if not math.isnan(time_to_goal_std) else float("nan"),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results for {len(rows)} folders to {OUTPUT_CSV}")

    # Plotting SPL violin plots
    if spl_data:
        num_plots = len(spl_data)
        cols = 4
        rows_count = math.ceil(num_plots / cols)
        fig, axes = plt.subplots(rows_count, cols, figsize=(4 * cols, 4 * rows_count), sharex=True, sharey=True)
        
        # Handle the case where there is only one row/column correctly
        if rows_count == 1 and cols == 1:
            axes = [axes]
        elif rows_count == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        colors = COLORS# 4 colors for 4 columns
        used_indices = set()
        
        for i, (folder, data) in enumerate(spl_data.items()):
            row_idx = i % rows_count
            col_idx = i // rows_count
            flat_idx = row_idx * cols + col_idx
            
            ax = axes[flat_idx]
            used_indices.add(flat_idx)
            color = colors[row_idx % len(colors)]
            
            if len(data) > 0:
                try:
                    parts = ax.violinplot(data, showmedians=True, bw_method=0.05)
                    for pc in parts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_edgecolor('black')
                        pc.set_alpha(0.7)
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                        if partname in parts:
                            vp = parts[partname]
                            vp.set_edgecolor(color)
                            vp.set_linewidth(4.0)
                except (np.linalg.LinAlgError, ValueError):
                    pass
            
            ax.set_title(folder, fontsize=10)
            ax.axis('off')
            
        # Hide any unused subplots
        for i in range(rows_count * cols):
            if i not in used_indices:
                fig.delaxes(axes[i])
            
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), "spl_violin_plots.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Saved SPL violin plots to {plot_path}")
        plt.close()

        # Plotting SPL histograms
        if spl_data:
            fig_hist, axes_hist = plt.subplots(rows_count, 1, figsize=(8, 2.5 * rows_count), sharex=True)
            if rows_count == 1:
                axes_hist = [axes_hist]
                
            for i, (folder, data) in enumerate(spl_data.items()):
                row_idx = i % rows_count
                col_idx = i // rows_count
                
                ax = axes_hist[row_idx]
                color = colors[col_idx % len(colors)]
                
                label_name = os.path.basename(folder)
                
                if len(data) > 1:
                    try:
                        kde = gaussian_kde(data, bw_method=0.05)
                        x_eval = np.linspace(max(0, min(data) - 0.1), max(data) + 0.1, 200)
                        ax.plot(x_eval, kde(x_eval), color=color, linewidth=2, label=label_name)
                        # ax.fill_between(x_eval, kde(x_eval), color=color, alpha=0.1)
                    except (np.linalg.LinAlgError, ValueError):
                        val = data[0] if len(data) > 0 else 0
                        ax.axvline(val, color=color, linewidth=2, label=label_name)
            
            for r in range(rows_count):
                axes_hist[r].legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
                axes_hist[r].set_ylabel("Density")
                axes_hist[r].set_yticks([])
                
            axes_hist[-1].set_xlabel("SPL")
            
            plt.tight_layout()
            hist_plot_path = os.path.join(os.path.dirname(__file__), "spl_histograms.png")
            plt.savefig(hist_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved SPL histograms to {hist_plot_path}")
            plt.close()

        # Plotting SPL accumulated histograms (1 down to 0 CDF)
        if spl_data:
            fig_acc, axes_acc = plt.subplots(rows_count, 1, figsize=(8, 4.2 * rows_count), sharex=True)
            if rows_count == 1:
                axes_acc = [axes_acc]
                
            for i, (folder, data) in enumerate(spl_data.items()):
                row_idx = i % rows_count
                col_idx = i // rows_count
                
                ax = axes_acc[row_idx]
                color = colors[col_idx % len(colors)]
                label_name = os.path.basename(folder)
                
                if len(data) > 0:
                    # we want x from 1 down to 0, or sort data and plot reverse cum-density
                    x_eval = np.linspace(1.0, 0.0, 500)
                    y_eval = np.array([(data >= x).mean() for x in x_eval])
                    ax.plot(x_eval, y_eval, color=color, linewidth=2, label=label_name)
            
            for r in range(rows_count):
                # axes_acc[r].legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
                # axes_acc[r].set_ylabel("Proportion >= SPL")
                # Flip x axis so it goes from 1.0 down to 0.0
                axes_acc[r].set_xlim(1.0, 0.0)
                
            axes_acc[-1].set_xlabel("SPL")
            
            plt.tight_layout()
            acc_plot_path = os.path.join(os.path.dirname(__file__), "spl_accumulated_histograms.png")
            plt.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved SPL accumulated histograms to {acc_plot_path}")
            plt.close(fig_acc)

    # Grouped bar-chart figures only need out_df's aggregate columns (not spl_data's raw
    # per-episode arrays), so this runs unconditionally rather than nested under `if spl_data:`.
    plot_grouped_bars(out_df)


# --------------------------------------------------------------------------
# Grouped Bar Plots for Metrics
# --------------------------------------------------------------------------
def plot_grouped_bars(out_df, output_dir=None):
    """Draw the 6 grouped bar-chart figures (collisions/success/SPL/speed/duration across
    4 models x 4 vision conditions) from an aggregate stats DataFrame with the same row
    layout collision_statistics() produces (16 rows: model-major, condition-minor).
    Split out from collision_statistics() so it can also be re-run directly from an
    already-computed stats CSV via regenerate_bars_from_stats_csv() below, e.g. when the
    raw eval_data folders that produced that CSV have since been deleted/rotated.
    """
    out_dir = output_dir or os.path.dirname(__file__)
    models = ["FLYNN", "EfficientNet", "MobileNet", "SmallWorldNet"]

    # User defined conditions and order
    display_conditions = ["Full vision", "Left eye only", "Right eye only", "Total blindness"]
    # Map to their row offsets within each group of 4 (model block)
    # Full: 0, Left: 1, Right: 2, Blind: 3
    condition_mapping = {
        "Full vision": 0,
        "Right eye only": 2,
        "Left eye only": 1,
        "Total blindness": 3
    }

    condition_colors = COLORS

    plot_configs = [
        {"title": "Collisions", "y_col": "collisions_mean", "err_col": "collisions_std", "stacked_col": "Collisions | Success", "stacked_err_col": "Collisions Std | Success", "filename": "bar_collisions.png"},
        {"title": "Success Rate", "y_col": "success_rate", "err_col": None, "filename": "bar_success_rate.png"},
        {"title": "Average SPL", "y_col": "Average SPL", "err_col": None, "filename": "bar_average_spl.png"},
        {"title": "Average SPL | Success", "y_col": "Average SPL | Success", "err_col": "Std SPL | Success", "filename": "bar_average_spl_success.png"},
        {"title": "Average Speed", "y_col": "Average Speed Mean", "err_col": "Average Speed Std", "filename": "bar_average_speed.png"},
        {"title": "Episode Duration (s)", "y_col": "Time to Goal Mean", "err_col": "Time to Goal Std", "filename": "bar_time_to_goal.png"},
    ]

    for pcfg in plot_configs:
        if pcfg["y_col"] not in out_df.columns:
            print(f"[SKIP] {pcfg['title']}: column '{pcfg['y_col']}' not in stats DataFrame")
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        x_groups = np.arange(len(models))
        width = 0.2

        has_stack = "stacked_col" in pcfg and pcfg["stacked_col"] in out_df.columns

        for cond_idx, cond_name in enumerate(display_conditions):
            data_cond_offset = condition_mapping[cond_name]
            color = condition_colors[cond_idx]

            y_means = []
            y_errs = []
            y_stacks = []
            y_stack_errs = []

            for model_idx, model in enumerate(models):
                row_idx = model_idx * 4 + data_cond_offset
                val, err = 0, 0
                s_val, s_err = 0, 0

                if row_idx < len(out_df):
                    val = out_df.iloc[row_idx][pcfg["y_col"]]
                    if pcfg.get("err_col") and not pd.isna(out_df.iloc[row_idx][pcfg["err_col"]]):
                        err = out_df.iloc[row_idx][pcfg["err_col"]]

                    if has_stack:
                        s_val = out_df.iloc[row_idx][pcfg["stacked_col"]]
                        if pcfg.get("stacked_err_col") and not pd.isna(out_df.iloc[row_idx][pcfg["stacked_err_col"]]):
                            s_err = out_df.iloc[row_idx][pcfg["stacked_err_col"]]
                        if pd.isna(s_val):
                            s_val = 0

                if pd.isna(val):
                    val = 0

                y_means.append(val)
                y_errs.append(err)
                y_stacks.append(s_val)
                y_stack_errs.append(s_err)

            pos = x_groups + (cond_idx - 1.5) * width

            if has_stack:
                # Plot bottom stack (Collisions in Success)
                ax.bar(pos, y_stacks, width, color=color, edgecolor='black', label=cond_name)
                # Plot top stack (Failures / Remaining Collisions)
                top_heights = [max(0, m - s) for m, s in zip(y_means, y_stacks)]
                top_kwargs = dict(width=width, bottom=y_stacks, color=color, alpha=0.5, hatch='//', edgecolor='black')

                if pcfg.get("err_col"):
                    ax.bar(pos, top_heights, yerr=y_errs, capsize=4, **top_kwargs)
                else:
                    ax.bar(pos, top_heights, **top_kwargs)
            else:
                if pcfg.get("err_col"):
                    ax.bar(pos, y_means, width, yerr=y_errs, color=color, label=cond_name, edgecolor='black', capsize=4)
                else:
                    ax.bar(pos, y_means, width, color=color, label=cond_name, edgecolor='black')

        ax.set_ylabel(pcfg.get("display_y", pcfg["title"]))
        # ax.set_title(pcfg["title"])
        ax.set_xticks(x_groups)
        ax.set_xticklabels(models, rotation=0, ha='center')

        handles, labels = ax.get_legend_handles_labels()
        # If stacked, add extra patches to explain hatching
        if has_stack:
            from matplotlib.patches import Patch
            stack_handles = [
                Patch(facecolor='grey', edgecolor='black', label='Success Avg'),
                Patch(facecolor='grey', alpha=0.5, hatch='//', edgecolor='black', label='Failure Avg (Remaining)')
            ]
            handles.extend(stack_handles)
            labels.extend(['Success Avg', 'Failure Avg'])

        ax.legend(handles=handles, labels=labels, fontsize=12, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plot_path = os.path.join(out_dir, pcfg["filename"])
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved {pcfg['title']} plot to {plot_path}")
        plt.close(fig)


def regenerate_bars_from_stats_csv(csv_path, output_dir=None):
    """Regenerate the grouped bar-chart figures directly from an already-computed aggregate
    stats CSV (collision_statistics()'s OUTPUT_CSV row layout: 16 rows, model-major,
    condition-minor), without needing the underlying per-episode eval_data folders.

    Useful when the raw eval_data run that produced a given CSV has since been deleted or
    rotated away (as happened here -- see CLEANUP_PLAN.md Progress log / former §1.4):
    eval_data/ no longer contains the condition-named folders (connectome_full_vision, etc.)
    that either of collision_statistics.py's __main__ folder lists expect, so the root-level
    bar_*.png had regressed to showing only the 4-folder OOD sweep with empty bars for 3 of 4
    models. collision_statistics_checker_texture.csv still holds the correct, previously
    -computed 16-row checkerboard aggregate, so this reconstructs the bar charts from that
    instead of re-running (expensive, and no longer possible without fresh eval_data) rollouts.
    """
    out_df = pd.read_csv(csv_path)
    plot_grouped_bars(out_df, output_dir=output_dir)


if __name__ == "__main__":
    
    base_dir = "eval_data"
    
    folders =   [
        # "connectome_full_vision", "connectome_left_eye", "connectome_right_eye", "connectome_blind",
        # "vision_efficientnet_robust_full_vision", "vision_efficientnet_robust_left_eye", "vision_efficientnet_robust_right_eye", "vision_efficientnet_robust_blind",
        # "vision_mobilenet_robust_full_vision", "vision_mobilenet_robust_left_eye", "vision_mobilenet_robust_right_eye", "vision_mobilenet_robust_blind",
        # "small_world_full_vision", "small_world_left_eye", "small_world_right_eye", "small_world_total_blind",
    ]
    # folders =   [
    #     # "connectome_full_vision_4", "connectome_left_eye_4", "connectome_right_eye_4", "connectome_blind_4",
    #     "connectome_full_vision", "connectome_left_eye", "connectome_right_eye", "connectome_blind",
    #     "vision_efficientnet_robust_full_vision", "vision_efficientnet_robust_left_eye", "vision_efficientnet_robust_right_eye", "vision_efficientnet_robust_blind",
    #     "vision_mobilenet_robust_full_vision", "vision_mobilenet_robust_left_eye", "vision_mobilenet_robust_right_eye", "vision_mobilenet_robust_blind",
    #     "small_world_full_vision", "small_world_left_eye", "small_world_right_eye", "small_world_total_blind",
    # ]
    folders = ["connectome_textured_env", "small_world_textured_env", "vision_efficientnet_robust_textured_env", "vision_mobilenet_robust_textured_env"]
    
    folders_full_path = []
    for folder in folders:
        folders_full_path.append(os.path.join(base_dir, folder))

    existing_folders = [f for f in folders_full_path if os.path.isdir(f)]
    if not existing_folders:
        # eval_data/ has drifted to timestamped run dirs (connectome_rnn_<timestamp>, etc.)
        # instead of the condition-named folders listed above -- skip rather than silently
        # overwriting OUTPUT_CSV/the root bar_*.png with an empty result.
        print(f"[WARN] None of {folders_full_path} exist under eval_data/ -- skipping "
              f"collision_statistics() to avoid clobbering {OUTPUT_CSV} with empty data.")
    else:
        if len(existing_folders) < len(folders_full_path):
            print(f"[WARN] Only {len(existing_folders)}/{len(folders_full_path)} configured "
                  f"folders exist; proceeding with those.")
        collision_statistics(existing_folders)

    # Regenerate the checkerboard-condition bar charts (the paper's main "performance across
    # vision conditions" figure) directly from the already-computed aggregate CSV -- the raw
    # eval_data folders that produced it no longer exist (see regenerate_bars_from_stats_csv's
    # docstring), so this restores the root-level bar_*.png rather than recomputing from scratch.
    checker_csv = os.path.join(os.path.dirname(__file__), "collision_statistics_checker_texture.csv")
    if os.path.exists(checker_csv):
        regenerate_bars_from_stats_csv(checker_csv)
    else:
        print(f"[WARN] {checker_csv} not found -- cannot regenerate checkerboard bar charts.")
