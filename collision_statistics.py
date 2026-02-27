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

        mean = df["collisions"].mean()
        std = df["collisions"].std()
        goal_mean = df["goal_reached"].mean() if "goal_reached" in df.columns else float("nan")
        
        avg_spl = df["SPL"].mean() if "SPL" in df.columns else float("nan")
        avg_spl_success = float("nan")
        std_spl_success = float("nan")
        
        avg_speed_mean = df["average_speed"].mean() if "average_speed" in df.columns else float("nan")
        avg_speed_std = df["average_speed"].std() if "average_speed" in df.columns else float("nan")
        
        if "SPL" in df.columns:
            spl_data[folder] = df["SPL"].dropna().values

        if "SPL" in df.columns and "goal_reached" in df.columns:
            success_df = df[df["goal_reached"] == True]
            if not success_df.empty:
                avg_spl_success = success_df["SPL"].mean()
                std_spl_success = success_df["SPL"].std()

        rows.append({
            "folder": folder,
            "collisions_mean": round(mean, 4),
            "collisions_std": round(std, 4),
            "success_rate": round(goal_mean, 4),
            "Average SPL": round(avg_spl, 4),
            "Average SPL | Success": round(avg_spl_success, 4),
            "Std SPL | Success": round(std_spl_success, 4),
            "Average Speed Mean": round(avg_speed_mean, 4) if not math.isnan(avg_speed_mean) else float("nan"),
            "Average Speed Std": round(avg_speed_std, 4) if not math.isnan(avg_speed_std) else float("nan"),
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
            
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]# 4 colors for 4 columns
        used_indices = set()
        
        for i, (folder, data) in enumerate(spl_data.items()):
            row_idx = i % rows_count
            col_idx = i // rows_count
            flat_idx = row_idx * cols + col_idx
            
            ax = axes[flat_idx]
            used_indices.add(flat_idx)
            color = colors[row_idx % len(colors)]
            
            if len(data) > 0:
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
                    kde = gaussian_kde(data, bw_method=0.05)
                    x_eval = np.linspace(max(0, min(data) - 0.1), max(data) + 0.1, 200)
                    ax.plot(x_eval, kde(x_eval), color=color, linewidth=2, label=label_name)
                    # ax.fill_between(x_eval, kde(x_eval), color=color, alpha=0.1)
            
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
            fig_acc, axes_acc = plt.subplots(rows_count, 1, figsize=(8, 2.5 * rows_count), sharex=True)
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
            plt.close()


if __name__ == "__main__":
    
    base_dir = "eval_data"
    
    folders =   [
        "connectome_full_vision", "connectome_left_eye", "connectome_right_eye", "connectome_blind",
        "vision_efficientnet_robust_full_vision", "vision_efficientnet_robust_left_eye", "vision_efficientnet_robust_right_eye", "vision_efficientnet_robust_blind",
        "vision_mobilenet_robust_full_vision", "vision_mobilenet_robust_left_eye", "vision_mobilenet_robust_right_eye", "vision_mobilenet_robust_blind",
        "small_world_full_vision", "small_world_left_eye", "small_world_right_eye", "small_world_total_blind",
    ]
    
    folders_full_path = []
    for folder in folders:
        folders_full_path.append(os.path.join(base_dir, folder))
    
    collision_statistics(folders_full_path)
