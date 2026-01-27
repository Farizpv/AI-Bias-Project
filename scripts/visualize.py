import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CSV = os.path.join(PROJECT_ROOT, 'data', 'analysis_real.csv')
AI_CSV = os.path.join(PROJECT_ROOT, 'data', 'analysis_ai.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data')

# Set the style for professional academic charts
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Roles to ensure we keep the order consistent
ROLES_ORDER = ["doctor", "nurse", "teacher", "scientist", "chef", "artist", "builder", "driver"]


def get_gender_stats(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None

    df = pd.read_csv(csv_path)
    # Filter for valid faces
    df = df[df['face_detected'] == True]

    stats = []
    for role in ROLES_ORDER:
        subset = df[df['role'] == role]
        total = len(subset)
        if total == 0:
            pct = 0
        else:
            women = len(subset[subset['gender'] == 'Woman'])
            pct = (women / total) * 100
        stats.append({'Role': role.title(), 'Female_Pct': pct})

    return pd.DataFrame(stats)


def plot_individual_chart(df, title, color, filename):
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Role', y='Female_Pct', data=df, color=color, edgecolor='black')

    # Add labels
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('% Female Representation', fontsize=12)
    plt.xlabel('Role', fontsize=12)
    plt.ylim(0, 100)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)  # Parity line

    # Add numbers on bars
    for i, v in enumerate(df['Female_Pct']):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {filename}")
    plt.close()


def plot_comparison_chart(real_df, ai_df):
    real_df['Source'] = 'Real World (Flickr)'
    ai_df['Source'] = 'AI Generated (SD v1.5)'
    combined = pd.concat([real_df, ai_df])

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Role', y='Female_Pct', hue='Source', data=combined,
                     palette={'Real World (Flickr)': '#3498db', 'AI Generated (SD v1.5)': '#e74c3c'},
                     edgecolor='black')

    plt.title('Bias Comparison: Real World vs. AI', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('% Female Representation', fontsize=12)
    plt.xlabel('')
    plt.ylim(0, 100)
    plt.legend(title=None)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'bias_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved: bias_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Generating Project Charts...")

    # 1. Get Data
    real_data = get_gender_stats(REAL_CSV)
    ai_data = get_gender_stats(AI_CSV)

    if real_data is not None and ai_data is not None:
        # 2. Make Real World Chart (Blue)
        plot_individual_chart(real_data,
                              "Real World Baseline (Flickr30k)",
                              "#3498db",
                              "chart_real_world_baseline.png")

        # 3. Make AI World Chart (Red)
        plot_individual_chart(ai_data,
                              "AI Generated Distribution (Stable Diffusion)",
                              "#e74c3c",
                              "chart_ai_generated.png")

        # 4. Make Comparison Chart
        plot_comparison_chart(real_data, ai_data)

        print("\nAll 3 charts saved to 'data/' folder!")