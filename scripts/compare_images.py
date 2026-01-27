import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_CSV = os.path.join(PROJECT_ROOT, 'data', 'analysis_real.csv')
AI_CSV = os.path.join(PROJECT_ROOT, 'data', 'analysis_ai.csv')

ROLES = ["doctor", "nurse", "teacher", "scientist", "chef", "artist", "builder", "driver"]


def compare():
    print("FINAL BIAS REPORT")

    # Load Data
    if not os.path.exists(REAL_CSV) or not os.path.exists(AI_CSV):
        print("Error: CSV file is missing.")
        return

    real = pd.read_csv(REAL_CSV)
    ai = pd.read_csv(AI_CSV)

    real = real[real['face_detected'] == True]
    ai = ai[ai['face_detected'] == True]

    print(f"Total Real Faces: {len(real)}")
    print(f"Total AI Faces:   {len(ai)}")

    print("\n--- % FEMALE REPRESENTATION ---")
    print(f"{'ROLE':<15} | {'REAL WORLD':<12} | {'AI WORLD':<12} | {'DIFFERENCE'}")
    print("-" * 55)

    for role in ROLES:
        # Real Stats
        r_subset = real[real['role'] == role]
        r_total = len(r_subset)

        if r_total == 0:
            r_pct = 0.0
            r_display = "N/A"
        else:
            r_women = len(r_subset[r_subset['gender'] == 'Woman'])
            r_pct = (r_women / r_total) * 100
            r_display = f"{r_pct:.1f}%"

        a_subset = ai[ai['role'] == role]
        a_total = len(a_subset)

        if a_total == 0:
            a_pct = 0.0
            a_display = "N/A"
        else:
            a_women = len(a_subset[a_subset['gender'] == 'Woman'])
            a_pct = (a_women / a_total) * 100
            a_display = f"{a_pct:.1f}%"

        if r_total > 0 and a_total > 0:
            diff = a_pct - r_pct
            diff_display = f"{diff:>+8.1f}%"
        else:
            diff_display = "    --"

        print(f"{role:<15} | {r_display:>10} | {a_display:>10} | {diff_display}")


if __name__ == "__main__":
    compare()