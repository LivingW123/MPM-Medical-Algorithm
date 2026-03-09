import pandas as pd
import os

data_dir = r"c:\Users\oneal\VS Code Stuff\MPM-Medical-Algorithm\Manufacturing Data CSVs"

report = "# Manufacturing Data Summary\n\n"

for f in sorted(os.listdir(data_dir)):
    if f.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join(data_dir, f), nrows=5)
            # Remove entirely empty or Unnamed columns just for cleaner display in report
            df_clean = df.dropna(axis=1, how='all')
            columns = df_clean.columns.tolist()
            report += f"## {f}\n"
            report += f"- **Shape**: (rows unknown, {df.shape[1]} columns)\n"
            report += f"- **Columns**: {', '.join([str(c) for c in columns])}\n"
            report += f"\n```\n{df_clean.head(2).to_markdown()}\n```\n\n"
        except Exception as e:
            report += f"## {f}\nError reading: {e}\n\n"

with open(r"c:\Users\oneal\VS Code Stuff\MPM-Medical-Algorithm\data_summary.md", "w") as out:
    out.write(report)
print("Data summary generated.")
