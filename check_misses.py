import pandas as pd
df = pd.read_csv("error_analysis/glioma_slice_metrics.csv")
miss = df[df.dice < 0.1]
print(miss.groupby("patient").size())
print(miss[["patient", "gt_area", "pred_area", "dice"]])