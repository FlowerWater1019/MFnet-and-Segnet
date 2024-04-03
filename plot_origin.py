import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    table_str = """
    | name | train | val | test |
| :-: | :-: | :-: | :-: |
| Seg-rgb | 79.83%/58.84% | 52.85%/50.07% | 56.19%/49.67% |
| Seg-inf | 78.72%/51.24% | 36.94%/36.07% | 49.33%/38.22% |
| Seg-mix | 84.51%/62.37% | 54.62%/50.38% | 56.41%/50.61% |
| MF      | 84.45%/82.15% | 53.83%/62.54% | 60.79%/57.70% |
"""
    pattern = r"\| (.*?) \| (.*?)%\/(.*?)% \| (.*?)%\/(.*?)% \| (.*?)%\/(.*?)% \|"
    matches = re.findall(pattern, table_str)
    
    columns = ["name", "train", "val", "test"]
    data_acc = {col: [] for col in columns}
    data_miou = {col: [] for col in columns}

    for match in matches:
        for i, col in enumerate(columns):
            if i == 0:
                data_acc[col].append(match[i])
                data_miou[col].append(match[i])
            else:
                data_acc[col].append(float(match[2*i - 1]))
                data_miou[col].append(float(match[2*i]))
            
    df_acc = pd.DataFrame(data_acc)
    df_miou = pd.DataFrame(data_miou)
    df_acc.set_index("name", inplace=True)
    df_miou.set_index("name", inplace=True)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(df_acc, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Accuracy Heatmap")
    plt.subplot(1, 2, 2)
    sns.heatmap(df_miou, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("mIoU Heatmap")

    plt.savefig("img_result\origin_heatmap.png", dpi=600)
    plt.show()

if __name__ == "__main__":
    main()