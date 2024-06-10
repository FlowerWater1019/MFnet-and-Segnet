import os
import re
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

img_path = './img'
os.makedirs(img_path, exist_ok=True)


def plot_origin():
    columns = ["name", "train", "val", "test"]
    table_str = """
    | name | train | val | test |
| :-: | :-: | :-: | :-: |
| Seg-rgb | 79.83%/58.84% | 52.85%/50.07% | 56.19%/49.67% |
| Seg-inf | 78.72%/51.24% | 36.94%/36.07% | 49.33%/38.22% |
| Seg-mix | 84.51%/62.37% | 54.62%/50.38% | 56.41%/50.61% |
| MF      | 84.45%/82.15% | 53.83%/62.54% | 60.79%/57.70% |
| MF-advtrain | 54.22%/62.88% | 25.13%/30.07% | 29.13%/27.49% |
"""
    pattern = r"\| (.*?) \| (.*?)%\/(.*?)% \| (.*?)%\/(.*?)% \| (.*?)%\/(.*?)% \|"
    matches = re.findall(pattern, table_str)

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
    plt.savefig(img_path + "/origin_heatmap.png", dpi=600)
    plt.show()


def plot_attack(args):
    if args.atk == 'PGD':
        columns = ["name", "limit", "e,a,s=4,1,10", "e,a,s=8,1,10", "e,a,s=8,1,20", "e,a,s=16,1,20"]
        table_str = """
    | name | limit | e=4,a=1,s=10 | e=8,a=1,s=10 | e=8,a=1,s=20 | e=16,a=1,s=20 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Seg-rgb |   -    | 10.28%/8.90% | 5.76%/6.70% | 11.79%/4.27% | 6.45%/2.81% | 
| Seg-inf |   -    | 9.17%/4.89% | 6.28%/3.40% | 11.00%/2.89% | 13.40%/2.94% | 
| Seg-mix |   -    | 15.75%/8.64% | 14.09%/7.52% | 15.96%/5.31% | 11.88%/4.74% | 
| Seg-mix |  rgb   | 16.72%/11.43% | 14.32%/9.30% | 15.53%/6.04% | 10.93%/4.97% | 
| Seg-mix |  inf   | 23.30%/21.93% | 18.33%/17.93% | 15.09%/13.53% | 10.97%/8.61% | 
| MF      |  -     | 12.93%/11.87% | 11.27%/6.70% | 12.64%/5.32% | 9.25%/3.40% | 
| MF      |  rgb   | 15.30%/16.04% | 12.38%/8.19% | 12.19%/6.25% | 7.34%/4.05% | 
| MF      |  inf   | 26.09%/35.67% | 20.45%/19.87% | 18.15%/16.87% | 10.07%/9.32% | 
| MF-advtrain | -  | 22.86%/00.00% | 19.44%/00.00% | 17.44%/00.00% | 12.81%/15.33% |
"""
    else:   # 'FGSM'
        columns = ["name", "limit", "eps=4", "eps=8", "eps=12", "eps=16"]
        table_str = """
        | name | limit | eps=4 | eps=8 | eps=12 | eps=16 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Seg-rgb |   -    | 18.30%/23.97% | 12.11%/23.02% | 8.24%/22.80% | 5.07%/23.31% |
| Seg-inf |   -    | 11.06%/23.07% | 4.36%/18.99% | 1.42%/19.56% | 0.22%/21.01% |
| Seg-mix |   -    | 22.73%/16.89% | 19.41%/12.00% | 17.47%/10.56% | 15.83%/9.81% |
| Seg-mix |  rgb   | 24.99%/20.08% | 21.80%/14.27% | 20.01%/11.91% | 18.64%/10.75% |
| Seg-mix |  inf   | 32.43%/34.67% | 27.00%/29.19% | 24.25%/22.66% | 22.25%/20.12% |
| MF      |  -     | 21.84%/22.43% | 16.31%/12.59% | 13.46%/9.82% | 11.72%/ 9.62% |
| MF      |  rgb   | 28.01%/25.91% | 23.85%/18.06% | 22.16%/16.06% | 20.92%/15.01% |
| MF      |  inf   | 47.78%/51.14% | 44.24%/49.72% | 39.85%/42.84% | 37.61%/34.62% |
| MF-advtrain | -  | 22.89%/00.00% | 18.22%/00.00% | 14.81%/15.95% | 12.42%/14.12% |
"""
    pattern = r"\| (.*?) \| (.*?) \| (.*?)%\/(.*?)% \| (.*?)%\/(.*?)% \| (.*?)%\/(.*?)% \| (.*?)%\/(.*?)% \|"
    matches = re.findall(pattern, table_str)
    data_acc = {col: [] for col in columns}
    data_miou = {col: [] for col in columns}

    for match in matches:
        for i, col in enumerate(columns):
            if i <= 1:
                data_acc[col].append(match[i])
                data_miou[col].append(match[i])
            else:
                data_acc[col].append(float(match[2*i - 2]))
                data_miou[col].append(float(match[2*i - 1]))
            
    df_acc = pd.DataFrame(data_acc)
    df_miou = pd.DataFrame(data_miou)
    df_acc.set_index(["name", "limit"], inplace=True)
    df_miou.set_index(["name", "limit"], inplace=True)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(df_acc, annot=True, cmap="viridis", fmt=".2f")
    plt.title("Accuracy Heatmap")
    plt.subplot(1, 2, 2)
    sns.heatmap(df_miou, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("mIoU Heatmap")
    plt.savefig(img_path + f"/{args.atk}_heatmap.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data from markdown table')
    parser.add_argument('-atk', type=str, default='origin', choices=['origin', 'FGSM', 'PGD'])
    args = parser.parse_args()
    if args.atk == 'origin':
        plot_origin()
    else: 
        plot_attack(args)
