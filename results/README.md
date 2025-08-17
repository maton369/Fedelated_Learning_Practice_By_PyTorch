# 連合学習結果の可視化

このディレクトリには連合学習の実験結果が保存されます。

## ディレクトリ構造

```
results/
├── README.md                           # このファイル
├── training_history_clientX_YYYYMMDD_HHMMSS.csv  # 学習履歴（CSV形式）
├── detailed_results_clientX_YYYYMMDD_HHMMSS.json # 詳細結果（JSON形式）
├── server_evaluation_YYYYMMDD_HHMMSS.csv        # サーバー側評価履歴（CSV形式）
├── server_detailed_results_YYYYMMDD_HHMMSS.json # サーバー側詳細結果（JSON形式）
├── summary_report.txt                  # サマリーレポート
├── plots/                             # 生成されたグラフ
│   ├── accuracy_trends.png            # 精度推移グラフ
│   ├── loss_trends.png               # 損失推移グラフ
│   ├── client_comparison.png         # クライアント比較チャート
│   └── data_distribution.png         # データ分布チャート
├── logs/                             # ログファイル（将来使用）
└── data/                            # 生データファイル（将来使用）
```

## ファイル形式

### CSV形式（training_history_*.csv）
- **round**: ラウンド番号
- **loss**: 損失値
- **accuracy**: 精度（%）
- **samples**: サンプル数
- **timestamp**: タイムスタンプ

### JSON形式（detailed_results_*.json）
- **experiment_info**: 実験設定情報
- **training_history**: 学習履歴データ
- **summary**: 統計サマリー
- **data_distribution_details**: データ分布詳細

## 可視化の実行方法

### 1. 基本的な可視化
```bash
python visualize_results.py
```

### 2. 特定のディレクトリを指定
```bash
python visualize_results.py --results_dir /path/to/results
```

### 3. Docker環境での実行
```bash
docker-compose exec server python /app/visualize_results.py --results_dir /app/results
```

## 生成されるグラフ

1. **accuracy_trends.png**: 各クライアントの精度推移 + サーバー側集約モデルの評価精度（赤い星印）
2. **loss_trends.png**: 各クライアントの損失推移 + サーバー側集約モデルの評価損失（赤い星印）
3. **client_comparison.png**: クライアント間の性能比較 + サーバー結果（赤いバー）
4. **data_distribution.png**: 各クライアントのデータ分布

## カスタム分析

CSVファイルとJSONファイルは標準的な形式なので、以下のツールで分析可能です：

- **Python**: pandas, matplotlib, seaborn
- **R**: readr, ggplot2, dplyr
- **Excel**: CSVファイルを直接開いて分析
- **Jupyter Notebook**: 対話的な分析

## サンプルコード（Python）

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込み
df = pd.read_csv('training_history_client1_20231201_120000.csv')

# 精度推移をプロット
plt.plot(df['round'], df['accuracy'])
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Progress')
plt.show()
```