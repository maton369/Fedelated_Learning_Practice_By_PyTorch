#!/usr/bin/env python3
"""
連合学習の結果を可視化するスクリプト

使用方法:
    python visualize_results.py --results_dir ./results

要件:
    pip install matplotlib seaborn pandas
"""

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import numpy as np
from datetime import datetime

# 日本語フォント設定
plt.rcParams["font.family"] = [
    "Arial Unicode MS",
    "Hiragino Sans",
    "Yu Gothic",
    "Meiryo",
    "Takao",
    "IPAexGothic",
    "IPAPGothic",
    "VL PGothic",
    "Noto Sans CJK JP",
]


class FederatedLearningVisualizer:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.plot_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def load_csv_results(self):
        """CSVファイルから学習履歴を読み込み"""
        csv_files = glob(os.path.join(self.results_dir, "training_history_*.csv"))

        if not csv_files:
            print("CSVファイルが見つかりません")
            return None

        all_data = []
        for csv_file in csv_files:
            # ファイル名からクライアントIDを抽出
            filename = os.path.basename(csv_file)
            client_id = filename.split("_")[2]  # training_history_client1_timestamp.csv

            df = pd.read_csv(csv_file)
            df["client_id"] = client_id
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True) if all_data else None

    def load_json_results(self):
        """JSONファイルから詳細結果を読み込み"""
        json_files = glob(os.path.join(self.results_dir, "detailed_results_*.json"))

        results = {}
        for json_file in json_files:
            filename = os.path.basename(json_file)
            client_id = filename.split("_")[
                2
            ]  # detailed_results_client1_timestamp.json

            with open(json_file, "r", encoding="utf-8") as f:
                results[client_id] = json.load(f)

        return results

    def load_server_results(self):
        """サーバー側評価結果を読み込み"""
        # CSVファイル
        server_csv_files = glob(
            os.path.join(self.results_dir, "server_evaluation_*.csv")
        )
        server_df = None
        if server_csv_files:
            server_df = pd.read_csv(server_csv_files[-1])  # 最新のファイルを使用

        # JSONファイル
        server_json_files = glob(
            os.path.join(self.results_dir, "server_detailed_results_*.json")
        )
        server_json = None
        if server_json_files:
            with open(server_json_files[-1], "r", encoding="utf-8") as f:
                server_json = json.load(f)

        return server_df, server_json

    def plot_accuracy_trends(self, df, server_df=None):
        """精度の推移をプロット（サーバー評価を含む）"""
        plt.figure(figsize=(14, 8))

        # 各クライアントの精度推移
        for client in df["client_id"].unique():
            client_data = df[df["client_id"] == client]
            plt.plot(
                client_data["round"],
                client_data["accuracy"],
                marker="o",
                linewidth=2,
                markersize=6,
                label=f"クライアント {client}",
            )

        # サーバー側評価を追加
        if server_df is not None and not server_df.empty:
            plt.plot(
                server_df["round"],
                server_df["accuracy"],
                marker="*",
                linewidth=3,
                markersize=10,
                color="red",
                label="サーバー側評価（集約モデル）",
            )

        plt.title("連合学習における精度の推移", fontsize=16, fontweight="bold")
        plt.xlabel("ラウンド", fontsize=12)
        plt.ylabel("精度 (%)", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self.plot_dir, "accuracy_trends.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"精度推移グラフを保存しました: {plot_path}")

    def plot_loss_trends(self, df, server_df=None):
        """損失の推移をプロット（サーバー評価を含む）"""
        plt.figure(figsize=(14, 8))

        for client in df["client_id"].unique():
            client_data = df[df["client_id"] == client]
            plt.plot(
                client_data["round"],
                client_data["loss"],
                marker="s",
                linewidth=2,
                markersize=6,
                label=f"クライアント {client}",
            )

        # サーバー側評価を追加
        if server_df is not None and not server_df.empty:
            plt.plot(
                server_df["round"],
                server_df["loss"],
                marker="*",
                linewidth=3,
                markersize=10,
                color="red",
                label="サーバー側評価（集約モデル）",
            )

        plt.title("連合学習における損失の推移", fontsize=16, fontweight="bold")
        plt.xlabel("ラウンド", fontsize=12)
        plt.ylabel("損失", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self.plot_dir, "loss_trends.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"損失推移グラフを保存しました: {plot_path}")

    def plot_comparison_chart(self, json_results, server_json=None):
        """クライアント間の比較チャート（サーバー結果を含む）"""
        if not json_results:
            return

        clients = list(json_results.keys())
        metrics = {
            "final_accuracy": [],
            "accuracy_improvement": [],
            "best_accuracy": [],
            "final_loss": [],
        }

        for client in clients:
            summary = json_results[client]["summary"]
            metrics["final_accuracy"].append(summary["final_accuracy"])
            metrics["accuracy_improvement"].append(summary["accuracy_improvement"])
            metrics["best_accuracy"].append(summary["best_accuracy"])
            metrics["final_loss"].append(summary["final_loss"])

        # サーバー結果を追加
        if server_json:
            clients.append("サーバー")
            server_summary = server_json["summary"]
            metrics["final_accuracy"].append(server_summary["final_accuracy"])
            metrics["accuracy_improvement"].append(
                server_summary["accuracy_improvement"]
            )
            metrics["best_accuracy"].append(server_summary["best_accuracy"])
            metrics["final_loss"].append(server_summary["best_loss"])

        # 比較チャート作成
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 色設定（サーバーは赤、クライアントは他の色）
        colors = (
            ["skyblue"] * (len(clients) - 1) + ["red"]
            if server_json
            else ["skyblue"] * len(clients)
        )

        # 最終精度
        bars = axes[0, 0].bar(
            clients, metrics["final_accuracy"], color=colors, alpha=0.8
        )
        axes[0, 0].set_title("最終精度の比較", fontsize=14)
        axes[0, 0].set_ylabel("精度 (%)", fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 精度向上
        colors2 = (
            ["lightgreen"] * (len(clients) - 1) + ["red"]
            if server_json
            else ["lightgreen"] * len(clients)
        )
        axes[0, 1].bar(
            clients, metrics["accuracy_improvement"], color=colors2, alpha=0.8
        )
        axes[0, 1].set_title("精度向上の比較", fontsize=14)
        axes[0, 1].set_ylabel("精度向上 (%)", fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 最高精度
        colors3 = (
            ["orange"] * (len(clients) - 1) + ["red"]
            if server_json
            else ["orange"] * len(clients)
        )
        axes[1, 0].bar(clients, metrics["best_accuracy"], color=colors3, alpha=0.8)
        axes[1, 0].set_title("最高精度の比較", fontsize=14)
        axes[1, 0].set_ylabel("精度 (%)", fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis="x", rotation=45)

        # 最終損失
        colors4 = (
            ["salmon"] * (len(clients) - 1) + ["red"]
            if server_json
            else ["salmon"] * len(clients)
        )
        axes[1, 1].bar(clients, metrics["final_loss"], color=colors4, alpha=0.8)
        axes[1, 1].set_title("最終損失の比較", fontsize=14)
        axes[1, 1].set_ylabel("損失", fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, "client_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"クライアント比較チャートを保存しました: {plot_path}")

    def plot_data_distribution(self, json_results):
        """データ分布の可視化"""
        if not json_results:
            return

        fig, axes = plt.subplots(
            1, len(json_results), figsize=(5 * len(json_results), 6)
        )
        if len(json_results) == 1:
            axes = [axes]

        for i, (client_id, data) in enumerate(json_results.items()):
            dist = data["data_distribution_details"]
            labels = list(map(str, dist.keys()))
            values = list(dist.values())

            axes[i].pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            axes[i].set_title(f"クライアント {client_id}\nデータ分布", fontsize=12)

        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, "data_distribution.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"データ分布チャートを保存しました: {plot_path}")

    def generate_summary_report(self, json_results, df, server_json=None):
        """サマリーレポートを生成（サーバー結果を含む）"""
        report_path = os.path.join(self.results_dir, "summary_report.txt")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== 連合学習実験サマリーレポート ===\\n\\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")

            if json_results:
                # 実験情報
                first_client = list(json_results.values())[0]
                exp_info = first_client["experiment_info"]
                f.write("== 実験設定 ==\\n")
                f.write(f"実験ID: {exp_info['experiment_id']}\\n")
                f.write(f"デバイス: {exp_info['device']}\\n")
                f.write(f"データ分散: {exp_info['data_distribution']}\\n")
                f.write(f"ローカルエポック数: {exp_info['local_epochs']}\\n")
                f.write(f"バッチサイズ: {exp_info['batch_size']}\\n")
                f.write(f"総ラウンド数: {exp_info['total_rounds']}\\n\\n")

                # サーバー側結果
                if server_json:
                    f.write("== サーバー側評価結果 ==\\n")
                    server_exp = server_json["experiment_info"]
                    server_summary = server_json["summary"]
                    f.write(f"集約タイプ: {server_exp['aggregation_type']}\\n")
                    f.write(
                        f"最小集約クライアント数: {server_exp['min_clients_for_aggregation']}\\n"
                    )
                    f.write(f"テストサンプル数: {server_exp['test_samples']}\\n")
                    f.write(f"初期精度: {server_summary['initial_accuracy']:.2f}%\\n")
                    f.write(f"最終精度: {server_summary['final_accuracy']:.2f}%\\n")
                    f.write(
                        f"精度向上: {server_summary['accuracy_improvement']:.2f}%\\n"
                    )
                    f.write(f"最高精度: {server_summary['best_accuracy']:.2f}%\\n")
                    f.write(f"最低損失: {server_summary['best_loss']:.4f}\\n\\n")

                # クライアント別結果
                f.write("== クライアント別結果 ==\\n")
                for client_id, data in json_results.items():
                    summary = data["summary"]
                    f.write(f"\\nクライアント {client_id}:\\n")
                    f.write(
                        f"  総サンプル数: {data['experiment_info']['total_samples']}\\n"
                    )
                    f.write(f"  初期精度: {summary['initial_accuracy']:.2f}%\\n")
                    f.write(f"  最終精度: {summary['final_accuracy']:.2f}%\\n")
                    f.write(f"  精度向上: {summary['accuracy_improvement']:.2f}%\\n")
                    f.write(f"  最高精度: {summary['best_accuracy']:.2f}%\\n")
                    f.write(f"  最終損失: {summary['final_loss']:.4f}\\n")

                # 全体統計
                all_final_acc = [
                    data["summary"]["final_accuracy"] for data in json_results.values()
                ]
                all_improvements = [
                    data["summary"]["accuracy_improvement"]
                    for data in json_results.values()
                ]

                f.write("\\n== 全体統計（クライアント） ==\\n")
                f.write(f"平均最終精度: {np.mean(all_final_acc):.2f}%\\n")
                f.write(f"最高最終精度: {np.max(all_final_acc):.2f}%\\n")
                f.write(f"最低最終精度: {np.min(all_final_acc):.2f}%\\n")
                f.write(f"平均精度向上: {np.mean(all_improvements):.2f}%\\n")

                # サーバーとクライアントの比較
                if server_json:
                    server_final = server_json["summary"]["final_accuracy"]
                    client_avg = np.mean(all_final_acc)
                    f.write("\\n== サーバー vs クライアント ==\\n")
                    f.write(f"サーバー最終精度: {server_final:.2f}%\\n")
                    f.write(f"クライアント平均精度: {client_avg:.2f}%\\n")
                    f.write(f"精度差: {server_final - client_avg:.2f}%\\n")

        print(f"サマリーレポートを保存しました: {report_path}")

    def visualize_all(self):
        """全ての可視化を実行（サーバー評価を含む）"""
        print("連合学習結果の可視化を開始します...")

        # データ読み込み
        df = self.load_csv_results()
        json_results = self.load_json_results()
        server_df, server_json = self.load_server_results()

        if df is None and not json_results and server_df is None:
            print("結果ファイルが見つかりません")
            return

        # サーバーデータの存在を報告
        if server_df is not None:
            print(f"サーバー側評価データが見つかりました: {len(server_df)}ラウンド")

        # グラフ生成
        if df is not None:
            self.plot_accuracy_trends(df, server_df)
            self.plot_loss_trends(df, server_df)
        elif server_df is not None:
            # クライアントデータがない場合はサーバーデータのみ
            self.plot_accuracy_trends(pd.DataFrame(), server_df)
            self.plot_loss_trends(pd.DataFrame(), server_df)

        if json_results:
            self.plot_comparison_chart(json_results, server_json)
            self.plot_data_distribution(json_results)
            self.generate_summary_report(json_results, df, server_json)
        elif server_json:
            # クライアント結果がない場合はサーバー結果のみでレポート生成
            self.generate_summary_report({}, df, server_json)

        print(f"\\n可視化完了! 結果は {self.plot_dir} に保存されました。")


def main():
    parser = argparse.ArgumentParser(description="連合学習結果の可視化")
    parser.add_argument(
        "--results_dir",
        default="./results",
        help="結果ファイルが保存されているディレクトリ",
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"結果ディレクトリが見つかりません: {args.results_dir}")
        return

    visualizer = FederatedLearningVisualizer(args.results_dir)
    visualizer.visualize_all()


if __name__ == "__main__":
    main()
