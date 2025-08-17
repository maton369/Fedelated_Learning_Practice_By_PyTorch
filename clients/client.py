import argparse
import json
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import sys
import os
import numpy as np
from collections import defaultdict
import csv
from datetime import datetime
import random

# shared モジュールをパスに追加
sys.path.append("/app/shared")
from model_utils import get_model, save_model_to_bytes, load_state_dict_from_bytes


class FederatedClient:
    def __init__(self, config):
        self.client_id = config["client_id"]
        self.server_url = config["server_url"]
        self.local_epochs = config.get("local_epochs", 1)
        self.batch_size = config.get("batch_size", 64)
        self.device = torch.device(config.get("device", "cpu"))
        self.max_rounds = config.get("max_rounds", 10)
        self.data_distribution = config.get(
            "data_distribution", "iid"
        )  # "iid" or "non_iid"
        self.non_iid_alpha = config.get("non_iid_alpha", 0.5)  # Non-IIDの程度

        # データ設定
        self.data_config = config.get("data_config", {})
        self.data_dir = self.data_config.get("data_dir", "/app/data")
        self.force_download = self.data_config.get("force_download", False)
        self.reuse_existing = self.data_config.get("reuse_existing", True)

        # モデル初期化
        self.model = get_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.02)

        # 統計情報
        self.training_history = []

        # 結果出力設定
        self.results_dir = "/app/results"
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 同期設定
        self.sync_wait_for_server = config.get("sync_config", {}).get(
            "wait_for_server_ready", True
        )
        self.status_check_interval = config.get("sync_config", {}).get(
            "status_check_interval", 5
        )

        # データセット準備
        self.setup_data()

        print(f"クライアント {self.client_id} が初期化されました")
        print(f"サーバーURL: {self.server_url}")
        print(f"データ数: {len(self.train_dataset)}")
        print(f"データ分散: {self.data_distribution}")

    def setup_data(self):
        """データセットの準備（IIDまたはNon-IID）"""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # 既存データのチェック
        mnist_path = os.path.join(self.data_dir, "MNIST")
        data_exists = os.path.exists(mnist_path) and len(os.listdir(mnist_path)) > 0

        # ダウンロード設定
        should_download = (
            self.force_download or (not data_exists) or (not self.reuse_existing)
        )

        if data_exists and self.reuse_existing and not self.force_download:
            print(
                f"クライアント {self.client_id} - 既存のMNISTデータを再利用します: {mnist_path}"
            )
        elif should_download:
            print(f"クライアント {self.client_id} - MNISTデータをダウンロード中...")

        # MNISTデータセットをダウンロード
        full_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=should_download,
            transform=transform,
        )

        client_num = int(self.client_id.replace("client", ""))

        if self.data_distribution == "non_iid":
            indices = self._create_non_iid_split(full_dataset, client_num)
        else:
            indices = self._create_iid_split(full_dataset, client_num)

        self.train_dataset = Subset(full_dataset, indices)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # データ分布を表示
        self._print_data_distribution()

    def _create_iid_split(self, dataset, client_num):
        """IID（独立同分布）データ分割"""
        total_size = len(dataset)
        chunk_size = total_size // 3
        start_idx = (client_num - 1) * chunk_size

        if client_num == 3:
            end_idx = total_size
        else:
            end_idx = client_num * chunk_size

        return list(range(start_idx, end_idx))

    def _create_non_iid_split(self, dataset, client_num):
        """Non-IID（非独立同分布）データ分割"""
        # ラベルごとにデータをグループ化
        label_groups = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            label_groups[label].append(idx)

        # 各クライアントに偏ったラベル分布を作成
        client_indices = []

        # ディリクレ分布を使用してラベル分布を生成
        np.random.seed(42 + client_num)

        for label in range(10):
            label_indices = label_groups[label]
            n_samples = len(label_indices)

            # クライアントごとの分布を生成
            proportions = np.random.dirichlet([self.non_iid_alpha] * 3)
            client_samples = int(proportions[client_num - 1] * n_samples)

            # ランダムサンプリング
            selected_indices = np.random.choice(
                label_indices,
                size=min(client_samples, len(label_indices)),
                replace=False,
            )
            client_indices.extend(selected_indices)

        return client_indices

    def _print_data_distribution(self):
        """データ分布を表示"""
        label_counts = defaultdict(int)
        for idx in self.train_dataset.indices:
            _, label = self.train_dataset.dataset[idx]
            label_counts[label] += 1

        print(f"クライアント {self.client_id} のデータ分布:")
        for label in sorted(label_counts.keys()):
            print(f"  ラベル {label}: {label_counts[label]} サンプル")

    def train_local(self):
        """ローカルデータで学習"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for epoch in range(self.local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.train_loader)

        # 統計情報を記録
        self.training_history.append(
            {
                "loss": avg_loss,
                "accuracy": accuracy,
                "samples": total,
                "timestamp": datetime.now().isoformat(),
            }
        )

        print(f"クライアント {self.client_id} - 学習完了")
        print(f"  損失: {avg_loss:.4f}, 精度: {accuracy:.2f}%")

        return avg_loss, accuracy

    def upload_model(self, max_retries=3):
        """学習済みモデルをサーバーにアップロード（リトライ機能付き）"""
        for attempt in range(max_retries):
            try:
                model_bytes = save_model_to_bytes(self.model)
                files = {
                    "model": ("model.pth", model_bytes, "application/octet-stream")
                }

                response = requests.post(
                    f"{self.server_url}/upload/{self.client_id}",
                    files=files,
                    timeout=300,  # 5分に延長
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"クライアント {self.client_id} - モデルアップロード成功")
                    print(
                        f"  進捗: {result.get('uploaded_count', 0)}/{result.get('expected_count', 3)}"
                    )
                    return True, result
                else:
                    print(
                        f"クライアント {self.client_id} - アップロード失敗 (試行 {attempt + 1}): {response.text}"
                    )

            except Exception as e:
                print(
                    f"クライアント {self.client_id} - アップロードエラー (試行 {attempt + 1}): {str(e)}"
                )

            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # 指数バックオフ

        return False, None

    def check_server_ready(self):
        """サーバーが集約準備完了まで待機（同期）"""
        print(f"クライアント {self.client_id} - サーバー準備完了を待機中...")
        start_time = time.time()
        timeout = 120  # 2分タイムアウト

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/status", timeout=10)
                if response.status_code == 200:
                    status = response.json()
                    # サーバーが新しいラウンドの準備ができているかチェック
                    if status.get("current_round", 0) >= 0:
                        print(f"クライアント {self.client_id} - サーバー準備完了を確認")
                        return True
                time.sleep(self.status_check_interval)
            except Exception as e:
                print(
                    f"クライアント {self.client_id} - ステータスチェックエラー: {str(e)}"
                )
                time.sleep(self.status_check_interval)

        print(f"クライアント {self.client_id} - サーバー準備タイムアウト")
        return False

    def wait_for_all_clients_uploaded(self):
        """全クライアントのアップロード完了まで待機（同期）"""
        print(
            f"クライアント {self.client_id} - 全クライアントアップロード完了を待機中..."
        )
        start_time = time.time()
        timeout = 300  # 5分タイムアウトに延長

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/status", timeout=10)
                if response.status_code == 200:
                    status = response.json()
                    # 全クライアントがアップロード完了したかチェック
                    if status.get("can_aggregate", False):
                        print(
                            f"クライアント {self.client_id} - 全クライアントアップロード完了を確認"
                        )
                        return True

                    # 一部のクライアントが終了した場合の処理
                    uploaded_count = status.get("uploaded_clients", 0)
                    expected_count = status.get("expected_clients", 3)
                    if uploaded_count > 0 and uploaded_count < expected_count:
                        print(
                            f"クライアント {self.client_id} - 現在 {uploaded_count}/{expected_count} クライアントがアップロード完了"
                        )
                        print(
                            f"残り {expected_count - uploaded_count} クライアントの完了を待機中..."
                        )

                time.sleep(self.status_check_interval)
            except Exception as e:
                print(
                    f"クライアント {self.client_id} - ステータスチェックエラー: {str(e)}"
                )
                time.sleep(self.status_check_interval)

        print(f"クライアント {self.client_id} - 全クライアントアップロードタイムアウト")
        return False

    def download_model(self, max_retries=3):
        """サーバーから集約済みモデルをダウンロード（リトライ機能付き）"""
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"{self.server_url}/download", timeout=120
                )  # 2分に延長

                if response.status_code == 200:
                    import io

                    buffer = io.BytesIO(response.content)
                    state_dict = torch.load(buffer, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    print(f"クライアント {self.client_id} - モデルダウンロード成功")
                    return True
                else:
                    print(
                        f"クライアント {self.client_id} - ダウンロード失敗 (試行 {attempt + 1}): {response.text}"
                    )

            except Exception as e:
                print(
                    f"クライアント {self.client_id} - ダウンロードエラー (試行 {attempt + 1}): {str(e)}"
                )

            if attempt < max_retries - 1:
                time.sleep(2**attempt)

        return False

    def wait_for_server(self, max_retries=10):
        """サーバーが起動するまで待機"""
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.server_url}/", timeout=5)
                if response.status_code == 200:
                    print(f"クライアント {self.client_id} - サーバー接続確認")
                    return True
            except:
                pass

            print(
                f"クライアント {self.client_id} - サーバー待機中... ({i+1}/{max_retries})"
            )
            time.sleep(5)

        print(f"クライアント {self.client_id} - サーバー接続失敗")
        return False

    def print_summary(self):
        """学習結果のサマリーを出力"""
        if not self.training_history:
            return

        print(f"\n=== クライアント {self.client_id} 学習サマリー ===")
        print(f"ラウンド数: {len(self.training_history)}")
        print(f"最終精度: {self.training_history[-1]['accuracy']:.2f}%")
        print(f"最終損失: {self.training_history[-1]['loss']:.4f}")

        # 精度の推移
        accuracies = [h["accuracy"] for h in self.training_history]
        print(f"精度向上: {accuracies[0]:.2f}% → {accuracies[-1]:.2f}%")

    def save_results_to_csv(self):
        """学習結果をCSV形式で保存"""
        if not self.training_history:
            return

        os.makedirs(self.results_dir, exist_ok=True)

        # CSVファイルパス
        csv_file = os.path.join(
            self.results_dir,
            f"training_history_{self.client_id}_{self.experiment_id}.csv",
        )

        # CSVヘッダー
        fieldnames = ["round", "loss", "accuracy", "samples", "timestamp"]

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, history in enumerate(self.training_history):
                writer.writerow(
                    {
                        "round": i + 1,
                        "loss": history["loss"],
                        "accuracy": history["accuracy"],
                        "samples": history["samples"],
                        "timestamp": history.get("timestamp", ""),
                    }
                )

        print(f"学習履歴を保存しました: {csv_file}")
        return csv_file

    def save_results_to_json(self):
        """詳細な結果をJSON形式で保存"""
        if not self.training_history:
            return

        os.makedirs(self.results_dir, exist_ok=True)

        # 詳細結果データ
        results = {
            "experiment_info": {
                "client_id": self.client_id,
                "experiment_id": self.experiment_id,
                "timestamp": datetime.now().isoformat(),
                "device": str(self.device),
                "data_distribution": self.data_distribution,
                "local_epochs": self.local_epochs,
                "batch_size": self.batch_size,
                "total_samples": len(self.train_dataset),
                "total_rounds": len(self.training_history),
            },
            "training_history": self.training_history,
            "summary": {
                "initial_accuracy": (
                    self.training_history[0]["accuracy"] if self.training_history else 0
                ),
                "final_accuracy": (
                    self.training_history[-1]["accuracy"]
                    if self.training_history
                    else 0
                ),
                "initial_loss": (
                    self.training_history[0]["loss"] if self.training_history else 0
                ),
                "final_loss": (
                    self.training_history[-1]["loss"] if self.training_history else 0
                ),
                "accuracy_improvement": (
                    self.training_history[-1]["accuracy"]
                    - self.training_history[0]["accuracy"]
                    if len(self.training_history) > 1
                    else 0
                ),
                "best_accuracy": (
                    max([h["accuracy"] for h in self.training_history])
                    if self.training_history
                    else 0
                ),
                "best_loss": (
                    min([h["loss"] for h in self.training_history])
                    if self.training_history
                    else 0
                ),
            },
        }

        # データ分布情報を追加
        label_counts = defaultdict(int)
        for idx in self.train_dataset.indices:
            _, label = self.train_dataset.dataset[idx]
            label_counts[label] += 1

        results["data_distribution_details"] = dict(label_counts)

        # JSONファイルパス
        json_file = os.path.join(
            self.results_dir,
            f"detailed_results_{self.client_id}_{self.experiment_id}.json",
        )

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"詳細結果を保存しました: {json_file}")
        return json_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="クライアント設定ファイルのパス"
    )
    args = parser.parse_args()

    # 設定ファイル読み込み
    with open(args.config, "r") as f:
        config = json.load(f)

    # クライアント初期化
    client = FederatedClient(config)

    # サーバー接続待機
    if not client.wait_for_server():
        return

    print(f"\n=== クライアント {client.client_id} 連合学習開始 ===")

    # 連合学習ループ
    successful_rounds = 0
    for round_num in range(client.max_rounds):
        print(f"\n--- ラウンド {round_num + 1} ---")

        try:
            # サーバーが集約準備完了まで待機
            print(f"サーバーの集約準備完了を待機中...")
            if not client.check_server_ready():
                print(f"ラウンド {round_num + 1} でサーバー準備タイムアウト")
                continue

            # ローカル学習
            print(f"ローカル学習開始...")
            client.train_local()

            # モデルをサーバーにアップロード
            print(f"モデルをサーバーにアップロード中...")
            success, result = client.upload_model()
            if not success:
                print(f"ラウンド {round_num + 1} でアップロード失敗、スキップします")
                continue

            # 全クライアントのアップロード完了まで待機
            print(f"全クライアントのアップロード完了を待機中...")
            if not client.wait_for_all_clients_uploaded():
                print(
                    f"ラウンド {round_num + 1} で全クライアントアップロードタイムアウト"
                )
                continue

            # 集約済みモデルをダウンロード
            print(f"集約済みモデルをダウンロード中...")
            if client.download_model():
                successful_rounds += 1
                print(f"ラウンド {round_num + 1} 完了")
            else:
                print(f"ラウンド {round_num + 1} でダウンロード失敗")

        except Exception as e:
            print(f"ラウンド {round_num + 1} でエラー発生: {str(e)}")
            continue

    print(f"\nクライアント {client.client_id} 連合学習完了")
    print(f"成功ラウンド: {successful_rounds}/{client.max_rounds}")

    # サーバーに学習完了を通知
    print(f"サーバーに学習完了を通知中...")
    try:
        response = requests.post(
            f"{client.server_url}/learning_completed/{client.client_id}", timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            print(f"学習完了通知成功: {result['message']}")
            print(
                f"完了クライアント数: {result['completed_clients']}/{result['total_clients']}"
            )
            if result.get("all_completed"):
                print("全クライアントの学習が完了しました！")
        else:
            print(f"学習完了通知失敗: {response.status_code}")
    except Exception as e:
        print(f"学習完了通知エラー: {str(e)}")

    # サマリー出力
    client.print_summary()

    # 結果をファイルに保存
    if client.training_history:
        print(f"\n=== 結果ファイル出力 ===")
        client.save_results_to_csv()
        client.save_results_to_json()
        print("結果の出力が完了しました。")


if __name__ == "__main__":
    main()
