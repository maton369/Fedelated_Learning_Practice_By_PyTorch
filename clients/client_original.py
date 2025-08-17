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

# shared モジュールをパスに追加
sys.path.append("/app/shared")
from model_utils import get_model, save_model_to_bytes, load_state_dict_from_bytes


class FederatedClient:
    def __init__(self, config):
        self.client_id = config["client_id"]
        self.server_url = config["server_url"]
        self.local_epochs = config["local_epochs"]
        self.batch_size = config["batch_size"]
        self.device = torch.device(config["device"])

        # モデル初期化
        self.model = get_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.02)

        # データセット準備
        self.setup_data()

        print(f"クライアント {self.client_id} が初期化されました")
        print(f"サーバーURL: {self.server_url}")
        print(f"データ数: {len(self.train_dataset)}")

    def setup_data(self):
        """MNISTデータセットを準備し、クライアントごとに分割"""
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # MNISTデータセットをダウンロード
        full_dataset = torchvision.datasets.MNIST(
            root="/app/data", train=True, download=True, transform=transform
        )

        # シンプルな分割: クライアントIDに基づいてデータを分割
        total_size = len(full_dataset)
        client_num = int(self.client_id.replace("client", ""))

        # 3つのクライアント用にデータを分割
        chunk_size = total_size // 3
        start_idx = (client_num - 1) * chunk_size

        if client_num == 3:  # 最後のクライアントは残り全て
            end_idx = total_size
        else:
            end_idx = client_num * chunk_size

        indices = list(range(start_idx, end_idx))
        self.train_dataset = Subset(full_dataset, indices)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

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

        print(f"クライアント {self.client_id} - 学習完了")
        print(f"  損失: {avg_loss:.4f}, 精度: {accuracy:.2f}%")

        return avg_loss, accuracy

    def upload_model(self):
        """学習済みモデルをサーバーにアップロード"""
        try:
            model_bytes = save_model_to_bytes(self.model)
            files = {"model": ("model.pth", model_bytes, "application/octet-stream")}

            response = requests.post(
                f"{self.server_url}/upload/{self.client_id}", files=files, timeout=30
            )

            if response.status_code == 200:
                print(f"クライアント {self.client_id} - モデルアップロード成功")
                return True
            else:
                print(
                    f"クライアント {self.client_id} - モデルアップロード失敗: {response.text}"
                )
                return False
        except Exception as e:
            print(f"クライアント {self.client_id} - アップロードエラー: {str(e)}")
            return False

    def download_model(self):
        """サーバーから集約済みモデルをダウンロード"""
        try:
            response = requests.get(f"{self.server_url}/download", timeout=30)

            if response.status_code == 200:
                # バイトデータからモデルを読み込み
                import io

                buffer = io.BytesIO(response.content)
                state_dict = torch.load(buffer, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"クライアント {self.client_id} - モデルダウンロード成功")
                return True
            else:
                print(
                    f"クライアント {self.client_id} - モデルダウンロード失敗: {response.text}"
                )
                return False
        except Exception as e:
            print(f"クライアント {self.client_id} - ダウンロードエラー: {str(e)}")
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
    for round_num in range(10):  # 10ラウンド実行
        print(f"\n--- ラウンド {round_num + 1} ---")

        # ローカル学習
        client.train_local()

        # モデルをサーバーにアップロード
        if not client.upload_model():
            continue

        # 他のクライアントの完了を待つ（簡単な待機）
        print(f"クライアント {client.client_id} - 他のクライアント完了待機中...")
        time.sleep(10)

        # サーバーに集約を要求（client1のみ）
        if client.client_id == "client1":
            try:
                response = requests.post(f"{client.server_url}/aggregate", timeout=60)
                if response.status_code == 200:
                    print("モデル集約完了")
                else:
                    print(f"集約失敗: {response.text}")
            except Exception as e:
                print(f"集約要求エラー: {str(e)}")

        # 集約待機
        time.sleep(5)

        # 集約済みモデルをダウンロード
        client.download_model()

    print(f"\nクライアント {client.client_id} 連合学習完了")


if __name__ == "__main__":
    main()
