#!/usr/bin/env python3
"""
連合学習システム管理ツール
"""

import json
import subprocess
import time
import requests
import argparse
import os
import sys


class FederatedLearningManager:
    def __init__(self, config_file="fl_config.json"):
        with open(config_file, "r") as f:
            self.config = json.load(f)

        self.system_config = self.config["system_config"]
        self.server_url = f"http://localhost:{self.system_config['server_port']}"

    def start_system(self):
        """改善されたシステムを起動"""
        print("=== 連合学習システム起動中 ===")

        # 既存システムを停止
        self.stop_system()

        # Dockerイメージをビルド
        print("Dockerイメージをビルド中...")
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "build"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"ビルドエラー: {result.stderr}")
            return False

        # システムを起動
        print("システムを起動中...")
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "up", "-d"],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"起動エラー: {result.stderr}")
            return False

        # サーバーの起動を待機
        print("サーバー起動を待機中...")
        if not self.wait_for_server():
            print("サーバー起動失敗")
            return False

        print("✅ システム起動完了")
        return True

    def stop_system(self):
        """システムを停止"""
        print("既存システムを停止中...")
        subprocess.run(["docker-compose", "down"], capture_output=True)
        subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "down"],
            capture_output=True,
        )

    def wait_for_server(self, timeout=60):
        """サーバーの起動を待機"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.server_url}/", timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(2)
        return False

    def get_system_status(self):
        """システム状況を取得"""
        try:
            # サーバー状況
            response = requests.get(f"{self.server_url}/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                print(f"📊 システム状況:")
                print(f"  現在のラウンド: {status.get('current_round', 0)}")
                print(
                    f"  アップロード済み: {status.get('uploaded_clients', 0)}/{status.get('expected_clients', 3)}"
                )
                print(
                    f"  ラウンド完了: {'✅' if status.get('is_complete', False) else '❌'}"
                )
                print(
                    f"  タイムアウト: {'⚠️' if status.get('is_timeout', False) else '✅'}"
                )
                return status
            else:
                print("❌ サーバーからのステータス取得失敗")
                return None
        except Exception as e:
            print(f"❌ サーバー接続エラー: {str(e)}")
            return None

    def monitor_training(self, duration=300):
        """学習進捗を監視"""
        print(f"📈 学習進捗を{duration}秒間監視中...")
        start_time = time.time()
        last_round = -1

        while time.time() - start_time < duration:
            status = self.get_system_status()
            if status:
                current_round = status.get("current_round", 0)
                if current_round > last_round:
                    print(f"🔄 ラウンド {current_round} 開始")
                    last_round = current_round

            time.sleep(10)

    def show_logs(self, service="", follow=False):
        """ログを表示"""
        cmd = ["docker-compose", "-f", "docker-compose.yml", "logs"]

        if follow:
            cmd.append("-f")

        if service:
            cmd.append(service)

        subprocess.run(cmd)

    def health_check(self):
        """システムヘルスチェック"""
        print("🏥 システムヘルスチェック実行中...")

        # コンテナ状況確認
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "ps"],
            capture_output=True,
            text=True,
        )

        print("📦 コンテナ状況:")
        print(result.stdout)

        # サーバー状況確認
        status = self.get_system_status()
        if status:
            print("✅ サーバー正常")
        else:
            print("❌ サーバー異常")

        return status is not None

    def restart_client(self, client_id):
        """特定のクライアントを再起動"""
        print(f"🔄 クライアント {client_id} を再起動中...")
        subprocess.run(
            [
                "docker-compose",
                "-f",
                "docker-compose.yml",
                "restart",
                client_id,
            ]
        )

    def cleanup(self):
        """システムのクリーンアップ"""
        print("🧹 システムクリーンアップ中...")
        subprocess.run(["docker-compose", "-f", "docker-compose.yml", "down", "-v"])
        subprocess.run(["docker", "system", "prune", "-f"])
        print("✅ クリーンアップ完了")


def main():
    parser = argparse.ArgumentParser(description="連合学習システム管理ツール")
    parser.add_argument(
        "command",
        choices=[
            "start",
            "stop",
            "status",
            "monitor",
            "logs",
            "health",
            "restart",
            "cleanup",
        ],
        help="実行するコマンド",
    )
    parser.add_argument("--service", help="特定のサービス（ログや再起動用）")
    parser.add_argument("--follow", "-f", action="store_true", help="ログをフォロー")
    parser.add_argument("--duration", type=int, default=300, help="監視時間（秒）")
    parser.add_argument("--config", default="fl_config.json", help="設定ファイル")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"❌ 設定ファイル {args.config} が見つかりません")
        sys.exit(1)

    manager = FederatedLearningManager(args.config)

    if args.command == "start":
        if manager.start_system():
            print("🚀 システム起動成功！")
            print("監視を開始するには: python system_manager.py monitor")
        else:
            print("❌ システム起動失敗")
            sys.exit(1)

    elif args.command == "stop":
        manager.stop_system()
        print("🛑 システム停止完了")

    elif args.command == "status":
        manager.get_system_status()

    elif args.command == "monitor":
        manager.monitor_training(args.duration)

    elif args.command == "logs":
        manager.show_logs(args.service, args.follow)

    elif args.command == "health":
        if manager.health_check():
            print("✅ システム正常")
        else:
            print("❌ システム異常")
            sys.exit(1)

    elif args.command == "restart":
        if args.service:
            manager.restart_client(args.service)
        else:
            print("❌ --service オプションでクライアントIDを指定してください")

    elif args.command == "cleanup":
        manager.cleanup()


if __name__ == "__main__":
    main()
