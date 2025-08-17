from flask import Flask, request, send_file, jsonify
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json
import time
import csv
from datetime import datetime
from threading import Lock, Thread
import concurrent.futures
from queue import Queue
import threading
from shared.model_utils import (
    get_model,
    load_state_dict_from_bytes,
    save_model_to_bytes,
)

app = Flask(__name__)

UPLOAD_DIR = "uploaded_models"
AGG_MODEL_PATH = "agg_model.pth"
STATUS_FILE = "round_status.json"

# モデル保存用ディレクトリの作成
os.makedirs(UPLOAD_DIR, exist_ok=True)


# グローバル状態管理
class FederatedLearningState:
    def __init__(self):
        self.lock = Lock()
        self.current_round = 0
        self.expected_clients = 3
        self.wait_for_all_clients = True  # 同期: 全クライアント完了まで待機
        self.uploaded_clients = set()
        self.round_start_time = None
        self.aggregation_interval = 30  # 30秒間隔で集約チェック
        self.auto_aggregation_enabled = True  # 自動集約機能
        self.server_evaluation_history = []  # サーバー側評価履歴
        self.test_dataset = None  # テストデータセット
        self.test_loader = None  # テストデータローダー

        # 並列処理用（最適化）
        self.upload_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),  # 動的なワーカー数設定
            thread_name_prefix="upload_worker",
        )  # アップロード処理用
        self.upload_queue = Queue()  # アップロードタスクキュー

        # 学習完了管理
        self.clients_completed_learning = set()  # 学習完了したクライアントのセット
        self.all_clients_completed = False  # 全クライアント学習完了フラグ

    def reset_round(self):
        # ロックを取得せずに状態を更新（呼び出し元でロックが取得されている場合があるため）
        self.uploaded_clients.clear()
        self.round_start_time = time.time()

    def add_client(self, client_id):
        print(f"DEBUG: Attempting to acquire lock for {client_id}...")
        with self.lock:
            print(f"DEBUG: Lock acquired for {client_id}")
            self.uploaded_clients.add(client_id)
            result = len(self.uploaded_clients)
            print(f"DEBUG: Lock will be released for {client_id}, result: {result}")
            return result

    def is_round_complete(self):
        with self.lock:
            return len(self.uploaded_clients) >= self.expected_clients

    def can_start_aggregation(self):
        """同期: 全クライアント完了まで待機"""
        print(f"DEBUG: Attempting to acquire lock for aggregation check...")
        with self.lock:
            print(f"DEBUG: Lock acquired for aggregation check")
            result = len(self.uploaded_clients) >= self.expected_clients
            print(
                f"DEBUG: Lock will be released for aggregation check, result: {result}"
            )
            return result

    def is_round_timeout(self):
        with self.lock:
            if self.round_start_time is None:
                return False
            return time.time() - self.round_start_time > self.round_timeout

    def get_status(self):
        print(f"DEBUG: get_status called, attempting to acquire lock...")
        with self.lock:
            print(f"DEBUG: get_status lock acquired, preparing response...")
            try:
                # ロックを保持したまま直接計算（メソッド呼び出しを避ける）
                uploaded_count = len(self.uploaded_clients)
                is_complete = uploaded_count >= self.expected_clients

                response = {
                    "current_round": self.current_round,
                    "uploaded_clients": uploaded_count,
                    "expected_clients": self.expected_clients,
                    "is_complete": is_complete,
                    "can_aggregate": is_complete,
                    "waiting_for_clients": self.expected_clients - uploaded_count,
                    "learning_completed_clients": len(self.clients_completed_learning),
                    "all_clients_completed": self.all_clients_completed,
                }
                print(f"DEBUG: get_status response prepared: {response}")
                return response
            except Exception as e:
                print(f"ERROR: get_status error: {str(e)}")
                raise


fl_state = FederatedLearningState()


def save_model_file_async(client_id, file_stream):
    """非同期でモデルファイルを保存"""
    try:
        save_path = os.path.join(UPLOAD_DIR, f"{client_id}.pth")

        # ファイルを直接保存（メモリ効率を改善）
        with open(save_path, "wb") as f:
            while True:
                chunk = file_stream.read(8192)  # 8KBずつ読み込み
                if not chunk:
                    break
                f.write(chunk)

        print(f"Model file saved for {client_id}: {save_path}")
        return True, save_path

    except Exception as e:
        print(f"Error saving model file for {client_id}: {str(e)}")
        return False, None


def setup_test_dataset():
    """サーバー側でテストデータセットを準備"""
    try:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # MNISTテストデータセットを準備
        test_dataset = torchvision.datasets.MNIST(
            root="/app/data", train=False, download=True, transform=transform
        )

        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        fl_state.test_dataset = test_dataset
        fl_state.test_loader = test_loader

        print(f"テストデータセット準備完了: {len(test_dataset)}サンプル")
        return True

    except Exception as e:
        print(f"テストデータセット準備エラー: {str(e)}")
        return False


def evaluate_model(model, device="cpu"):
    """集約済みモデルをテストデータセットで評価"""
    if fl_state.test_loader is None:
        print("テストデータセットが準備されていません")
        return None, None

    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in fl_state.test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(fl_state.test_loader)

    return accuracy, avg_loss


def save_server_evaluation(round_num, accuracy, loss):
    """サーバー側評価結果を記録"""
    evaluation_entry = {
        "round": round_num,
        "accuracy": accuracy,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
        "test_samples": len(fl_state.test_dataset) if fl_state.test_dataset else 0,
    }

    fl_state.server_evaluation_history.append(evaluation_entry)
    print(f"サーバー評価 - ラウンド {round_num}: 精度 {accuracy:.2f}%, 損失 {loss:.4f}")


def save_server_results_to_files():
    """サーバー側評価結果をファイルに保存"""
    if not fl_state.server_evaluation_history:
        return

    os.makedirs("/app/results", exist_ok=True)
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV形式で保存
    csv_file = f"/app/results/server_evaluation_{experiment_id}.csv"
    fieldnames = ["round", "accuracy", "loss", "timestamp", "test_samples"]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in fl_state.server_evaluation_history:
            writer.writerow(entry)

    # JSON形式で詳細結果を保存
    json_file = f"/app/results/server_detailed_results_{experiment_id}.json"
    server_results = {
        "experiment_info": {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "total_rounds": len(fl_state.server_evaluation_history),
            "test_samples": len(fl_state.test_dataset) if fl_state.test_dataset else 0,
            "aggregation_type": "async_federated_averaging",
            "wait_for_all_clients": fl_state.wait_for_all_clients,
        },
        "evaluation_history": fl_state.server_evaluation_history,
        "summary": {
            "initial_accuracy": (
                fl_state.server_evaluation_history[0]["accuracy"]
                if fl_state.server_evaluation_history
                else 0
            ),
            "final_accuracy": (
                fl_state.server_evaluation_history[-1]["accuracy"]
                if fl_state.server_evaluation_history
                else 0
            ),
            "best_accuracy": (
                max([h["accuracy"] for h in fl_state.server_evaluation_history])
                if fl_state.server_evaluation_history
                else 0
            ),
            "best_loss": (
                min([h["loss"] for h in fl_state.server_evaluation_history])
                if fl_state.server_evaluation_history
                else 0
            ),
            "accuracy_improvement": (
                fl_state.server_evaluation_history[-1]["accuracy"]
                - fl_state.server_evaluation_history[0]["accuracy"]
                if len(fl_state.server_evaluation_history) > 1
                else 0
            ),
        },
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(server_results, f, indent=2, ensure_ascii=False)

    print(f"サーバー評価結果を保存: {csv_file}, {json_file}")
    return csv_file, json_file


def auto_aggregation_worker():
    """バックグラウンドで同期集約を実行"""
    print("[Server] 同期集約ワーカーを開始しました")

    while fl_state.auto_aggregation_enabled:
        try:
            # 全クライアントがアップロード完了するまで待機
            while True:
                # ロックを最小限の時間だけ取得して状態をチェック
                with fl_state.lock:
                    current_count = len(fl_state.uploaded_clients)
                    all_clients_ready = current_count >= fl_state.expected_clients

                # 全クライアントが準備完了した場合
                if all_clients_ready:
                    print(
                        f"[Server] 全クライアント準備完了 (クライアント数: {current_count})"
                    )
                    break

                # 短い間隔でチェック（ロックを長時間保持しない）
                time.sleep(1.0)

            # 同期集約処理を実行
            if current_count > 0:
                print(
                    f"[Server] 同期集約処理を開始... (クライアント数: {current_count})"
                )
                perform_aggregation()
                print("[Server] 同期集約完了")

            # 次のラウンドまで待機
            time.sleep(fl_state.aggregation_interval)

        except Exception as e:
            print(f"[Server] 同期集約エラー: {str(e)}")
            time.sleep(5)


def perform_aggregation():
    """実際の集約処理を実行"""
    try:
        model_files = [
            os.path.join(UPLOAD_DIR, f)
            for f in os.listdir(UPLOAD_DIR)
            if f.endswith(".pth")
        ]

        if len(model_files) == 0:
            print("集約対象のモデルファイルが見つかりません")
            return False

        print(f"自動集約開始: {len(model_files)}個のモデル")

        # データサンプル数に基づく重み付き平均（簡単な実装では等重み）
        weights = [1.0 / len(model_files)] * len(model_files)

        # 初期モデルを取得
        agg_model = get_model()
        agg_state_dict = None

        # 重み付き平均集計
        for i, model_file in enumerate(model_files):
            state_dict = torch.load(model_file, map_location="cpu")
            if agg_state_dict is None:
                agg_state_dict = {
                    k: v.clone() * weights[i] for k, v in state_dict.items()
                }
            else:
                for k in agg_state_dict:
                    agg_state_dict[k] += state_dict[k] * weights[i]

        # モデルに読み込み
        agg_model.load_state_dict(agg_state_dict)

        # 保存
        torch.save(agg_model.state_dict(), AGG_MODEL_PATH)

        # ラウンド更新（ロックを取得して安全に更新）
        with fl_state.lock:
            fl_state.current_round += 1
            # サーバー側モデル評価
            if fl_state.test_loader is not None:
                accuracy, loss = evaluate_model(agg_model)
                if accuracy is not None and loss is not None:
                    save_server_evaluation(fl_state.current_round, accuracy, loss)
                    print(
                        f"サーバー評価 - ラウンド {fl_state.current_round}: 精度 {accuracy:.2f}%, 損失 {loss:.4f}"
                    )

            # ロックを保持したままreset_roundを呼び出し
            fl_state.reset_round()

        print(f"自動集約完了: {AGG_MODEL_PATH} - ラウンド {fl_state.current_round}")
        return True

    except Exception as e:
        print(f"集約処理エラー: {str(e)}")
        return False


@app.route("/upload/<client_id>", methods=["POST"])
def upload_model(client_id):
    """
    クライアントが学習済みモデル（バイナリ）をPOSTする（真の非同期並列化版）
    """
    try:
        print(f"Receiving model upload from {client_id}...")

        if "model" not in request.files:
            return jsonify({"error": "No model file uploaded"}), 400

        file = request.files["model"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

            # ファイル内容を事前に読み込み（安全な処理順序）
        print(f"DEBUG: Starting file read for {client_id}...")
        try:
            file_content = file.read()
            file_size = len(file_content)
            print(f"DEBUG: Model file buffered for {client_id}: {file_size} bytes")
        except Exception as e:
            print(f"ERROR: File read failed for {client_id}: {str(e)}")
            return jsonify({"error": f"Failed to read uploaded file: {str(e)}"}), 400

        # ファイル読み込み成功後、クライアント状態を更新
        print(f"DEBUG: Adding client {client_id} to state...")
        try:
            uploaded_count = fl_state.add_client(client_id)
            print(
                f"DEBUG: Client {client_id} added successfully, count: {uploaded_count}"
            )
        except Exception as e:
            print(f"ERROR: Failed to add client {client_id}: {str(e)}")
            return jsonify({"error": f"Failed to add client to state: {str(e)}"}), 500

            # レスポンス情報を準備
        print(f"DEBUG: Checking aggregation possibility for {client_id}...")
        try:
            # ロックを取得せずに状態をチェック
            with fl_state.lock:
                can_aggregate = (
                    len(fl_state.uploaded_clients) >= fl_state.expected_clients
                )
            print(
                f"DEBUG: Aggregation check completed for {client_id}: {can_aggregate}"
            )
        except Exception as e:
            print(f"ERROR: Failed to check aggregation for {client_id}: {str(e)}")
            return jsonify({"error": f"Failed to check aggregation: {str(e)}"}), 500

        response_start_time = time.time()

        # レスポンスを即座に返す
        response_data = {
            "message": f"Model from {client_id} received",
            "uploaded_count": uploaded_count,
            "expected_count": fl_state.expected_clients,
            "wait_for_all": fl_state.wait_for_all_clients,
            "round_complete": fl_state.is_round_complete(),
            "can_aggregate": can_aggregate,
            "processing_time": "async_parallel",
            "status": "processing",
            "response_timestamp": response_start_time,
        }

        print(
            f"Model from {client_id} registered. ({uploaded_count}/{fl_state.expected_clients})"
        )

        if can_aggregate:
            print(
                f"同期集約条件満たしました: {uploaded_count}クライアント (全{fl_state.expected_clients}クライアント)"
            )

        # ファイル保存を非同期で実行（レスポンス後）
        def async_file_save():
            try:
                save_path = os.path.join(UPLOAD_DIR, f"{client_id}.pth")

                # ファイル保存（効率的な書き込み）
                with open(save_path, "wb") as f:
                    f.write(file_content)

                print(
                    f"Background: Model file saved for {client_id}: {save_path} ({file_size} bytes)"
                )

            except Exception as e:
                print(f"Background save error for {client_id}: {str(e)}")
                # エラーが発生した場合、クライアントの登録を取り消し
                with fl_state.lock:
                    if client_id in fl_state.uploaded_clients:
                        fl_state.uploaded_clients.discard(client_id)
                        print(
                            f"Removed {client_id} from uploaded clients due to save error"
                        )

        # バックグラウンドでファイル保存を実行
        print(f"DEBUG: Submitting background save for {client_id}...")
        fl_state.upload_executor.submit(async_file_save)
        print(f"DEBUG: Background save submitted for {client_id}")

        print(f"DEBUG: Sending response to {client_id}...")
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error uploading model from {client_id}: {str(e)}")
        return jsonify({"error": f"Failed to upload model: {str(e)}"}), 500


@app.route("/download", methods=["GET"])
def download_model():
    """
    クライアントに集約済みモデルを返す
    """
    if not os.path.exists(AGG_MODEL_PATH):
        return "No aggregated model yet", 404

    return send_file(AGG_MODEL_PATH, as_attachment=True)


@app.route("/aggregate", methods=["POST"])
def aggregate_models():
    """
    すべてのアップロード済みクライアントモデルを重み付き平均して集約
    """
    try:
        # 非同期: 集約可能性チェック
        if not fl_state.can_start_aggregation():
            return (
                jsonify(
                    {
                        "error": "Aggregation conditions not met yet",
                        "status": fl_state.get_status(),
                        "message": f"Need all {fl_state.expected_clients} clients (current: {len(fl_state.uploaded_clients)})",
                    }
                ),
                400,
            )

        # 手動集約要求
        print("手動集約要求を受信")
        success = perform_aggregation()

        if not success:
            return jsonify({"error": "Aggregation failed"}), 500

        return (
            jsonify(
                {
                    "message": "Model aggregation completed",
                    "round": fl_state.current_round,
                    "aggregated_models": len(model_files),
                }
            ),
            200,
        )

    except Exception as e:
        print(f"Error during aggregation: {str(e)}")
        return jsonify({"error": f"Aggregation failed: {str(e)}"}), 500


@app.route("/status", methods=["GET"])
def get_status():
    """
    現在のラウンド状況を取得
    """
    print(f"DEBUG: /status endpoint called")
    try:
        status = fl_state.get_status()
        print(f"DEBUG: /status response: {status}")
        return jsonify(status), 200
    except Exception as e:
        print(f"ERROR: /status endpoint error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/ready/<client_id>", methods=["POST"])
def client_ready(client_id):
    """
    クライアントの準備完了を通知
    """
    # 新しいラウンドの開始を検知
    if len(fl_state.uploaded_clients) == 0:
        fl_state.reset_round()

    return (
        jsonify(
            {
                "message": f"Client {client_id} ready acknowledged",
                "current_round": fl_state.current_round,
            }
        ),
        200,
    )


@app.route("/learning_completed/<client_id>", methods=["POST"])
def client_learning_completed(client_id):
    """
    クライアントの学習完了を通知
    """
    try:
        with fl_state.lock:
            fl_state.clients_completed_learning.add(client_id)
            completed_count = len(fl_state.clients_completed_learning)
            total_clients = fl_state.expected_clients

            print(
                f"[Server] クライアント {client_id} の学習完了を記録 (完了: {completed_count}/{total_clients})"
            )

            # 全クライアントが学習完了したかチェック
            if completed_count >= total_clients:
                fl_state.all_clients_completed = True
                print(f"[Server] 全クライアントの学習が完了しました！")

                # 最終結果を保存
                try:
                    csv_file, json_file = save_server_results_to_files()
                    print(f"[Server] 最終結果を保存しました: {csv_file}, {json_file}")
                except Exception as e:
                    print(f"[Server] 最終結果保存エラー: {str(e)}")

                # サーバーを終了
                print("[Server] 全クライアント学習完了。サーバーを終了します。")
                os._exit(0)
            else:
                # 一部のクライアントが終了した場合の処理
                print(
                    f"[Server] 現在 {completed_count}/{total_clients} クライアントが学習完了"
                )
                print(
                    f"[Server] 残り {total_clients - completed_count} クライアントの完了を待機中..."
                )

            return (
                jsonify(
                    {
                        "message": f"Client {client_id} learning completion acknowledged",
                        "completed_clients": completed_count,
                        "total_clients": total_clients,
                        "all_completed": fl_state.all_clients_completed,
                    }
                ),
                200,
            )

    except Exception as e:
        print(f"[Server] 学習完了通知エラー: {str(e)}")
        return (
            jsonify({"error": f"Failed to record learning completion: {str(e)}"}),
            500,
        )


@app.route("/save_results", methods=["POST"])
def save_results():
    """サーバー側評価結果をファイルに保存"""
    try:
        csv_file, json_file = save_server_results_to_files()
        return (
            jsonify(
                {
                    "message": "Server evaluation results saved",
                    "csv_file": csv_file,
                    "json_file": json_file,
                    "total_rounds": len(fl_state.server_evaluation_history),
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to save results: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def index():
    # 最初のリクエスト時にテストデータセットを準備
    if fl_state.test_dataset is None:
        print("テストデータセットをセットアップ中...")
        if setup_test_dataset():
            print("テストデータセットの準備が完了しました")
        else:
            print("警告: テストデータセットの準備に失敗しました")

    return (
        jsonify(
            status="Federated Learning Server running",
            current_round=fl_state.current_round,
            expected_clients=fl_state.expected_clients,
        ),
        200,
    )


if __name__ == "__main__":
    print("Starting Federated Learning Server...")
    print(f"Expected clients: {fl_state.expected_clients}")
    print(f"Wait for all clients: {fl_state.wait_for_all_clients}")
    print(f"Auto-aggregation enabled: {fl_state.auto_aggregation_enabled}")
    print(f"Server will exit when all clients complete learning")
    print(f"Use /learning_completed/<client_id> endpoint to notify completion")

    # 自動集約スレッドを開始
    aggregation_thread = Thread(target=auto_aggregation_worker, daemon=True)
    aggregation_thread.start()
    print("同期集約スレッドを開始しました")

    # アップロードファイルサイズの制限を緩和
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

    try:
        # 並列処理能力を向上させるために設定を調整
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            processes=1,  # プロセス数（メモリ共有のため1に設定）
            use_reloader=False,  # リローダー無効化
        )
    except KeyboardInterrupt:
        print("\nサーバーを停止中...")
        fl_state.auto_aggregation_enabled = False

        # 評価結果をファイルに保存
        if fl_state.server_evaluation_history:
            try:
                save_server_results_to_files()
                print("サーバー側評価結果を保存しました")
            except Exception as e:
                print(f"結果保存エラー: {str(e)}")

        print("サーバーが正常に停止しました")
