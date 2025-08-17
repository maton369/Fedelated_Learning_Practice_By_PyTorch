# 連合学習システム実行方法

## 🚀 システム起動方法

### 1. 自動起動（推奨）
```bash
# プロジェクトディレクトリに移動
cd /Users/maton/Federated_Learning_Practice/By_PyTorch_Practice

# 起動スクリプトを実行
./start_federated_learning.sh
```

### 2. 手動起動
```bash
# プロジェクトディレクトリに移動
cd /Users/maton/Federated_Learning_Practice/By_PyTorch_Practice

# Dockerイメージをビルド
docker-compose build

# システムを起動
docker-compose up -d

# ログを確認
docker-compose logs -f
```

## 📊 ログ確認方法

### 全体のログを確認
```bash
docker-compose logs -f
```

### サーバーのログのみ確認
```bash
docker-compose logs -f server
```

### 特定のクライアントのログを確認
```bash
docker-compose logs -f client1
docker-compose logs -f client2
docker-compose logs -f client3
```

### 全クライアントのログを確認
```bash
docker-compose logs -f client1 client2 client3
```

## 🔧 システム管理

### コンテナ状態確認
```bash
docker-compose ps
```

### システム停止
```bash
docker-compose down
```

### システム再起動
```bash
docker-compose restart
```

### 完全クリーンアップ（データ・イメージも削除）
```bash
docker-compose down -v
docker-compose build --no-cache
```

## 📈 学習進捗の確認

システム起動後、以下のような流れで連合学習が実行されます：

1. **サーバー起動**: Flask APIサーバーがポート5000で起動
2. **クライアント接続**: 3つのクライアントがサーバーに接続
3. **データ準備**: 各クライアントがMNISTデータを分割取得
4. **学習ループ**: 10ラウンドの連合学習を実行
   - 各クライアントがローカル学習
   - モデルをサーバーにアップロード
   - サーバーでモデル集約（FedAvg）
   - 集約済みモデルを各クライアントにダウンロード

## 🎯 期待される結果

- **学習ラウンド**: 10ラウンド実行
- **学習時間**: 約10-15分程度
- **最終精度**: 85-90%程度（MNISTデータセット）

## ⚠️ トラブルシューティング

### ポート競合エラー
```bash
# ポート5000が使用中の場合
sudo lsof -i :5000
# 必要に応じてプロセスを終了
```

### メモリ不足エラー
```bash
# Dockerのメモリ制限を確認
docker system df
docker system prune  # 不要なデータを削除
```

### ネットワークエラー
```bash
# Dockerネットワークをリセット
docker-compose down
docker network prune
docker-compose up -d
```

## 📝 設定変更

クライアント設定は以下のファイルで変更可能：
- `clients/client_config/client1.json`
- `clients/client_config/client2.json`
- `clients/client_config/client3.json`

変更可能な設定項目：
- `local_epochs`: ローカル学習エポック数
- `batch_size`: バッチサイズ
- `device`: 使用デバイス（cpu/cuda）