#!/bin/bash

echo "=== 連合学習システム起動スクリプト ==="

# Docker ComposeでDockerイメージをビルド
echo "1. Dockerイメージをビルド中..."
docker-compose build

# 既存のコンテナを停止・削除
echo "2. 既存のコンテナを停止・削除中..."
docker-compose down

# サーバーとクライアントを起動
echo "3. 連合学習システムを起動中..."
docker-compose up -d

echo "4. 起動状況を確認中..."
sleep 5

# コンテナの状態を確認
docker-compose ps

echo ""
echo "=== 起動完了 ==="
echo "ログを確認するには以下のコマンドを使用してください："
echo "  サーバーログ: docker-compose logs -f server"
echo "  全クライアントログ: docker-compose logs -f client1 client2 client3"
echo "  特定クライアントログ: docker-compose logs -f client1"
echo ""
echo "システムを停止するには: docker-compose down"