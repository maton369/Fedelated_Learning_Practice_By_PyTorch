# ベースイメージ：軽量な Python 3.10
FROM python:3.10-slim

# 作業ディレクトリの設定
WORKDIR /app

# Pythonの依存関係ファイルをコピー
COPY requirements.txt .

# 必要なパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードを全てコピー（docker-composeでマウントするのであくまで初期用）
COPY . .

# Flaskの標準ポートを開放（必要に応じて）
EXPOSE 5000

# デフォルトはヘルスチェック代わり。サーバー・クライアントごとに `command:` で上書き
CMD ["python", "-c", "print('Dockerイメージが正しく起動しました')"]