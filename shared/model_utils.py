import torch
import torch.nn as nn
import io

# ===============================
# モデル定義（全クライアント・サーバー共通）
# ===============================


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # ロジット出力（ソフトマックスは使わない）
        return x


def get_model():
    """
    サーバーおよび各クライアントで同じモデルを初期化するための関数。
    """
    return SimpleMLP()


# ===============================
# モデルのシリアライズ / デシリアライズ処理
# ===============================


def save_model_to_bytes(model):
    """
    PyTorchモデルのstate_dictをバイト列に変換し、POST送信可能にする。
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer


def load_state_dict_from_bytes(model, byte_data):
    """
    バイト列からstate_dictを読み込み、指定したモデルにロードする。
    """
    buffer = io.BytesIO(byte_data)
    state_dict = torch.load(buffer, map_location="cpu")
    model.load_state_dict(state_dict)
    return model
