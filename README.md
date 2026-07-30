# video-transcriber

Python + Streamlit 製の動画文字起こし・字幕焼き込み Web アプリ。

## 概要

1. Streamlit 画面で動画をアップロード（最大 4GB）
2. 常駐 FFmpeg コンテナ（linuxserver/ffmpeg）で音声抽出（ストリームコピー）＋ 動画長取得
3. ローカル Whisper（faster-whisper `large-v3` / CPU / int8）で文字起こし
4. `st.data_editor` で文字起こしテキストをセグメント単位で編集
5. 「動画作成」選択時、字幕（SRT）を動画に焼き込んで出力・ダウンロード

## ディレクトリ構成

```
video-transcriber/
├── app.py                  # Streamlit アプリ本体
├── docker-compose.yml      # FFmpeg コンテナ定義
├── requirements.txt        # Python 依存パッケージ
├── pytest.ini              # pytest 設定
├── .streamlit/
│   └── config.toml         # maxUploadSize = 4096 MB
├── src/
│   ├── ffmpeg_client.py    # Docker SDK 経由の FFmpeg/ffprobe クライアント
│   ├── transcriber.py      # faster-whisper 文字起こし
│   ├── subtitle.py         # SRT 生成ユーティリティ
│   ├── job_logger.py       # JSON Lines ロガー（ジョブID付き）
│   └── jobs.py             # ジョブディレクトリ管理・クリーンアップ
├── shared/
│   ├── fonts/              # 日本語フォント配置先（手順は下記参照）
│   └── jobs/               # ジョブごとの一時ファイル（自動生成）
├── logs/
│   └── app.jsonl           # JSON Lines ログ（日次ローテーション）
└── tests/
    ├── unit/               # 単体テスト（Docker・Whisper 不要）
    ├── integration/        # 結合テスト（ffmpeg-worker コンテナ必要）
    └── e2e/                # E2E テスト（Streamlit AppTest）
```

## 前提環境

- Windows 11 + Docker Desktop（WSL2）
- Python 3.10 以上
- GPU 不要（CPU のみで動作）

## セットアップ手順

### 1. Docker Desktop を起動

Docker Desktop を起動し、WSL2 バックエンドが有効であることを確認してください。

### 2. FFmpeg コンテナを起動

```bash
docker compose up -d
```

`ffmpeg-worker` コンテナが起動します（`docker ps` で確認）。

### 3. 日本語フォントを配置（字幕焼き込みに必要）

1. [Noto CJK フォント](https://github.com/googlefonts/noto-cjk/releases) から
   `NotoSansCJK-Regular.ttc`（または任意の CJK フォントファイル）をダウンロード
2. `shared/fonts/` フォルダに配置

```
shared/
└── fonts/
    └── NotoSansCJK-Regular.ttc   ← ここに配置
```

フォントが未配置の場合はアプリが警告を表示します（CJK 文字が文字化けする可能性あり）。

### 4. Python 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

> **注意**: `faster-whisper` の初回起動時に `large-v3` モデル（約 3GB）が
> 自動ダウンロードされます。

### 5. アプリを起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開いてください。

## 機能説明

| 機能 | 説明 |
|---|---|
| 動画アップロード | mp4 / mov / avi / mkv 等、最大 4GB |
| 音声抽出 | FFmpeg コンテナでストリームコピー（無劣化・高速） |
| 動画長取得 | ffprobe で取得し、進捗表示に利用 |
| 文字起こし | faster-whisper `large-v3`（CPU / int8 量子化） |
| テキスト編集 | セグメント単位で開始・終了時刻とテキストを編集可能 |
| 字幕焼き込み | FFmpeg コンテナで SRT 字幕を動画に焼き込み |
| ダウンロード | 字幕入り動画（mp4）と SRT ファイルをダウンロード |

## ログ仕様

- ファイル: `logs/app.jsonl`（JSON Lines 形式、日次ローテーション）
- 全ログにジョブ ID（UUID）を付与

各ログレコードのフィールド:

| フィールド | 内容 |
|---|---|
| `timestamp` | ISO 8601 UTC タイムスタンプ |
| `job_id` | ジョブ UUID |
| `step` | 工程名（upload / extract_audio / probe_duration / transcribe / burn_subtitles / cleanup） |
| `filename` | 入力ファイル名 |
| `video_duration` | 動画長（秒） |
| `elapsed` | 処理時間（秒） |
| `success` | 成否（true / false / null=進行中） |
| `error` | エラー内容 |

## テスト実行方法

### 単体テスト（Docker・Whisper 不要）

```bash
pytest -m "not integration and not e2e"
```

### 結合テスト（ffmpeg-worker コンテナ必要）

```bash
docker compose up -d
pytest -m integration
```

### E2E テスト（Streamlit AppTest 利用）

```bash
pytest -m e2e
```

### 全テスト実行

```bash
pytest
```

## 注意事項

- `large-v3` モデルは CPU 環境で 1 時間の音声を文字起こしするのに **1〜3 時間程度** かかる場合があります。
- 字幕焼き込みも CPU で全編再エンコードするため、長時間動画では相応の時間がかかります。
- 処理中はブラウザのタブを閉じないでください（処理が中断されます）。
- `shared/` および `logs/` は `.gitignore` に含まれており、Git 管理対象外です。
