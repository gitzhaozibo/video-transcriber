# video-transcriber

Python + Streamlit 製の動画文字起こし・翻訳・字幕焼き込み Web アプリ。

## 概要

処理の流れ（④ のアプリ主処理フロー）:

1. Streamlit 画面で動画をアップロード（最大 4GB）
2. アップロードされた動画を FFmpeg コンテナへ転送（`docker cp`）
3. FFmpeg コンテナ側で音声の長さ取得（ffprobe）と音声抽出（ffmpeg）を実行（`exec_run`）
4. 抽出された音声ファイルを Streamlit サーバー側へ転送（`docker cp`）
5. ローカル Whisper（faster-whisper `large-v3` / CPU / int8）で文字起こし
6. 文字起こし結果を GPT（OpenAI API）で翻訳（`OPENAI_API_KEY` が未設定の場合はスキップ）
7. 文字起こしテキストと翻訳テキストを画面に表示（編集も可能）
8. 動画＋字幕（SRT）を FFmpeg コンテナ側へ転送（`docker cp`）
9. FFmpeg コンテナ側で動画に字幕を焼き込んで動画生成（`exec_run`）
10. 生成された動画を Streamlit サーバー側へ転送（`docker cp`）し、画面に表示・ダウンロード

## ディレクトリ構成

```
video-transcriber/
├── app.py                  # Streamlit アプリ本体
├── docker-compose.yml      # FFmpeg コンテナ + PostgreSQL 定義
├── requirements.txt        # Python 依存パッケージ
├── pytest.ini              # pytest 設定
├── .streamlit/
│   └── config.toml         # maxUploadSize = 4096 MB
├── src/
│   ├── ffmpeg_client.py    # Docker SDK クライアント（docker cp / exec_run / 診断）
│   ├── transcriber.py      # faster-whisper 文字起こし
│   ├── translator.py       # GPT 翻訳（OpenAI API、未設定時はスキップ）
│   ├── subtitle.py         # SRT 生成ユーティリティ
│   ├── db.py               # PostgreSQL 実行履歴（jobs / step_events テーブル）
│   ├── job_logger.py       # JSON Lines ロガー（DB へもデュアルライト）
│   └── jobs.py             # ジョブディレクトリ管理・クリーンアップ
├── shared/
│   ├── fonts/              # 日本語フォント配置先（コンテナへ /fonts としてマウント）
│   └── jobs/               # ジョブごとの一時ファイル（自動生成）
├── logs/
│   └── app.jsonl           # JSON Lines ログ（日次ローテーション）
└── tests/
    ├── unit/               # 単体テスト（Docker・Whisper・DB 不要）
    ├── integration/        # 結合テスト（ffmpeg-worker コンテナ必要）
    └── e2e/                # E2E テスト（Streamlit AppTest）
```

## 前提環境

- Windows 11 + Docker Desktop（Hyper-V バックエンド）
- Python 3.10 以上
- GPU 不要（CPU のみで動作）

## セットアップ手順

### 1. Docker Desktop を起動

Docker Desktop を起動し、Hyper-V バックエンドが有効であることを確認してください。

### 2. FFmpeg コンテナと PostgreSQL を起動

```bash
docker compose up -d
```

以下の 2 つのコンテナが常駐起動します（`docker ps` で確認）:

| コンテナ | イメージ | 用途 |
|---|---|---|
| `ffmpeg-worker` | `jrottenberg/ffmpeg:7.1-ubuntu` | ffprobe / ffmpeg を `exec_run` で実行。`sleep infinity` で常駐。複数セッションからの同時実行に対応 |
| `video-transcriber-db` | `postgres:16-alpine` | 実行履歴の保存（ポート 5432） |

### 3. 日本語フォントを配置（字幕焼き込みに必要）

1. [Noto CJK フォント](https://github.com/googlefonts/noto-cjk/releases) から
   `NotoSansCJK-Regular.ttc`（または任意の CJK フォントファイル）をダウンロード
2. `shared/fonts/` フォルダに配置（コンテナ内では `/fonts` として参照）

### 4. Python 依存パッケージをインストール

```bash
pip install -r requirements.txt
```

> **注意**: `faster-whisper` の初回起動時に `large-v3` モデル（約 3GB）が
> 自動ダウンロードされます。

### 5. （任意）OpenAI API キーを設定

翻訳機能を使う場合:

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

未設定の場合は翻訳ステップがスキップされ、文字起こしテキストがそのまま字幕になります。

### 6. アプリを起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開いてください。

## 環境変数

| 変数 | 既定値 | 説明 |
|---|---|---|
| `DATABASE_URL` | `******localhost:5432/video_transcriber` | 実行履歴 DB の接続文字列 |
| `OPENAI_API_KEY` | （なし） | GPT 翻訳用の API キー。未設定なら翻訳スキップ |
| `KEEP_TEMP_FILES` | （なし） | `1` / `true` を設定すると、完了・異常終了時にホスト・コンテナ両方の temp フォルダを削除しない（テスト実行用） |
| `EXEC_TIMEOUT` | `21600` (6h) | コンテナ内コマンド 1 回あたりのタイムアウト秒数 |
| `SHARED_DIR` | `shared` | ホスト側共有ディレクトリ |
| `LOG_DIR` | `logs` | JSONL ログ出力先 |

## 実行履歴（DB）

PostgreSQL に以下のテーブルが自動作成され、全ステップの開始・成功・失敗が
ジョブ ID 付きで記録されます（アプリのサイドバー「Execution History」でも確認可能）。

- `jobs` … ジョブごとの 1 行（`status`, `current_step`, `error`, タイムスタンプ）
- `step_events` … ステップイベントごとの 1 行（`status` = running / success / failure）

ハング箇所の特定例:

```sql
-- いまどこで止まっているか（running のままのステップを探す）
SELECT job_id, step, started_at
FROM step_events
WHERE status = 'running'
ORDER BY started_at;

-- 失敗したジョブとエラー内容
SELECT job_id, current_step, error
FROM jobs
WHERE status = 'failed';
```

接続コマンド:

```bash
docker exec -it video-transcriber-db psql -U vt_user -d video_transcriber
```

## トラブルシューティング（コンテナ無反応・Docker Desktop クラッシュ調査）

### アプリ内の診断機能

サイドバーの「🐳 Docker Diagnostics」→「Collect diagnostics」で以下を収集して画面表示できます:

- `docker version` / `docker info`
- `ffmpeg-worker` の状態・イメージ・起動時刻
- CPU / メモリ使用状況（`docker stats` 相当）
- コンテナ内プロセス一覧（`docker top` 相当）
- コンテナログ末尾 100 行

### コマンドでの調査

```bash
# コンテナの状態確認
docker ps -a
docker inspect ffmpeg-worker --format '{{.State.Status}} (OOMKilled={{.State.OOMKilled}}, ExitCode={{.State.ExitCode}})'

# リソース使用状況
docker stats --no-stream

# コンテナ内のプロセス確認（ffmpeg が残っていないか）
docker exec ffmpeg-worker ps aux

# コンテナログ
docker logs --tail 200 ffmpeg-worker

# Docker Desktop 全体のイベント（直近 1 時間）
docker events --since 1h

# コンテナの無反応時は再起動
docker compose restart ffmpeg-worker

# DB コンテナのログ
docker logs --tail 200 video-transcriber-db
```

### テスト実行時に一時ファイルを残す

```powershell
$env:KEEP_TEMP_FILES = "1"
streamlit run app.py
```

完了・異常終了後も `shared/jobs/<job_id>/`（ホスト側）と
`/tmp/vt-jobs/<job_id>/`（コンテナ側）が残るので、中間ファイルを調査できます。

```bash
# コンテナ側の temp フォルダを確認
docker exec ffmpeg-worker ls -la /tmp/vt-jobs/
```

### 同時実行について

複数ユーザー（複数ブラウザセッション）からの同時実行に対応しています。
Docker API へのアクセスはアプリ側のロックで直列化し、ジョブごとに独立した
temp フォルダ（ホスト: `shared/jobs/<uuid>/`、コンテナ: `/tmp/vt-jobs/<uuid>/`）
を使うため、Docker Desktop / Hyper-V への同時アクセス過多による無反応を抑えます。

## ログ仕様

- ファイル: `logs/app.jsonl`（JSON Lines 形式、日次ローテーション）
- 全ログにジョブ ID（UUID）を付与
- 同じ内容が PostgreSQL の `step_events` テーブルにも書き込まれます
  （DB 停止時は JSONL のみで継続動作）

各ログレコードのフィールド:

| フィールド | 内容 |
|---|---|
| `timestamp` | ISO 8601 UTC タイムスタンプ |
| `job_id` | ジョブ UUID |
| `step` | 工程名（upload / copy_to_container / probe_duration / extract_audio / copy_from_container / transcribe / translate / burn_subtitles / cleanup） |
| `filename` | 入力ファイル名 |
| `video_duration` | 動画長（秒） |
| `elapsed` | 処理時間（秒） |
| `success` | 成否（true / false / null=進行中） |
| `error` | エラー内容 |

## テスト実行方法

### 単体テスト（Docker・Whisper・DB 不要）

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
- FFmpeg コンテナ内の temp フォルダ（`/tmp/vt-jobs/`）は完了・異常終了時に自動削除されます（`KEEP_TEMP_FILES` 設定時を除く）。
