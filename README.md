# video-transcriber

Python + Streamlit 製の動画文字起こし・翻訳・字幕焼き込み Web アプリ。

## 概要

処理の流れ（④ のアプリ主処理フロー）:

1. Streamlit 画面で動画をアップロード（最大 4GB）
2. アップロードされた動画を FFmpeg コンテナへ転送（`docker cp`）
3. FFmpeg コンテナ側で音声の長さ取得（ffprobe）と音声抽出（ffmpeg）を実行（`exec_run`）
4. 抽出された音声ファイルを Streamlit サーバー側へ転送（`docker cp`）
5. ローカル Whisper（faster-whisper `large-v3` / CPU / int8）で文字起こし
   （画面で Azure AI Speech を選んだ場合は Fast Transcription API で文字起こし。出力形式は Whisper と同じ）
6. 文字起こし結果を GPT（OpenAI API）で翻訳（`OPENAI_API_KEY` が未設定の場合はスキップ）
7. 文字起こしテキストと翻訳テキストを画面に表示（編集も可能）
8. 動画＋字幕（SRT）を FFmpeg コンテナ側へ転送（`docker cp`）
9. FFmpeg コンテナ側で動画に字幕を焼き込んで動画生成（`exec_run`）
10. 生成された動画を Streamlit サーバー側へ転送（`docker cp`）し、画面に表示・ダウンロード

各ジョブの実行結果（エンジン・工程ごとの処理時間・文字起こし・翻訳・字幕）は `results/` とジョブフォルダに JSON で保存されます（[実行結果（JSON）](#実行結果json)）。

## ディレクトリ構成

```
video-transcriber/
├── app.py                  # Streamlit アプリ本体
├── docker-compose.yml      # FFmpeg コンテナ + PostgreSQL 定義
├── requirements.txt        # Python 依存パッケージ
├── .env.example            # 設定ファイルのひな形（.env にコピーして使用）
├── pytest.ini              # pytest 設定
├── .streamlit/
│   └── config.toml         # maxUploadSize = 4096 MB
├── scripts/
│   └── compare_engines.py  # Whisper / Azure の文字起こし比較ツール
├── src/
│   ├── ffmpeg_client.py    # Docker SDK クライアント（docker cp / exec_run / 診断）
│   ├── transcriber.py      # faster-whisper 文字起こし / エンジン選択
│   ├── azure_transcriber.py  # Azure AI Speech Fast Transcription（REST）
│   ├── translator.py       # GPT 翻訳（OpenAI API、未設定時はスキップ）
│   ├── subtitle.py         # SRT 生成ユーティリティ
│   ├── compare.py          # エンジン比較の集計・レポート生成
│   ├── results.py          # ジョブごとの実行結果 JSON
│   ├── env.py              # .env の読み込み
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

### 5. 設定ファイル（.env）を作成

```powershell
copy .env.example .env
```

`.env` に API キーなどを記入します（`.env` は `.gitignore` 済み）。
起動時に自動で読み込まれ、PowerShell の `$env:NAME = ...` で設定済みの値がある場合はそちらが優先されます。
設定できる項目は「環境変数」の表を参照してください。

翻訳機能を使う場合は `OPENAI_API_KEY` を設定します:

```dotenv
OPENAI_API_KEY=sk-...
```

未設定の場合は翻訳ステップがスキップされ、文字起こしテキストがそのまま字幕になります。

### 6. （任意）Azure AI Speech で文字起こしする

文字起こしエンジンは画面の「Transcription engine」で Whisper / Azure AI Speech を選べます
（初期選択は `TRANSCRIBE_ENGINE`、未設定なら Whisper）。
Azure AI Speech（Fast Transcription API）を使う場合は、Speech リソース（Standard S0）の
キーとエンドポイント（またはリージョン）を `.env` に設定します:

```dotenv
TRANSCRIBE_ENGINE=azure
AZURE_SPEECH_KEY=<Speech リソースのキー>
AZURE_SPEECH_ENDPOINT=https://<リソース名>.cognitiveservices.azure.com
# または: AZURE_SPEECH_REGION=japaneast
```

- 音声は FFmpeg コンテナでモノラル 16kHz・Opus 32kbps（約 14MB/時間）に変換してから送信するため、
  動画ファイルが 1GB を超えていても API の上限（1 ファイル 500MB 未満・5 時間未満）に収まります。
- 音声は分割せず、1 ファイルを 1 回のリクエストで送信します。API の上限（1 ファイル 5 時間未満）を超える 5 時間以上の動画は、送信前にエラーにします。
- 言語は日本語・英語・中国語・ドイツ語（`ja-JP` / `en-US` / `zh-CN` / `de-DE`、固定）の中から、主要な 1 言語を自動判定します。
- 社名・人名・製品名などは `AZURE_PHRASE_LIST`（カンマ区切り）に登録すると認識されやすくなります（フレーズリスト）。
- 判定された言語は `result.json` の `transcription.language` に、発話時間の長い順で記録され、結果画面にも表示されます。
- 話者識別（diarization）を既定で有効にし、各セグメントに `speaker`（API が返す整数）を付けます。`AZURE_DIARIZATION=false` で無効化、`AZURE_MAX_SPEAKERS`（2〜35、既定 4）で想定する最大話者数を指定します。
  Fast Transcription API で指定できるのは最大話者数だけで、最小話者数の設定はありません。
- Azure が返す発話（phrase）1 件を、そのまま 1 セグメントにします（長い発話の分割はしません）。
- Azure の設定が不足している状態で Azure を選ぶと、画面にエラーを表示し、文字起こしボタンを無効にします（Whisper へは自動で切り替わりません）。
- 音声は Azure に送信され、利用量に応じて課金されます。

### 7. アプリを起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開いてください。

## 環境変数

`.env` または OS の環境変数で設定します（OS の環境変数が優先）。

| 変数 | 既定値 | 説明 |
|---|---|---|
| `DATABASE_URL` | `******localhost:5432/video_transcriber` | 実行履歴 DB の接続文字列 |
| `OPENAI_API_KEY` | （なし） | GPT 翻訳用の API キー。未設定なら翻訳スキップ |
| `TRANSCRIBE_ENGINE` | `whisper` | 画面のエンジン選択の初期値。`whisper`（ローカル）または `azure`（Azure AI Speech Fast Transcription） |
| `AZURE_SPEECH_KEY` | （なし） | Azure Speech リソースのキー（`azure` 時は必須） |
| `AZURE_SPEECH_ENDPOINT` | （なし） | 例: `https://<リソース名>.cognitiveservices.azure.com`。未設定なら `AZURE_SPEECH_REGION` から生成 |
| `AZURE_SPEECH_REGION` | （なし） | 例: `japaneast`（`AZURE_SPEECH_ENDPOINT` 未設定時に使用） |
| `AZURE_PHRASE_LIST` | （なし） | 認識させたい固有名詞・専門用語（カンマ区切り） |
| `AZURE_DIARIZATION` | `true` | Azure の話者識別。有効時は各セグメントに `speaker` を付与（Whisper は常に `null`） |
| `AZURE_MAX_SPEAKERS` | `4` | 話者識別で想定する最大話者数（2〜35）。最小話者数は指定できません |
| `AZURE_REQUEST_TIMEOUT` | `3600` | 応答待ちのタイムアウト秒数 |
| `KEEP_TEMP_FILES` | （なし） | `1` / `true` を設定すると、完了・異常終了時にホスト・コンテナ両方の temp フォルダを削除しない（テスト実行用） |
| `EXEC_TIMEOUT` | `21600` (6h) | コンテナ内コマンド 1 回あたりのタイムアウト秒数 |
| `SHARED_DIR` | `shared` | ホスト側共有ディレクトリ |
| `LOG_DIR` | `logs` | JSONL ログ出力先 |
| `RESULTS_DIR` | `results` | ジョブごとの実行結果 JSON の保存先（自動削除されない） |

## 実行結果（JSON）

ジョブごとに次の 2 か所へ同じ JSON ファイルを保存し、処理の進行に合わせて上書き更新します。

| 保存先 | 説明 |
|---|---|
| `results/<日時>_<ファイル名>_<ジョブID先頭8桁>/` | 自動削除されない保存用フォルダ（`RESULTS_DIR` で変更可能） |
| `shared/jobs/<job_id>/` | 動画・音声と同じ処理用フォルダ。「Start New Job」・「Clean Up Current Job」・エラー時に動画と一緒に削除されます（`KEEP_TEMP_FILES` 設定時を除く） |

| ファイル | 内容 |
|---|---|
| `result.json` | ジョブ全体（入力・エンジン設定・工程ごとの処理時間・各結果・エラー） |
| `transcript.json` | 文字起こし結果のセグメント配列 |
| `translation.json` | 翻訳結果のセグメント配列（翻訳した場合のみ） |
| `subtitles.json` | 動画に焼き込んだ字幕のセグメント配列（焼き込み後のみ） |
| `azure_response.json` | Azure Fast Transcription API のレスポンス（`response.json()`）をそのまま保存したもの（Azure 実行時のみ）。送信した音声ファイル名（`file`）と `definition`（候補言語・話者識別設定）も含みます。`response` 内の `offsetMilliseconds` は、そのまま動画上の時刻です |

セグメント配列は Whisper・Azure 共通で次の形式です（`speaker` は Azure の話者識別結果、Whisper では `null`）:

```json
[
  { "speaker": 1, "start": 0.5, "end": 2.0, "text": "こんにちは" },
  { "speaker": 2, "start": 2.3, "end": 4.1, "text": "よろしくお願いします" }
]
```

`result.json` の `status` は以下のタイミングで更新されます。

| タイミング | `status` |
|---|---|
| 文字起こし（＋翻訳）完了 | `transcribed` |
| 字幕焼き込み完了 | `completed`（焼き込みに使った字幕 `subtitles` を含む） |
| いずれかの工程で失敗 | `failed`（`error.step` / `error.message` に失敗工程と内容） |

画面の「⬇ Download Result JSON」からは `result.json` と同じ内容をダウンロードできます（編集中の字幕を反映した内容）。

```jsonc
{
  "schema_version": 1,
  "job_id": "…",
  "status": "transcribed",
  "created_at": "2026-11-11T07:30:00+00:00",
  "updated_at": "2026-11-11T07:31:12+00:00",
  "input": { "filename": "sample.mp4", "size_bytes": 123456789, "duration_seconds": 61.2 },
  "engine": { "name": "azure", "api_version": "2025-10-15", "locales": ["ja-JP", "en-US", "zh-CN", "de-DE"], "diarization": true, "max_speakers": 4,
              "audio_file": { "file": "sample_speech.ogg", "size_bytes": 245760 } },
  "steps": [ { "step": "upload", "elapsed_seconds": 1.2, "success": true, "error": null }, … ],
  "transcription": { "segment_count": 12, "text": "…", "segments": [ { "speaker": 1, "start": 0.5, "end": 2.0, "text": "…" } ] },
  "translation": { "status": "skipped", "model": null, "target_language": null, "message": "OPENAI_API_KEY is not set…", "segments": null },
  "subtitles": null,
  "output": null,
  "error": null
}
```

セグメント（`transcription` / `translation` / `subtitles` の `segments`）は Whisper・Azure 共通で `speaker` / `start` / `end` / `text` のキーを持ちます。`speaker` は Azure の話者識別結果（整数）で、Whisper（話者識別なし）では `null` です。

API キーなどの秘密情報は保存しません。

## 文字起こしエンジンの比較（Whisper / Azure）

同じ動画を両方のエンジンで文字起こしし、結果と処理時間を比較できます
（`docker compose up -d` 済み、Azure の設定は `.env` から読み込み）。

```powershell
# 先頭 5 分だけで比較（Whisper large-v3 は CPU だと長時間かかるため、まずは短めがおすすめ）
python scripts/compare_engines.py sample.mp4 --clip-seconds 300

# 正解の文字起こしテキスト（UTF-8）があれば、それぞれの一致率も算出
python scripts/compare_engines.py sample.mp4 --clip-seconds 300 --reference sample_ref.txt

# 片方だけ / Whisper のモデルや言語を変えて計測
python scripts/compare_engines.py sample.mp4 --engines azure
python scripts/compare_engines.py sample.mp4 --whisper-model medium --whisper-language ja
```

アプリと同じ処理（コンテナでの音声抽出 → コピー → 文字起こし）を工程ごとに計測し、
`comparison/<日時>_<ファイル名>/` に以下を出力します（`comparison/` は Git 管理対象外）。

| ファイル | 内容 |
|---|---|
| `report.html` | 速度・統計の表と、一定時間（既定 30 秒）ごとに両エンジンの文字起こしを横並びにした表 |
| `results.json` | 工程別の処理時間、RTF、セグメント数、文字数、一致率など |
| `whisper.srt` / `azure.srt` | 各エンジンの字幕ファイル |
| `azure_response.json` | Azure API のレスポンス（`response.json()`）そのまま |
| `whisper.txt` / `azure.txt` | 各エンジンのテキスト |

- **RTF** = 処理時間 ÷ 音声の長さ（1.0 未満なら実時間より速い）
- **一致率** は記号・空白・全角半角の違いを除いた文字単位の類似度です。エンジン間の一致率はどちらが正しいかを示さないため、精度を比べるには `--reference` を使ってください。
- Whisper の「モデル読み込み」は初回のモデルダウンロード時間を含む場合があります。Azure の「文字起こし」はアップロードとネットワーク往復を含みます。
- Azure は実行した音声の長さに応じて課金されます。

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

`.env` に `KEEP_TEMP_FILES=1` を設定して起動します:

```powershell
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

### 単体テスト（Docker・Whisper・Azure・DB 不要）

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

- Azure AI Speech を使う場合、文字起こし中は進捗バーではなくスピナーを表示します（結果は一括で返るため）。
- `large-v3` モデルは CPU 環境で 1 時間の音声を文字起こしするのに **1〜3 時間程度** かかる場合があります。
- 字幕焼き込みも CPU で全編再エンコードするため、長時間動画では相応の時間がかかります。
- 処理中はブラウザのタブを閉じないでください（処理が中断されます）。
- `shared/` および `logs/` は `.gitignore` に含まれており、Git 管理対象外です。
- FFmpeg コンテナ内の temp フォルダ（`/tmp/vt-jobs/`）は完了・異常終了時に自動削除されます（`KEEP_TEMP_FILES` 設定時を除く）。
