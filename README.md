# video-transcriber

Python + Streamlit 製の動画文字起こし・字幕焼き込み Web アプリ。

## 概要

1. Streamlit 画面で動画をアップロード
2. 常駐 FFmpeg コンテナ(linuxserver/ffmpeg)で音声抽出(ストリームコピー)+ 動画長取得
3. ローカル Whisper(faster-whisper large-v3 / CPU)で文字起こし
4. 画面上で文字起こしテキストを表示・編集
5. 「動画作成」選択時、字幕(SRT)を動画に焼き込んで出力

## 前提環境

- Windows 11 + Docker Desktop (WSL2)
- Python 3.10+
- GPU 不要(CPU のみで動作)

詳細なセットアップ手順・実装は今後追加予定です。
