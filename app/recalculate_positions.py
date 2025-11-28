"""
バックエンドの文字位置をフロントエンド形式に再計算するスクリプト

フロントエンドの正規化ルール:
- 改行(\n)を削除
- 全角スペース(\u3000)を削除
"""

import json
import sys
from pathlib import Path

def normalize_text(text: str) -> str:
    """フロントエンド形式にテキストを正規化

    フロントエンドは以下の正規化を行っている:
    1. 章マーカー(\n\n[一二三四五六七八九十]+\n\n)を削除
    2. 改行(\n)を削除
    3. 改行直後の全角スペース(\u3000)を削除
    4. 連続する全角スペースを削除
    5. 残った単一の全角スペースを半角スペースに変換
    """
    import re

    # Step 1: 章マーカーとその直後の全角スペースを削除（\n\n二\n\n\u3000 など）
    chapter_pattern = r'\n\n[一二三四五六七八九十]+\n\n\u3000?'
    text = re.sub(chapter_pattern, '', text)

    # Step 2: 改行+全角スペースのパターンを削除
    text = text.replace('\n\u3000', '')

    # Step 3: 残りの改行を削除
    text = text.replace('\n', '')

    # Step 4: 句点後の半角スペースを削除（。 → 。）
    text = text.replace('。 ', '。')

    # Step 5: 連続する全角スペース（2個以上）を削除
    text = re.sub(r'\u3000{2,}', '', text)

    # Step 6: 残った単一の全角スペースを半角スペースに変換
    text = text.replace('\u3000', ' ')

    return text

def create_position_mapping(backend_text: str) -> dict:
    """
    バックエンドの位置からフロントエンドの位置へのマッピングを作成

    Returns:
        dict: {backend_pos: frontend_pos}
    """
    mapping = {}
    frontend_pos = 0
    prev_char = ''

    for backend_pos, char in enumerate(backend_text):
        mapping[backend_pos] = frontend_pos

        # スキップする文字:
        # - 改行
        # - 句点の直後の半角スペース（。 の スペース部分）
        skip = False
        if char == '\n':
            skip = True
        elif char == ' ' and prev_char == '。':
            skip = True
        # 全角スペースはスキップしない（フロントエンドにも含まれている）

        if not skip:
            frontend_pos += 1

        prev_char = char

    # 最後の位置も追加
    mapping[len(backend_text)] = frontend_pos

    return mapping

def recalculate_chunks(chunks_path: Path, position_mapping: dict, max_pos: int) -> list:
    """chunks.jsonlの位置情報を再計算"""
    chunks = []
    cumulative_pos = 0  # 正規化後の累積位置

    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)

            # 位置情報を変換
            old_start = chunk['start_pos']
            old_end = chunk['end_pos']

            # 範囲外チェック
            if old_start > max_pos or old_end > max_pos:
                print(f"⚠️  Chunk {chunk['id']}: position out of range ({old_start}-{old_end}), skipping")
                continue

            # テキストを正規化
            normalized_text = normalize_text(chunk['text'])
            text_length = len(normalized_text)

            # 累積位置で新しいstart/endを設定
            new_start = cumulative_pos
            new_end = cumulative_pos + text_length

            chunk['start_pos'] = new_start
            chunk['end_pos'] = new_end
            chunk['text'] = normalized_text

            print(f"Chunk {chunk['id']}: {old_start}-{old_end} ({old_end-old_start}) -> {new_start}-{new_end} ({text_length})")

            cumulative_pos = new_end
            chunks.append(chunk)

    return chunks

def recalculate_events(events_path: Path) -> list:
    """events.jsonlの再計算（scene番号は変更なし）"""
    events = []

    with open(events_path, 'r', encoding='utf-8') as f:
        for line in f:
            event = json.loads(line)
            # イベントはscene番号のみで位置情報を持たないのでそのまま
            events.append(event)

    return events

def main():
    # パスの設定
    data_dir = Path('/Users/kanata/work/nnes/api-server/data')
    main_txt_path = data_dir / 'main.txt'
    chunks_path = data_dir / 'chunks.jsonl'
    events_path = data_dir / 'events.jsonl'

    # バックアップ
    chunks_backup = data_dir / 'chunks.jsonl.backup'
    events_backup = data_dir / 'events.jsonl.backup'

    print("📚 バックエンド位置情報の再計算を開始...")
    print()

    # main.txtを読み込み
    print("1. main.txtを読み込み中...")
    with open(main_txt_path, 'r', encoding='utf-8') as f:
        backend_text = f.read()
    print(f"   Backend text length: {len(backend_text)}")

    # 正規化後のテキスト長を確認
    normalized_text = normalize_text(backend_text)
    print(f"   Frontend text length: {len(normalized_text)}")
    print(f"   Difference: {len(backend_text) - len(normalized_text)}")
    print()

    # 位置マッピングを作成
    print("2. 位置マッピングを作成中...")
    position_mapping = create_position_mapping(backend_text)
    print(f"   Mapping created: {len(position_mapping)} positions")
    print()

    # chunksのバックアップと再計算
    print("3. chunks.jsonlをバックアップ中...")
    import shutil
    shutil.copy(chunks_path, chunks_backup)
    print(f"   Backup created: {chunks_backup}")
    print()

    print("4. chunks.jsonlの位置情報を再計算中...")
    max_pos = len(backend_text) - 1
    new_chunks = recalculate_chunks(chunks_path, position_mapping, max_pos)
    print(f"   Recalculated {len(new_chunks)} chunks")
    print()

    # 新しいchunksを保存
    print("5. 新しいchunks.jsonlを保存中...")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        for chunk in new_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"   Saved to {chunks_path}")
    print()

    # eventsのバックアップ
    print("6. events.jsonlをバックアップ中...")
    shutil.copy(events_path, events_backup)
    print(f"   Backup created: {events_backup}")
    print()

    # eventsは位置情報を持たないのでそのまま
    print("7. events.jsonlは位置情報を持たないため変更なし")
    print()

    print("✅ 完了!")
    print()
    print("次のステップ:")
    print("1. SQLiteデータベースをクリア: rm data/dialog.sqlite")
    print("2. バックエンドAPIサーバーを再起動")
    print("3. フロントエンドから動作確認")
    print()
    print("もし問題があれば、バックアップから復元できます:")
    print(f"  cp {chunks_backup} {chunks_path}")
    print(f"  cp {events_backup} {events_path}")

if __name__ == '__main__':
    main()
