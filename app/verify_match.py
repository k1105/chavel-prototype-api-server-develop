"""フロントエンドとバックエンドの完全一致を確認"""
import json
from pathlib import Path
import re

def normalize_text(text: str) -> str:
    """フロントエンド形式にテキストを正規化"""
    # Step 1: 章マーカーとその直後の全角スペースを削除
    chapter_pattern = r'\n\n[一二三四五六七八九十]+\n\n\u3000?'
    text = re.sub(chapter_pattern, '', text)
    
    # Step 2: 改行+全角スペースのパターンを削除
    text = text.replace('\n\u3000', '')
    
    # Step 3: 残りの改行を削除
    text = text.replace('\n', '')
    
    # Step 4: 句点後の半角スペースを削除
    text = text.replace('。 ', '。')
    
    # Step 5: 連続する全角スペース（2個以上）を削除
    text = re.sub(r'\u3000{2,}', '', text)
    
    return text

def main():
    data_dir = Path('/Users/kanata/work/nnes/api-server/data')
    main_txt_path = data_dir / 'main.txt'
    chunks_path = data_dir / 'chunks.jsonl'
    
    # main.txtを正規化
    with open(main_txt_path, 'r', encoding='utf-8') as f:
        backend_text = f.read()
    
    backend_normalized = normalize_text(backend_text)
    
    # chunks.jsonlから全テキストを結合
    chunks_combined = ''
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunks_combined += chunk['text']
    
    print("🔍 完全一致の検証")
    print()
    print(f"Backend normalized length: {len(backend_normalized)}")
    print(f"Chunks combined length:    {len(chunks_combined)}")
    print(f"Difference:                {abs(len(backend_normalized) - len(chunks_combined))}")
    print()
    
    if backend_normalized == chunks_combined:
        print("✅ 完全一致！フロントエンドとバックエンドのテキストが100%一致しています")
        return 0
    else:
        print("❌ 不一致があります")
        
        # 最初の不一致位置を見つける
        for i, (b_char, c_char) in enumerate(zip(backend_normalized, chunks_combined)):
            if b_char != c_char:
                print()
                print(f"First mismatch at position {i}:")
                start = max(0, i - 30)
                end = min(len(backend_normalized), i + 30)
                print(f"Backend:  '{backend_normalized[start:end]}'")
                print(f"Chunks:   '{chunks_combined[start:end]}'")
                break
        
        return 1

if __name__ == '__main__':
    exit(main())
