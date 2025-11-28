#!/usr/bin/env python3
"""
対話CLI

FastAPI 経由でキャラクターと対話する CLI ツール
"""

import argparse
import json
import sys
from typing import Optional

import requests

# API エンドポイント
API_BASE_URL = "http://localhost:8000"


class Colors:
    """ANSI カラーコード"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # キャラクター
    CHARACTER = "\033[96m"  # シアン

    # ユーザ
    USER = "\033[93m"  # 黄色

    # システム
    SYSTEM = "\033[90m"  # グレー
    SUCCESS = "\033[92m"  # 緑
    ERROR = "\033[91m"  # 赤

    # 引用
    CITATION = "\033[35m"  # マゼンタ


def print_header():
    """ヘッダーを表示"""
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}🐱 吾輩は猫である - 対話CLI{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print()


def print_help():
    """ヘルプを表示"""
    print(f"{Colors.SYSTEM}コマンド:{Colors.RESET}")
    print(f"  {Colors.BOLD}:help{Colors.RESET}      - ヘルプを表示")
    print(f"  {Colors.BOLD}:pos <数字>{Colors.RESET} - 現在位置を変更（例: :pos 20000）")
    print(f"  {Colors.BOLD}:char <名>{Colors.RESET}  - キャラクターを変更（例: :char 迷亭）")
    print(f"  {Colors.BOLD}:history{Colors.RESET}   - 会話履歴を表示")
    print(f"  {Colors.BOLD}:clear{Colors.RESET}     - 画面をクリア")
    print(f"  {Colors.BOLD}:quit{Colors.RESET}      - 終了（または Ctrl+C）")
    print()


def check_api_health() -> bool:
    """API の接続確認"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("ok")
    except Exception as e:
        print(f"{Colors.ERROR}❌ API に接続できません: {e}{Colors.RESET}")
        print(f"{Colors.SYSTEM}サーバを起動してください: python -m uvicorn app.dialog_api:app --port 8000{Colors.RESET}")
        return False


def get_available_characters() -> list:
    """利用可能なキャラクター一覧（静的リスト）"""
    return [
        "吾輩", "水島 寒月", "迷亭", "金田", "越智 東風",
        "珍野 苦沙弥", "八木 独仙", "御三", "雪江",
        "甘木先生", "金田 鼻子", "多々良 三平"
    ]


def send_message(
    character: str,
    question: str,
    pos: int,
    session_id: str,
    k: int = 8,
    temperature: float = 0.4
) -> Optional[dict]:
    """API にメッセージを送信"""
    try:
        payload = {
            "character": character,
            "question": question,
            "pos": pos,
            "session_id": session_id,
            "k": k,
            "temperature": temperature
        }

        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(f"{Colors.ERROR}❌ エラー ({response.status_code}): {response.text}{Colors.RESET}")
            return None

    except Exception as e:
        print(f"{Colors.ERROR}❌ 送信エラー: {e}{Colors.RESET}")
        return None


def get_session_history(session_id: str) -> Optional[list]:
    """セッション履歴を取得"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/sessions/{session_id}",
            timeout=10
        )

        if response.status_code == 200:
            return response.json()
        else:
            return None

    except Exception as e:
        print(f"{Colors.ERROR}❌ 履歴取得エラー: {e}{Colors.RESET}")
        return None


def display_response(data: dict, verbose: bool = False):
    """レスポンスを整形表示"""
    answer = data["answer"]
    citations = data["citations"]

    # 返答を表示
    print(f"{Colors.CHARACTER}{data.get('character', '???')}: {Colors.RESET}{answer}")
    print()

    # 詳細情報（verbose モード）
    if verbose:
        print(f"{Colors.DIM}─ 詳細情報 ─{Colors.RESET}")
        print(f"{Colors.DIM}検索方法: {data['method']}{Colors.RESET}")
        print(f"{Colors.DIM}使用チャンク: {len(data['used_chunks'])} 件{Colors.RESET}")

        if citations:
            print(f"{Colors.DIM}引用箇所:{Colors.RESET}")
            for i, c in enumerate(citations[:3], 1):
                print(f"{Colors.DIM}  {i}. 第{c['chapter']}章 位置 {c['start']}-{c['end']}{Colors.RESET}")
            if len(citations) > 3:
                print(f"{Colors.DIM}  ... 他 {len(citations) - 3} 件{Colors.RESET}")
        print()


def display_history(history: list):
    """履歴を表示"""
    print(f"{Colors.SYSTEM}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}📜 会話履歴 ({len(history)} ターン){Colors.RESET}")
    print(f"{Colors.SYSTEM}{'─' * 60}{Colors.RESET}")

    for i, msg in enumerate(history, 1):
        role = msg["role"]
        content = msg["content"]
        pos = msg.get("pos")

        if role == "user":
            print(f"{Colors.USER}[{i}] あなた:{Colors.RESET} {content}")
        else:
            # 長い場合は省略
            display_content = content[:150] + "..." if len(content) > 150 else content
            print(f"{Colors.CHARACTER}[{i}] キャラクター:{Colors.RESET} {display_content}")

        if pos:
            print(f"{Colors.DIM}    (位置: {pos}){Colors.RESET}")
        print()


def interactive_mode(
    character: str,
    pos: int,
    session_id: str,
    verbose: bool
):
    """対話モード"""
    print_header()

    # API 接続確認
    if not check_api_health():
        sys.exit(1)

    print(f"{Colors.SUCCESS}✓ API 接続確認{Colors.RESET}")
    print()
    print(f"{Colors.SYSTEM}対話相手: {Colors.CHARACTER}{character}{Colors.RESET}")
    print(f"{Colors.SYSTEM}現在位置: {Colors.BOLD}{pos}{Colors.RESET}")
    print(f"{Colors.SYSTEM}セッション: {Colors.DIM}{session_id}{Colors.RESET}")
    print()
    print(f"{Colors.DIM}コマンドは ':help' で確認できます{Colors.RESET}")
    print(f"{Colors.DIM}終了するには ':quit' または Ctrl+C{Colors.RESET}")
    print()
    print(f"{Colors.SYSTEM}{'─' * 60}{Colors.RESET}")
    print()

    current_character = character
    current_pos = pos

    try:
        while True:
            # 入力を受け付け
            try:
                user_input = input(f"{Colors.USER}> {Colors.RESET}").strip()
            except EOFError:
                print()
                break

            if not user_input:
                continue

            # コマンド処理
            if user_input.startswith(":"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1] if len(cmd_parts) > 1 else None

                if cmd == ":help":
                    print_help()
                    continue

                elif cmd == ":pos":
                    if arg and arg.isdigit():
                        current_pos = int(arg)
                        print(f"{Colors.SUCCESS}✓ 位置を {current_pos} に変更{Colors.RESET}")
                    else:
                        print(f"{Colors.ERROR}使い方: :pos <数字>{Colors.RESET}")
                    print()
                    continue

                elif cmd == ":char":
                    if arg:
                        available = get_available_characters()
                        if arg in available:
                            current_character = arg
                            print(f"{Colors.SUCCESS}✓ キャラクターを {current_character} に変更{Colors.RESET}")
                        else:
                            print(f"{Colors.ERROR}利用可能なキャラクター:{Colors.RESET}")
                            for c in available:
                                print(f"  - {c}")
                    else:
                        print(f"{Colors.ERROR}使い方: :char <キャラクター名>{Colors.RESET}")
                    print()
                    continue

                elif cmd == ":history":
                    history = get_session_history(session_id)
                    if history:
                        display_history(history)
                    else:
                        print(f"{Colors.SYSTEM}履歴がありません{Colors.RESET}")
                        print()
                    continue

                elif cmd == ":clear":
                    print("\033[2J\033[H")  # 画面クリア
                    print_header()
                    continue

                elif cmd == ":quit" or cmd == ":exit":
                    break

                else:
                    print(f"{Colors.ERROR}不明なコマンド: {cmd}{Colors.RESET}")
                    print(f"{Colors.SYSTEM}':help' でヘルプを表示{Colors.RESET}")
                    print()
                    continue

            # 通常の質問として送信
            print(f"{Colors.DIM}送信中...{Colors.RESET}", end="\r")

            response = send_message(
                character=current_character,
                question=user_input,
                pos=current_pos,
                session_id=session_id
            )

            if response:
                print(" " * 20, end="\r")  # "送信中..." を消す
                display_response(response, verbose=verbose)
            else:
                print()

    except KeyboardInterrupt:
        print()
        print(f"{Colors.SYSTEM}終了します{Colors.RESET}")

    print()
    print(f"{Colors.SYSTEM}{'─' * 60}{Colors.RESET}")
    print(f"{Colors.DIM}セッション ID: {session_id}{Colors.RESET}")
    print(f"{Colors.DIM}履歴は保存されています{Colors.RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="吾輩は猫である - 対話CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python app/chat_cli.py                          # デフォルト設定で起動
  python app/chat_cli.py --char 迷亭              # 迷亭と対話
  python app/chat_cli.py --pos 20000              # 位置20000から
  python app/chat_cli.py --session my_session     # セッションIDを指定
  python app/chat_cli.py --verbose                # 詳細情報を表示
        """
    )

    parser.add_argument(
        "--char", "--character",
        default="吾輩",
        help="対話するキャラクター名（デフォルト: 吾輩）"
    )

    parser.add_argument(
        "--pos", "--position",
        type=int,
        default=10000,
        help="本文の現在位置（デフォルト: 10000）"
    )

    parser.add_argument(
        "--session", "--session-id",
        default="cli_session",
        help="セッションID（デフォルト: cli_session）"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="詳細情報を表示"
    )

    args = parser.parse_args()

    # 対話モード起動
    interactive_mode(
        character=args.char,
        pos=args.pos,
        session_id=args.session,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
