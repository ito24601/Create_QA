import json
import os
import random
import asyncio
import argparse
from collections import defaultdict
from typing import List, Set, Tuple
from pydantic import BaseModel
import openai # `WebSearch.py` には直接 `openai` のimportはないが、エージェント内で利用されている想定
from dotenv import load_dotenv

load_dotenv("/app/.env", override=True) # .envファイルから環境変数を読み込む

# ペルソナ候補
PERSONAS = ["エンジニア", "マーケター", "学生", "経営者", "一般ユーザー", "営業担当者", "カスタマーサポート", "法務担当者"]
# 1 URL あたりの最大 QA 数
MAX_QA_PER_URL = 50
# 1回のLLM呼び出しで生成を試みるQA数
QA_PER_REQUEST = 3 # WebSearch.py のエージェント指示に合わせて調整

# 出力フォーマット (WebSearch.py の QAPair を参考)
class QAPair(BaseModel):
    question: str
    answer: str
    source_url: str
    persona: str # どのペルソナの質問か

async def generate_qa_for_text(
    url: str,
    text_content: str,
    existing_qa_for_url: List[str],
    model_name: str,
    persona: str
) -> List[QAPair]:
    """
    指定されたテキストコンテンツとペルソナに基づいてQAペアを生成する。
    WebSearch.py のエージェント呼び出し部分を模倣。
    """
    # WebSearch.py のエージェント指示を参考にプロンプトを構築
    existing_qa_instructions_segment = "現在、このURLに関する既存のQ&Aはありません。"
    if existing_qa_for_url:
        existing_qa_str = "\n".join(existing_qa_for_url)
        existing_qa_instructions_segment = (
            f"以下のQ&Aペアは、このURL ({url}) に関して既に存在します。\n"
            f"これらとは異なる、新しい情報や視点からのQ&Aペアを生成してください。\n"
            f"---既存のQ&Aここから---\n"
            f"{existing_qa_str}\n"
            f"---既存のQ&Aここまで---"
        )

    system_prompt = (
        "You are a knowledge extraction assistant.\n"
        f"1. Your primary task is to analyze the provided text content from the URL: {url}.\n"
        f"2. The user is a '{persona}'. Generate questions and answers from their perspective.\n"
        f"3. {existing_qa_instructions_segment}\n"
        f"4. From this text content, extract up to {QA_PER_REQUEST} new question-answer pairs that would be genuinely helpful. Each pair must include the source URL, which MUST be exactly '{url}', and the persona '{persona}'.\n"
        "5. Avoid duplicate or trivial questions, including those listed in the existing Q&A section if provided.\n"
        "6. The extracted question and answer MUST be in Japanese.\n"
        "Return the result as a JSON list of QAPair objects. Each QAPair object should have 'question', 'answer', 'source_url', and 'persona' fields."
    )
    user_prompt = f"URL: {url}\nPersona: {persona}\nText Content:\n{text_content}"

    try:
        # OpenAI API呼び出し (WebSearch.pyではAgentクラス経由だが、ここでは直接呼び出す例)
        # 実際のエージェントライブラリの非同期呼び出しに合わせるのが理想
        response = await openai.ChatCompletion.acreate( # 非同期呼び出し
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"} # JSONモードを試みる
        )
        content = response.choices[0].message.content
        # JSON文字列がリスト形式で返ってくることを期待しているが、モデルの出力によっては調整が必要
        # 例: {"qa_pairs": [...]} のような形式で返ってくる場合
        if content:
            try:
                # モデルが直接リストを返さない場合、キーを指定して抽出
                # ここでは仮に 'qa_pairs' というキーでリストが返ってくると想定
                data = json.loads(content)
                qa_list_data = data.get("qa_pairs", []) if isinstance(data, dict) else data

                # QAPairオブジェクトに変換
                generated_qa_pairs = []
                for item in qa_list_data:
                    if isinstance(item, dict) and "question" in item and "answer" in item:
                         # モデルが source_url と persona を含めてくれない場合、ここで付与
                        generated_qa_pairs.append(QAPair(
                            question=item["question"],
                            answer=item["answer"],
                            source_url=url, # LLMが正しく付与しない場合があるので強制
                            persona=persona # LLMが正しく付与しない場合があるので強制
                        ))
                return generated_qa_pairs
            except json.JSONDecodeError as e:
                print(f"JSONデコードエラー: {e}, Content: {content}")
                return []
            except Exception as e:
                print(f"QAペアのパース中に予期せぬエラー: {e}")
                return []
        return []
    except Exception as e:
        print(f"OpenAI API呼び出し中にエラー: {e}")
        return []


async def process_url_content(
    url: str,
    full_text_content: str,
    outfile: str,
    model_name: str,
    max_qa_per_url: int
):
    """
    単一URLの全文コンテンツからQAを生成し、ファイルに追記する。
    WebSearch.py の collect_qa のロジックを参考にする。
    """
    print(f"\n--- URLの処理開始: {url} ---")
    # 既存のQAを読み込む (このURLに特化したもの + 全体で重複を避けるためのもの)
    existing_qa_for_this_url_display: List[str] = [] # プロンプト用
    all_existing_qa_tuples: Set[Tuple[str, str]] = set() # 全体の重複チェック用 (質問, 回答)

    if os.path.exists(outfile):
        try:
            with open(outfile, "r", encoding="utf-8") as reader:
                for line in reader:
                    try:
                        qa_obj_dict = json.loads(line)
                        q = qa_obj_dict.get("question")
                        a = qa_obj_dict.get("answer")
                        if q and a:
                            all_existing_qa_tuples.add((q, a))
                            if qa_obj_dict.get("source_url") == url:
                                existing_qa_for_this_url_display.append(f"- Q: {q}\n  A: {a}")
                    except json.JSONDecodeError:
                        print(f"警告: 出力ファイル '{outfile}' の行のJSONデコードに失敗しました: {line.strip()}")
        except Exception as e:
            print(f"警告: 既存の出力ファイル '{outfile}' の読み込み中にエラー: {e}")

    generated_qa_for_this_url_count = 0
    # このURLに対して既に生成されたQAの数を初期化 (ファイルから読み込んだ数ではない)
    # ファイルに書き込まれたQAの数をカウントするため、ループ内でインクリメントする

    # 試行回数やループ条件は WebSearch.py の collect_qa を参考にする
    # ここでは、MAX_QA_PER_URLに達するか、新しいQAが生成されなくなるまで繰り返す
    # 複数のペルソナを試すため、試行回数には余裕を持たせる
    max_attempts_per_persona_cycle = (max_qa_per_url // QA_PER_REQUEST) + len(PERSONAS) + 5 # 十分な試行回数

    for attempt in range(max_attempts_per_persona_cycle):
        if generated_qa_for_this_url_count >= max_qa_per_url:
            print(f"ℹ️ URL '{url}' のQA生成数が上限 ({max_qa_per_url}) に達しました。")
            break

        current_persona = random.choice(PERSONAS)
        print(f"試行 {attempt + 1}/{max_attempts_per_persona_cycle}, ペルソナ: {current_persona}, URL: {url}")

        # LLMからQAペアを取得
        # 既存のQA情報を渡して重複を避けるように指示
        newly_generated_pairs: List[QAPair] = await generate_qa_for_text(
            url,
            full_text_content,
            existing_qa_for_this_url_display, # このURLに関する既存QAを渡す
            model_name,
            current_persona
        )

        if not newly_generated_pairs:
            print(f"ℹ️ ペルソナ '{current_persona}' では新しいQAは生成されませんでした。")
            # 連続で生成失敗する場合、ペルソナを変えても効果が薄い可能性があるので、
            # 何回か失敗したらそのURLの処理を打ち切るロジックも検討できる
            if attempt > len(PERSONAS) * 2 and not newly_generated_pairs : # 簡易的な打ち切り条件
                 print(f"ℹ️ URL '{url}' で複数回新しいQAを生成できなかったため、処理を終了します。")
                 break
            await asyncio.sleep(1) # 短い待機
            continue

        # 重複チェックとファイル書き込み
        added_in_this_attempt_count = 0
        with open(outfile, "a", encoding="utf-8") as writer:
            for qa_pair in newly_generated_pairs:
                if generated_qa_for_this_url_count >= max_qa_per_url:
                    break # このURLのQA上限に達したらこの試行の残りもスキップ

                current_qa_tuple = (qa_pair.question, qa_pair.answer)
                # 全体の既存QAセットと、このURLで今回追加しようとしているものとの重複チェック
                if current_qa_tuple not in all_existing_qa_tuples:
                    writer.write(json.dumps(qa_pair.model_dump(), ensure_ascii=False) + "\n")
                    all_existing_qa_tuples.add(current_qa_tuple) # 全体セットに追加
                    # プロンプト用の既存QAリストにも追加 (次のLLM呼び出しのため)
                    existing_qa_for_this_url_display.append(f"- Q: {qa_pair.question}\n  A: {qa_pair.answer}")
                    generated_qa_for_this_url_count += 1
                    added_in_this_attempt_count += 1
                else:
                    print(f"フィルタリング(重複): Q: {qa_pair.question}")

        if added_in_this_attempt_count > 0:
            print(f"✨ この試行で {added_in_this_attempt_count} 件を新たに書き出しました (URL: {url}, Persona: {current_persona})。")
            print(f"累計: {generated_qa_for_this_url_count}/{max_qa_per_url} (URL: {url})")
        else:
            print(f"ℹ️ この試行では重複しない新しいQAはありませんでした (URL: {url}, Persona: {current_persona})。")

        if generated_qa_for_this_url_count >= max_qa_per_url: # 再度チェック
            print(f"ℹ️ URL '{url}' のQA生成数が上限 ({max_qa_per_url}) に達しました。")
            break
        
        await asyncio.sleep(random.uniform(1, 3)) # API負荷軽減のための待機

    print(f"--- URLの処理完了: {url}, 合計 {generated_qa_for_this_url_count} 件のQAを生成 ---")
    return generated_qa_for_this_url_count


async def main_process(input_jsonl_file: str, outfile: str, model_name: str, max_qa_per_url: int):
    # aflac_with_body.jsonl から URL と本文を読み込む
    # WebSearch.py では単一URLを引数に取るが、こちらはJSONL内の全URLを処理
    url_to_text_map = defaultdict(str)
    if not os.path.exists(input_jsonl_file):
        print(f"エラー: 入力ファイル '{input_jsonl_file}' が見つかりません。")
        return

    with open(input_jsonl_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            try:
                data = json.loads(line)
                url = data.get("url")
                body = data.get("body", "")
                if url and body:
                    # 同一URLの本文は連結していく（WebSearch.pyにはこの集約ロジックはない）
                    url_to_text_map[url] += body + "\n"
            except json.JSONDecodeError:
                print(f"警告: 入力ファイル '{input_jsonl_file}' の行のJSONデコードに失敗: {line.strip()}")

    if not url_to_text_map:
        print(f"エラー: 入力ファイル '{input_jsonl_file}' から処理可能なデータが読み込めませんでした。")
        return

    print(f"{len(url_to_text_map)} 件のユニークURLを処理対象とします。")
    total_qa_generated_all_urls = 0

    # 各URLのコンテンツに対してQA生成処理を呼び出す
    # asyncio.gather を使って並列処理も可能だが、APIレート制限やファイル書き込みの競合を考慮し、
    # ここでは順次処理とする。並列化する場合は書き込み処理の排他制御が必要。
    for url, full_text in url_to_text_map.items():
        if not full_text.strip():
            print(f"ℹ️ URL '{url}' の本文が空のためスキップします。")
            continue
        count = await process_url_content(url, full_text, outfile, model_name, max_qa_per_url)
        total_qa_generated_all_urls += count
        # URL間の処理にも少し間隔を空ける
        await asyncio.sleep(random.uniform(2, 5))


    print(f"\n--- 全処理完了 ---")
    print(f"🎉 合計 {total_qa_generated_all_urls} 件の新しいQ&Aをセッション中に書き出しました → {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定されたJSONLファイルから本文を読み込み、URLごとにQAペアを収集します。")
    parser.add_argument(
        "--input_file",
        type=str,
        default="aflac_with_body.jsonl",
        help="入力JSONLファイル名 (例: aflac_with_body.jsonl)"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="generated_qa_from_jsonl.jsonl",
        help="出力ファイル名 (例: output_qa.jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo", # より高速・安価なモデルをデフォルトに
        help="使用するモデル名 (例: gpt-4o, gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--max_qa_per_url",
        type=int,
        default=MAX_QA_PER_URL,
        help=f"1つのURLから生成する最大のQA数 (デフォルト: {MAX_QA_PER_URL})"
    )
    # WebSearch.py にあった max_attempts は、このスクリプトでは
    # max_qa_per_url と QA_PER_REQUEST, PERSONAS の数から動的に計算されるため、直接の引数にはしない。

    args = parser.parse_args()

    # OPENAI_API_KEY のチェック
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: 環境変数 OPENAI_API_KEY が設定されていません。")
        print("スクリプトの先頭近くにある load_dotenv('/app/.env', override=True) を確認するか、環境変数を設定してください。")
    else:
        asyncio.run(main_process(args.input_file, args.outfile, args.model, args.max_qa_per_url))
