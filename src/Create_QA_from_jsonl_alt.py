\
import asyncio
import jsonlines
import os
import argparse
from typing import List, Set, Tuple, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv

# agentsモジュールが Create_QA ディレクトリの親ディレクトリにあると仮定
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from agents import Agent, Runner # agentsモジュールからAgentとRunnerをインポート

load_dotenv("/app/.env", override=True)

# --- WebSearch.pyから流用 ---
class QAPair(BaseModel):
    question: str
    answer: str
    source_url: str # JSONLの各エントリの出典を示すために流用
    questioner_persona: str # 追加: どのような人がする質問か
    information_category: str  # 追加: 情報のカテゴリ
    related_keywords: List[str] # 追加: 関連キーワード

# --- WebSearch.pyのgenerate_qa_for_urlを改変 ---
async def generate_qa_for_content(
    source_identifier: str, # URLやファイル名など、コンテンツの出典
    text_content: str,
    existing_qa_for_source_display: List[str],
    model_name: str,
    max_q_per_content: int
) -> List[QAPair]:
    """
    与えられたテキストコンテンツからQ&Aペアを生成します。
    """
    if not existing_qa_for_source_display:
        existing_qa_instructions_segment = "There are currently no existing Q&A pairs for this source."
    else:
        existing_qa_str = "\\\\n".join(existing_qa_for_source_display) # existing_qa_for_source_display は "Q: ...\\nA: ..." 形式の文字列リストを想定
        existing_qa_instructions_segment = (
            f"The following Q&A pairs already exist for this source ({source_identifier}):\\\\n"
            f"Please generate new Q&A pairs that are different from these, offering new information or perspectives.\\\\n"
            f"---Existing Q&A Start---\\\\n"
            f"{existing_qa_str}\\\\n"
            f"---Existing Q&A End---"
        )

    qa_agent = Agent(
        name="JSONL Content QA Extractor",
        instructions=(
            "You are a knowledge extraction assistant.\\\\n"
            f"1. Your primary task is to analyze the provided text content which was extracted from the source: {source_identifier} (likely a life insurance company's webpage).\\\\n"
            f"2. The provided text content is: \\\\\\\\n---TEXT CONTENT BEGIN---\\\\\\\\n{text_content}\\\\\\\\n---TEXT CONTENT END---\\\\\\\\n"
            f"3. {existing_qa_instructions_segment}\\\\\\\\n"
            f"4. From THIS TEXT CONTENT, extract up to {max_q_per_content} new question-answer pairs that would be genuinely helpful for an FAQ, considering the existing Q&A above. Each pair must include:\\\\n"
            "    a. The question (in Japanese).\\\\n"
            "    b. The answer (in Japanese).\\\\n"
            f"   c. The source identifier, which MUST be exactly '{source_identifier}'. (This will be the 'source_url' in the output QAPair object)\\\\n"
            "    d. A brief description of the type of person who would ask this question, considering they are likely viewing a life insurance company's website (e.g., '契約検討中の顧客', '既契約者', '保険金受取人', '就職活動中の学生', '一般的な情報収集者') - this is the 'questioner_persona' (in Japanese).\\\\n"
            "    e. An appropriate 'information_category' for this Q&A (e.g., '契約手続き', '保障内容', '保険金請求', '商品比較', '税金・控除', '健康増進サービス', '会社情報') (in Japanese).\\\\n"
            "    f. A list of 'related_keywords' (3-5 keywords) relevant to this Q&A (in Japanese).\\\\n"
            "5. Avoid duplicate / trivial questions, including those listed in the existing Q&A section if provided.\\\\\\\\n"
            "6. The extracted question, answer, questioner_persona, information_category, and related_keywords MUST be in Japanese. If the source content is in another language, translate them to Japanese.\\\\\\\\n"
            "7. The answer should be self-contained and directly address the question. Avoid answers that primarily redirect the user elsewhere (e.g., 'Please refer to page X', 'Contact customer support for this'). If the provided text only allows for a redirecting answer, then do not generate a Q&A pair for that specific point.\\\\n"
            "Return the result as List[QAPair]. Ensure each QAPair object includes 'question', 'answer', 'source_url', 'questioner_persona', 'information_category', and 'related_keywords'."
        ),
        # WebSearchToolは不要
        output_type=List[QAPair],
        model=model_name,
    )

    # Agentへの入力は、分析対象のテキストコンテンツそのものではなく、
    # Agentのinstructions内でコンテンツを直接扱っているため、ここではsource_identifierを渡します。
    result = await Runner.run(qa_agent, input=f"Generate Q&A for content from {source_identifier}")

    if result.final_output:
        # WebSearch.pyと同様に、source_urlフィールドに出典情報を強制的に設定
        processed_qas = []
        for qa in result.final_output:
            qa_dict = qa.model_dump()
            if qa_dict.get("source_url") != source_identifier:
                # print(f"修正: agentが返したsource_url '{qa_dict.get('source_url')}' を '{source_identifier}' に修正します。")
                qa_dict["source_url"] = source_identifier
            processed_qas.append(QAPair(**qa_dict))
        return processed_qas
    return []

# --- WebSearch.pyのmain処理ロジックを改変 ---
async def process_jsonl_and_generate_qa(
    input_jsonl_path: str,
    outfile: str,
    model_name: str,
    source_id_field: str, # JSONL内の出典IDフィールド名
    content_field: str,   # JSONL内のテキストコンテンツフィールド名
    max_q_per_entry: int = 3,
    max_entries_to_process: int = -1,
    # WebSearch.pyにあったmax_tries_per_urlは、コンテンツが静的なので単純化のため削除
):
    """
    JSONLファイルを処理し、各エントリの指定されたテキストフィールドからQ&Aを生成します。
    WebSearch.pyのメインロジックをベースにしています。
    """
    if not os.path.exists(input_jsonl_path):
        print(f"エラー: 入力ファイル '{input_jsonl_path}' が見つかりません。")
        return

    # --- WebSearch.pyから流用: 既存Q&Aの読み込み ---
    existing_qa_set: Set[Tuple[str, str]] = set()
    if os.path.exists(outfile):
        try:
            with jsonlines.open(outfile, "r") as reader:
                for qa_obj_dict in reader:
                    existing_qa_set.add((qa_obj_dict.get("question"), qa_obj_dict.get("answer")))
            print(f"既存の出力ファイル '{outfile}' から {len(existing_qa_set)} 件のQ&Aを読み込みました。")
        except Exception as e:
            print(f"警告: 既存の出力ファイル '{outfile}' の読み込み中にエラー: {e}")

    total_newly_added_in_session = 0
    processed_entry_count = 0

    # JSONLファイルを読み込むループ (WebSearch.pyのURLリストループの代わり)
    with jsonlines.open(input_jsonl_path, "r") as reader:
        for i, entry in enumerate(reader):
            if max_entries_to_process != -1 and processed_entry_count >= max_entries_to_process:
                print(f"指定された最大処理エントリ数 ({max_entries_to_process}) に達しました。")
                break
            
            print(f"\\n--- JSONLエントリ {i+1} を処理中 ---")

            source_identifier = entry.get(source_id_field)
            text_content = entry.get(content_field)

            if not source_identifier:
                print(f"警告: エントリ {i+1} に '{source_id_field}' が見つからないか空です。スキップします。")
                continue
            if not text_content:
                print(f"警告: エントリ {i+1} に '{content_field}' が見つからないか空です。スキップします。")
                continue

            print(f"ソースID: {source_identifier}")
            # print(f"コンテンツ (先頭100文字): {text_content[:100]}...") # デバッグ用

            # --- WebSearch.pyから流用: 現在のソースに関する既存Q&Aの収集 ---
            existing_qa_for_current_source_display: List[str] = []
            if os.path.exists(outfile): # 再度ファイルを開くのは非効率だがWebSearch.pyの構造に合わせる
                try:
                    with jsonlines.open(outfile, "r") as r:
                        for qa_obj_dict in r:
                            if qa_obj_dict.get("source_url") == source_identifier: # QAPairのsource_urlフィールドを参照
                                q = qa_obj_dict.get("question")
                                a = qa_obj_dict.get("answer")
                                if q and a:
                                    existing_qa_for_current_source_display.append(f"- Q: {q}\\\\n  A: {a}")
                except Exception as e:
                    print(f"警告: 既存の出力ファイル '{outfile}' の読み込み中にエラー (エントリ処理中): {e}")
            
            # --- WebSearch.pyから流用: Q&A生成呼び出し ---
            generated_qas = await generate_qa_for_content(
                source_identifier,
                text_content,
                existing_qa_for_current_source_display,
                model_name,
                max_q_per_entry
            )

            current_entry_added_count = 0
            if generated_qas:
                with jsonlines.open(outfile, "a") as writer: # 'a'モードで追記
                    for qa_pair in generated_qas:
                        current_qa_tuple = (qa_pair.question, qa_pair.answer)
                        if current_qa_tuple not in existing_qa_set:
                            writer.write(qa_pair.model_dump())
                            existing_qa_set.add(current_qa_tuple)
                            total_newly_added_in_session += 1
                            current_entry_added_count += 1
                        else:
                            print(f"フィルタリング(重複): Q: {qa_pair.question}")
            
            if current_entry_added_count > 0:
                print(f"✨ このエントリで {current_entry_added_count} 件を新たに書き出しました。")
            else:
                print("ℹ️ このエントリでは新しいQ&Aは生成されませんでした。")
            
            processed_entry_count += 1
            await asyncio.sleep(0.5) # APIレート制限等を考慮した短い待機 (WebSearch.pyより短くても良いかも)

    print(f"\\n--- 全処理完了 ---")
    print(f"🎉 合計 {total_newly_added_in_session} 件の新しいQ&Aをセッション中に書き出しました → {outfile}")


if __name__ == "__main__":
    # --- WebSearch.pyから流用: argparseの設定を改変 ---
    parser = argparse.ArgumentParser(description="JSONLファイル内のテキストコンテンツからQ&Aペアを生成します。WebSearch.pyのロジックを流用。")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="入力JSONLファイルへのパス (例: /app/aflac_with_body.jsonl)"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="generated_qa_from_jsonl_alt.jsonl",
        help="出力ファイル名 (例: output_qa_alt.jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="使用するOpenAIモデル名 (例: gpt-4o, gpt-4-turbo)"
    )
    parser.add_argument(
        "--source_id_field",
        type=str,
        required=True,
        help="JSONLエントリ内でソース識別子(URL等)が格納されているフィールド名 (例: url, id)"
    )
    parser.add_argument(
        "--content_field",
        type=str,
        required=True,
        help="JSONLエントリ内でQ&A生成の元となるテキストコンテンツが格納されているフィールド名 (例: response_text, body)"
    )
    parser.add_argument(
        "--max_q_per_entry",
        type=int,
        default=2, # WebSearch.pyのmax_q_per_urlに相当
        help="1つのJSONLエントリから生成するQ&Aの最大数 (デフォルト: 2)"
    )
    parser.add_argument(
        "--max_entries",
        type=int,
        default=-1, # デフォルトは全件処理
        help="処理するJSONLエントリの最大数。テスト用 (デフォルト: -1 で全件処理)"
    )
    # WebSearch.pyにあった --urls_file, --max_tries_per_url, --max_urls は不要なので削除

    args = parser.parse_args()

    asyncio.run(process_jsonl_and_generate_qa(
        args.input_jsonl,
        args.outfile,
        args.model,
        args.source_id_field,
        args.content_field,
        args.max_q_per_entry,
        args.max_entries
    ))

"""
コマンドライン実行例:

python /app/Create_QA/src/Create_QA_from_jsonl_alt.py \
    --input_jsonl /app/aflac_tsumitasu_qa.jsonl \
    --outfile /app/Create_QA/output/QA/aflac_tsumitasu_qa_gen_alt.jsonl \
    --model gpt-4o-mini \
    --source_id_field url \
    --content_field answer \
    --max_q_per_entry 1 \
    --max_entries 3

説明:
--input_jsonl: 入力するJSONLファイルへのパス。
--outfile: 生成されたQ&Aを保存するファイルへのパス。
--model: Q&A生成に使用するOpenAIのモデル名。
--source_id_field: 入力JSONL内で、各Q&Aの出典元URLが格納されているフィールド名。
--content_field: 入力JSONL内で、Q&A生成の元となるテキストコンテンツが格納されているフィールド名。
                 (この例では既存の回答 'answer' を元に新しいQ&Aを生成しようとしていますが、
                  通常はより広範なテキストコンテンツを含むフィールドを指定します。例: 'body', 'response_text')
--max_q_per_entry: 1つの入力エントリから生成するQ&Aの最大数。
--max_entries: 処理する入力エントリの最大数 (テスト用などに使用)。-1で全件処理。
"""
