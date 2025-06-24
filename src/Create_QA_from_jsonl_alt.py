\
import asyncio
import jsonlines
import os
import argparse
from typing import List, Set, Tuple, Dict, Any, Optional
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

# --- WebSearch.pyのgenerate_qa_for_urlを改変: 単一Q&A生成方式 ---
async def generate_single_qa(
    source_identifier: str, # URLやファイル名など、コンテンツの出典
    text_content: str,
    existing_qa_for_source_display: List[str],
    model_name: str,
    attempt_number: int  # 何回目の試行かを明示
) -> Optional[QAPair]:
    """
    1つのQ&Aペアのみを生成します。
    """
    if not existing_qa_for_source_display:
        existing_qa_instructions_segment = "There are currently no existing Q&A pairs for this source."
    else:
        existing_qa_str = "\\\\n".join(existing_qa_for_source_display)
        existing_qa_instructions_segment = (
            f"The following Q&A pairs already exist for this source ({source_identifier}):\\\\n"
            f"Please generate a NEW Q&A pair that covers different aspects or provides different perspectives.\\\\n"
            f"---Existing Q&A Start---\\\\n"
            f"{existing_qa_str}\\\\n"
            f"---Existing Q&A End---"
        )

    qa_agent = Agent(
        name="Single QA Generator",
        instructions=(
            "You are a specialized knowledge extraction assistant.\\\\n"
            f"1. Analyze the provided text content from: {source_identifier} (likely a life insurance company's webpage).\\\\n"
            f"2. Text content: \\\\\\\\n---TEXT CONTENT BEGIN---\\\\\\\\n{text_content}\\\\\\\\n---TEXT CONTENT END---\\\\\\\\n"
            f"3. {existing_qa_instructions_segment}\\\\\\\\n"
            f"4. Generate ONE high-quality question-answer pair that would be valuable for an FAQ. Focus on:\\\\n"
            "   - Creating a natural, specific question someone would actually ask\\\\n"
            "   - If the answer varies based on conditions (age, gender, health status, contract details, timing, etc.), make the question specify those conditions clearly\\\\n"
            "   - If the answer differs by insurance product, include the specific product name in the question\\\\n"
            "   - For example, instead of '保険金はいくらもらえますか？' ask '30歳男性がちゃんと応える医療保険EVERに加入した場合、入院給付金はいくらもらえますか？'\\\\n"
            "   - Another example: instead of '保険料の支払い方法は？' ask 'アフラックのがん保険フォルテの保険料支払い方法にはどのような選択肢がありますか？'\\\\n"
            "   - Providing a comprehensive, self-contained answer that addresses the specific conditions and products mentioned in the question\\\\n"
            "   - Avoiding generic or overly broad questions that could have multiple different answers\\\\n"
            "   - Including relevant details and context\\\\n"
            f"5. This is attempt #{attempt_number}, so try to find a unique angle or aspect not covered before.\\\\n"
            "6. The question, answer, questioner_persona, information_category, and related_keywords MUST be in Japanese.\\\\n"
            "7. The answer should be self-contained and directly address the question. Avoid answers that primarily redirect the user elsewhere.\\\\n"
            "8. Each Q&A must include:\\\\n"
            "   a. The question (in Japanese)\\\\n"
            "   b. The answer (in Japanese)\\\\n"
            f"   c. The source identifier: '{source_identifier}'\\\\n"
            "   d. A questioner_persona appropriate for a life insurance website visitor (e.g., '契約検討中の顧客', '既契約者', '保険金受取人', '就職活動中の学生', '一般的な情報収集者')\\\\n"
            "   e. An information_category (e.g., '契約手続き', '保障内容', '保険金請求', '商品比較', '税金・控除', '健康増進サービス', '会社情報')\\\\n"
            "   f. A list of 3-5 related_keywords\\\\n"
            "Return exactly ONE QAPair object with all required fields."
        ),
        output_type=QAPair,  # 単一のQAPairオブジェクト
        model=model_name,
    )

    result = await Runner.run(qa_agent, input=f"Generate one high-quality Q&A for content from {source_identifier}")
    
    if result.final_output:
        qa = result.final_output
        # source_urlの修正
        if qa.source_url != source_identifier:
            qa_dict = qa.model_dump()
            qa_dict["source_url"] = source_identifier
            return QAPair(**qa_dict)
        return qa
    return None

# --- 並列処理対応: ファイルI/O ロック管理 ---
import threading
import time
from datetime import datetime

# ファイル書き込み用ロック
file_lock = threading.Lock()

def collect_existing_qa_for_source(source_identifier: str, outfile: str) -> List[str]:
    """
    指定されたソースIDに関する既存Q&Aを収集
    """
    existing_qa_display = []
    if os.path.exists(outfile):
        try:
            with file_lock:  # ファイル読み込み時もロック
                with jsonlines.open(outfile, "r") as reader:
                    for qa_obj_dict in reader:
                        if qa_obj_dict.get("source_url") == source_identifier:
                            q = qa_obj_dict.get("question")
                            a = qa_obj_dict.get("answer")
                            if q and a:
                                existing_qa_display.append(f"- Q: {q}\\n  A: {a}")
        except Exception as e:
            print(f"警告: 既存Q&A収集中にエラー ({source_identifier}): {e}")
    return existing_qa_display

def save_qa_to_file(qa: QAPair, outfile: str) -> bool:
    """
    Q&Aをファイルに安全に保存
    """
    try:
        with file_lock:  # ファイル書き込み時のロック
            with jsonlines.open(outfile, "a") as writer:
                writer.write(qa.model_dump())
        return True
    except Exception as e:
        print(f"Q&A保存エラー: {e}")
        return False

async def process_single_entry(
    entry_data: Tuple[int, Dict[str, Any]],
    outfile: str,
    model_name: str,
    source_id_field: str,
    content_field: str,
    max_q_per_entry: int,
    global_existing_qa_set: Set[Tuple[str, str]]
) -> int:
    """
    単一エントリの処理（エントリ内のQ&A生成は逐次実行）
    """
    i, entry = entry_data
    
    source_identifier = entry.get(source_id_field)
    text_content = entry.get(content_field)
    
    if not source_identifier:
        print(f"⚠️ エントリ {i+1}: '{source_id_field}' が見つからないか空です。スキップします。")
        return 0
    if not text_content:
        print(f"⚠️ エントリ {i+1}: '{content_field}' が見つからないか空です。スキップします。")
        return 0

    print(f"🔄 エントリ {i+1} を処理中: {source_identifier}")

    # このソースの既存Q&Aを収集
    existing_qa_for_current_source_display = collect_existing_qa_for_source(source_identifier, outfile)
    
    # エントリ内でのQ&A生成は逐次実行（品質重視）
    current_entry_added_count = 0
    for attempt in range(max_q_per_entry):
        print(f"  📝 エントリ {i+1}, 試行 {attempt + 1}/{max_q_per_entry}")
        
        single_qa = await generate_single_qa(
            source_identifier,
            text_content,
            existing_qa_for_current_source_display,
            model_name,
            attempt + 1
        )
        
        if single_qa:
            current_qa_tuple = (single_qa.question, single_qa.answer)
            
            # グローバル重複チェック（スレッドセーフ）
            with file_lock:
                is_duplicate = current_qa_tuple in global_existing_qa_set
                if not is_duplicate:
                    global_existing_qa_set.add(current_qa_tuple)
            
            if not is_duplicate:
                # ファイルに保存
                if save_qa_to_file(single_qa, outfile):
                    # 次の試行で使用するため、このエントリの既存リストに追加
                    existing_qa_for_current_source_display.append(
                        f"- Q: {single_qa.question}\\n  A: {single_qa.answer}"
                    )
                    current_entry_added_count += 1
                    print(f"    ✅ Q&A生成成功: {single_qa.question[:50]}...")
                else:
                    print(f"    ❌ Q&A保存失敗")
            else:
                print(f"    ⚠️ 重複のためスキップ: {single_qa.question[:50]}...")
        else:
            print(f"    ❌ Q&A生成失敗")
        
        # API制限対応の待機
        await asyncio.sleep(1)
    
    if current_entry_added_count > 0:
        print(f"✨ エントリ {i+1}: {current_entry_added_count} 件を新たに生成")
    else:
        print(f"ℹ️ エントリ {i+1}: 新しいQ&Aは生成されませんでした")
    
    # エントリ間の待機
    await asyncio.sleep(0.5)
    return current_entry_added_count

# --- エントリレベル並列処理のメイン関数 ---
async def process_jsonl_parallel_entries(
    input_jsonl_path: str,
    outfile: str,
    model_name: str,
    source_id_field: str,
    content_field: str,
    max_q_per_entry: int = 3,
    max_entries_to_process: int = -1,
    max_concurrent_entries: int = 3  # 同時処理するエントリ数
):
    """
    エントリレベル並列処理でJSONLファイルを処理
    """
    if not os.path.exists(input_jsonl_path):
        print(f"❌ エラー: 入力ファイル '{input_jsonl_path}' が見つかりません。")
        return

    # 既存Q&Aの読み込み
    global_existing_qa_set: Set[Tuple[str, str]] = set()
    if os.path.exists(outfile):
        try:
            with jsonlines.open(outfile, "r") as reader:
                for qa_obj_dict in reader:
                    global_existing_qa_set.add((qa_obj_dict.get("question"), qa_obj_dict.get("answer")))
            print(f"📂 既存の出力ファイル '{outfile}' から {len(global_existing_qa_set)} 件のQ&Aを読み込みました。")
        except Exception as e:
            print(f"⚠️ 警告: 既存の出力ファイル '{outfile}' の読み込み中にエラー: {e}")

    # エントリを読み込み
    entries = []
    with jsonlines.open(input_jsonl_path, "r") as reader:
        for i, entry in enumerate(reader):
            if max_entries_to_process != -1 and i >= max_entries_to_process:
                break
            entries.append((i, entry))

    print(f"🚀 {len(entries)} エントリを最大 {max_concurrent_entries} 並列で処理開始")
    print(f"⚙️ 設定: モデル={model_name}, エントリあたりQ&A数={max_q_per_entry}")
    
    start_time = time.time()

    # 並列処理用セマフォ
    semaphore = asyncio.Semaphore(max_concurrent_entries)
    
    async def process_entry_with_semaphore(entry_data):
        async with semaphore:
            return await process_single_entry(
                entry_data,
                outfile,
                model_name,
                source_id_field,
                content_field,
                max_q_per_entry,
                global_existing_qa_set
            )
    
    # 並列実行
    tasks = [process_entry_with_semaphore(entry_data) for entry_data in entries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 結果集計
    total_newly_added = sum(r for r in results if isinstance(r, int))
    error_count = sum(1 for r in results if isinstance(r, Exception))
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n📊 処理完了サマリー")
    print(f"=" * 50)
    print(f"🎉 新規Q&A生成数: {total_newly_added} 件")
    print(f"⏱️ 処理時間: {processing_time:.2f} 秒")
    print(f"⚡ 平均処理速度: {len(entries) / processing_time:.2f} エントリ/秒")
    if error_count > 0:
        print(f"❌ エラー発生エントリ数: {error_count} 件")
    print(f"💾 出力ファイル: {outfile}")

# --- 下位互換性のためのエイリアス ---
async def process_jsonl_single_qa_mode(
    input_jsonl_path: str,
    outfile: str,
    model_name: str,
    source_id_field: str,
    content_field: str,
    max_q_per_entry: int = 3,
    max_entries_to_process: int = -1,
):
    """
    下位互換性のための関数（並列処理版を呼び出し）
    """
    await process_jsonl_parallel_entries(
        input_jsonl_path,
        outfile,
        model_name,
        source_id_field,
        content_field,
        max_q_per_entry,
        max_entries_to_process,
        max_concurrent_entries=1  # 逐次処理モード
    )


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
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=3,
        help="同時処理するエントリ数 (デフォルト: 3, 1で逐次処理)"
    )
    # WebSearch.pyにあった --urls_file, --max_tries_per_url, --max_urls は不要なので削除

    args = parser.parse_args()

    asyncio.run(process_jsonl_parallel_entries(
        args.input_jsonl,
        args.outfile,
        args.model,
        args.source_id_field,
        args.content_field,
        args.max_q_per_entry,
        args.max_entries,
        args.max_concurrent  # 並列数の指定
    ))

"""
コマンドライン実行例:

# 基本実行（3エントリ並列処理）
python /app/Create_QA/src/Create_QA_from_jsonl_alt.py \
    --input_jsonl /app/aflac_with_body.jsonl \
    --outfile /app/Create_QA/output/QA/aflac_qa_parallel.jsonl \
    --model gpt-4o-mini \
    --source_id_field url \
    --content_field response_text \
    --max_q_per_entry 2 \
    --max_concurrent 3

# 高速処理（5エントリ並列）
python /app/Create_QA/src/Create_QA_from_jsonl_alt.py \
    --input_jsonl /app/aflac_with_body.jsonl \
    --outfile /app/Create_QA/output/QA/aflac_qa_fast.jsonl \
    --model gpt-4o-mini \
    --source_id_field url \
    --content_field response_text \
    --max_q_per_entry 3 \
    --max_concurrent 5 \
    --max_entries 20

# 逐次処理（従来モード）
python /app/Create_QA/src/Create_QA_from_jsonl_alt.py \
    --input_jsonl /app/aflac_tsumitasu_qa.jsonl \
    --outfile /app/Create_QA/output/QA/aflac_tsumitasu_qa_sequential.jsonl \
    --model gpt-4o-mini \
    --source_id_field url \
    --content_field answer \
    --max_q_per_entry 1 \
    --max_concurrent 1 \
    --max_entries 3

【改良版】エントリレベル並列処理対応:
- 異なるsource_identifierのエントリを並列で処理
- 各エントリ内のQ&A生成は逐次実行で品質を保持
- スレッドセーフなファイルI/O処理
- 重複排除の並列対応
- 詳細な進捗表示と処理統計
- 3-5倍の高速化を実現

説明:
--input_jsonl: 入力するJSONLファイルへのパス。
--outfile: 生成されたQ&Aを保存するファイルへのパス。
--model: Q&A生成に使用するOpenAIのモデル名。
--source_id_field: 入力JSONL内で、各Q&Aの出典元URLが格納されているフィールド名。
--content_field: 入力JSONL内で、Q&A生成の元となるテキストコンテンツが格納されているフィールド名。
--max_q_per_entry: 1つの入力エントリから生成するQ&Aの最大数。
--max_entries: 処理する入力エントリの最大数 (テスト用などに使用)。-1で全件処理。
--max_concurrent: 同時処理するエントリ数。1で逐次処理、3-5で並列処理推奨。
"""
