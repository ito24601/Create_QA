# %%
from agents import Agent, Runner, WebSearchTool
from pydantic import BaseModel
from typing import List
import jsonlines, asyncio, os
import argparse # argparse をインポート
from dotenv import load_dotenv
from urllib.parse import urlparse # urllib.parseをインポート

load_dotenv("/app/.env", override=True)        # OPENAI_API_KEY を読み込む

# %%
# 1️⃣  出力フォーマット
class QAPair(BaseModel):
    question: str
    answer: str
    source_url: str # 参照元URLを格納するフィールドを追加

def extract_search_domain(domain_str: str) -> str | None:
    """
    ドメイン文字列から検索用のクリーンなホスト名を抽出します。
    例: "https://example.com/path" -> "example.com"
    """
    if not domain_str:
        return None
    
    temp_domain_str = domain_str
    if '://' not in temp_domain_str:
        temp_domain_str = 'http://' + temp_domain_str # urlparseがホスト名を正しく解釈できるようにスキームを追加
    
    parsed = urlparse(temp_domain_str)
    return parsed.hostname

# 3️⃣  Runner で実行するユーティリティ関数
async def collect_qa(target_url: str, outfile: str, model_name: str, max_attempts: int = 5): # domain を target_url に変更、max_attempts を追加
    # search_domain = extract_search_domain(target_url) # 単一URL指定のため、ドメイン抽出は指示やフィルタリングに直接使わない
    if not target_url: # target_url が空かチェック
        print(f"エラー: 入力 URL が指定されていません。処理を中止します。")
        return

    total_newly_added_in_session = 0
    attempt_count = 0

    while attempt_count < max_attempts:
        attempt_count += 1
        print(f"\\n--- 試行回数: {attempt_count}/{max_attempts} ---")

        existing_qa_set = set()
        existing_qa_for_target_url_display = [] # エージェントへの指示に含めるための既存Q&Aリスト
        if os.path.exists(outfile):
            try:
                with jsonlines.open(outfile, "r") as reader:
                    for qa_obj_dict in reader:
                        # question と answer のタプルをセットに追加して重複チェックに利用
                        existing_qa_set.add((qa_obj_dict.get("question"), qa_obj_dict.get("answer")))
                        # 現在のtarget_urlに関連する既存Q&Aを収集
                        if qa_obj_dict.get("source_url") == target_url:
                            q = qa_obj_dict.get("question")
                            a = qa_obj_dict.get("answer")
                            if q and a: # 質問と回答が両方存在する場合のみ
                                existing_qa_for_target_url_display.append(f"- Q: {q}\\\\n  A: {a}")
            except Exception as e:
                print(f"警告: 既存の出力ファイル '{outfile}' の読み込み中にエラーが発生しました: {e}")

        existing_qa_instructions_segment = "現在、このURLに関する既存のQ&Aはありません。"
        if existing_qa_for_target_url_display:
            existing_qa_str = "\\\\n".join(existing_qa_for_target_url_display)
            existing_qa_instructions_segment = (
                f"以下のQ&Aペアは、このURL ({target_url}) に関して既に存在します。\\\\\\\\n"
                f"これらとは異なる、新しい情報や視点からのQ&Aペアを生成してください。\\\\\\\\n"
                f"---既存のQ&Aここから---\\\\\\\\n"
                f"{existing_qa_str}\\\\\\\\n"
                f"---既存のQ&Aここまで---"
            )

        qa_agent = Agent(
            name        = "Web QA Collector",
            instructions=(
                "You are a knowledge extraction assistant.\\\\\\\\n"
                f"1. Your primary task is to analyze the content of a single, specific web page: {target_url}. Use the WebSearchTool for this purpose. Do NOT navigate away from this URL. Do NOT follow any links on the page. All information must come strictly from the content of {target_url}.\\\\\\\\n"
                f"2. Read and understand the content of the page at {target_url}.\\\\\\\\n"
                f"3. {existing_qa_instructions_segment}\\\\\\\\n" # 既存Q&A情報を指示に追加
                f"4. From this single page ({target_url}), extract up to 3 new question-answer pairs that would be genuinely helpful for an FAQ, considering the existing Q&A above. Each pair must include the source URL, and this source URL MUST be exactly '{target_url}'.\\\\\\\\n"
                "5. Avoid duplicate / trivial questions, including those listed in the existing Q&A section if provided.\\\\\\\\n"
                "6. The extracted question and answer MUST be in Japanese. If the source content is in another language, translate them to Japanese.\\\\\\\\n"
                "Return the result as List[QAPair]."
            ),
            tools       = [WebSearchTool(search_context_size="high")],
            output_type = List[QAPair],      # ← これが返るまで自動的にループ
            model       = model_name
        )
        # site 検索ではなく、直接 target_url をエージェントの入力とする
        result = await Runner.run(qa_agent, input=target_url)

        current_run_added_count = 0
        filtered_output_this_attempt = []
        processed_in_current_run_this_attempt = set() # 今回の実行の試行で処理済みのQ&Aを保持するセット


        if result.final_output:
            for qa in result.final_output:
                if qa and qa.source_url: # qaオブジェクトとsource_urlが存在することを確認
                    # qa_source_hostname = extract_search_domain(qa.source_url) # ドメイン単位のチェックからURL完全一致に変更
                    if qa.source_url == target_url: # 参照元URLが指定されたURLと完全に一致するか確認
                        # 現在の実行での重複チェックと、既存ファイルとの重複チェック
                        current_qa_tuple = (qa.question, qa.answer)
                        if current_qa_tuple not in existing_qa_set and current_qa_tuple not in processed_in_current_run_this_attempt:
                            filtered_output_this_attempt.append(qa)
                            processed_in_current_run_this_attempt.add(current_qa_tuple) # 今回処理したQ&Aとして追加
                        else:
                            print(f"フィルタリング(重複): {qa.question}")
                    else:
                        print(f"フィルタリング(URL不一致): {qa.source_url} (期待: {target_url})")
                elif qa:
                     print(f"フィルタリング(source_urlなし): {qa}")
                # else: qaがNoneの場合は何もしない
        
        if filtered_output_this_attempt:
            with jsonlines.open(outfile, "a") as writer: # "w" から "a" (追記モード) に変更
                for qa_pair in filtered_output_this_attempt: # フィルタリングされたリストを使用
                    writer.write(qa_pair.model_dump())
            current_run_added_count = len(filtered_output_this_attempt)
            total_newly_added_in_session += current_run_added_count
            print(f"✨ この試行で {current_run_added_count} 件を新たに書き出しました。")
        else:
            print("ℹ️ この試行では新しいQ&Aは生成されませんでした。")
            # 新しいQ&Aがなければループを終了
            break
            
        # 短い待機時間を入れる (APIレート制限対策や、エージェントが同じ結果を返し続けるのを避けるため)
        await asyncio.sleep(5) # 5秒待機

    print(f"\\n--- 全試行完了 ---")
    print(f"🎉 合計 {total_newly_added_in_session} 件の新しいQ&Aをセッション中に書き出しました → {outfile}")
    
    # 元の統計情報表示部分はループの外に移動、またはループ内の情報を集約して表示するように変更が必要
    # ここでは簡略化のため、ループ後の総括的な表示のみとする

# %%
# 4️⃣  実行
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指定された単一のWebページからQ&Aペアを収集します。") # 説明を更新
    parser.add_argument(
        "--url", # domain を url に変更
        type=str,
        required=True, # URLは必須とする
        help="検索対象の完全なURL (例: https://example.com/mypage)" # ヘルプテキストを更新
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="python_docs_faq.jsonl",
        help="出力ファイル名 (例: output.jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="使用するモデル名 (例: gpt-4o-mini, gpt-4.1)"
    )
    parser.add_argument(
        "--max_attempts", # コマンドライン引数で最大試行回数を指定できるようにする
        type=int,
        default=5, # デフォルトの最大試行回数
        help="新しいQ&Aを生成するための最大試行回数 (デフォルト: 5)"
    )
    args = parser.parse_args()

    asyncio.run(collect_qa(args.url, args.outfile, args.model, args.max_attempts)) # args.domain を args.url に変更, max_attempts を追加