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
async def collect_qa(target_url: str, outfile: str, model_name: str): # domain を target_url に変更
    # search_domain = extract_search_domain(target_url) # 単一URL指定のため、ドメイン抽出は指示やフィルタリングに直接使わない
    if not target_url: # target_url が空かチェック
        print(f"エラー: 入力 URL が指定されていません。処理を中止します。")
        return

    existing_qa_set = set()
    if os.path.exists(outfile):
        try:
            with jsonlines.open(outfile, "r") as reader:
                for qa_obj in reader:
                    # question と answer のタプルをセットに追加して重複チェックに利用
                    existing_qa_set.add((qa_obj.get("question"), qa_obj.get("answer")))
        except Exception as e:
            print(f"警告: 既存の出力ファイル '{outfile}' の読み込み中にエラーが発生しました: {e}")


    qa_agent = Agent(
        name        = "Web QA Collector",
        instructions=(
            "You are a knowledge extraction assistant.\\\\n"
            f"1. Your primary task is to analyze the content of a single, specific web page: {target_url}. Use the WebSearchTool for this purpose. Do NOT navigate away from this URL. Do NOT follow any links on the page. All information must come strictly from the content of {target_url}.\\\\n"
            f"2. Read and understand the content of the page at {target_url}.\\\\n"
            f"3. From this single page ({target_url}), extract up to 3 question-answer pairs that would be genuinely helpful for an FAQ. Each pair must include the source URL, and this source URL MUST be exactly '{target_url}'.\\\\n"
            "4. Avoid duplicate / trivial questions.\\\\n"
            "5. The extracted question and answer MUST be in Japanese. If the source content is in another language, translate them to Japanese.\\\\n"  # 日本語での出力を指示
            "Return the result as List[QAPair]."
        ),
        tools       = [WebSearchTool(search_context_size="high")],
        output_type = List[QAPair],      # ← これが返るまで自動的にループ
        model       = model_name
    )
    # site 検索ではなく、直接 target_url をエージェントの入力とする
    result = await Runner.run(qa_agent, input=target_url)

    newly_added_count = 0
    filtered_output = []
    processed_in_current_run = set() # 今回の実行で処理済みのQ&Aを保持するセット

    if result.final_output:
        for qa in result.final_output:
            if qa and qa.source_url: # qaオブジェクトとsource_urlが存在することを確認
                # qa_source_hostname = extract_search_domain(qa.source_url) # ドメイン単位のチェックからURL完全一致に変更
                if qa.source_url == target_url: # 参照元URLが指定されたURLと完全に一致するか確認
                    # 現在の実行での重複チェックと、既存ファイルとの重複チェック
                    current_qa_tuple = (qa.question, qa.answer)
                    if current_qa_tuple not in existing_qa_set and current_qa_tuple not in processed_in_current_run:
                        filtered_output.append(qa)
                        processed_in_current_run.add(current_qa_tuple) # 今回処理したQ&Aとして追加
                    else:
                        print(f"フィルタリング: 重複のためQ&Aを除外しました: {qa.question}")
                else:
                    print(f"フィルタリング: Q&Aの参照元URLが指定されたURLと異なります: {qa.source_url} (期待URL: {target_url})")
            elif qa:
                 print(f"フィルタリング: Q&Aにsource_urlがありません: {qa}")
            # else: qaがNoneの場合は何もしない

    if filtered_output:
        with jsonlines.open(outfile, "a") as writer: # "w" から "a" (追記モード) に変更
            for qa_pair in filtered_output: # フィルタリングされたリストを使用
                writer.write(qa_pair.model_dump())
        newly_added_count = len(filtered_output)
            
    print(f"✨ {newly_added_count} 件を新たに書き出しました → {outfile}")
    
    original_count = len(result.final_output) if result.final_output else 0
    url_filtered_count = 0 # domain_filtered_count を url_filtered_count に変更
    if result.final_output:
        for qa in result.final_output:
            # if qa and qa.source_url and extract_search_domain(qa.source_url) != search_domain: # 旧ドメインフィルター
            if qa and qa.source_url and qa.source_url != target_url: # 新URLフィルター
                url_filtered_count +=1
            elif not qa or not qa.source_url: # source_urlがないものも除外対象としてカウント
                url_filtered_count +=1


    duplicate_count = original_count - url_filtered_count - newly_added_count
    
    if url_filtered_count > 0:
        print(f"ℹ️  {url_filtered_count} 件のQ&Aが指定URL外、または不正なため除外されました。")
    if duplicate_count > 0:
        print(f"ℹ️  {duplicate_count} 件のQ&Aが重複のため除外されました。")

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
    args = parser.parse_args()

    asyncio.run(collect_qa(args.url, args.outfile, args.model)) # args.domain を args.url に変更