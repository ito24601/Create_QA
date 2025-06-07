# %%
from agents import Agent, Runner, WebSearchTool
from pydantic import BaseModel
from typing import List
import jsonlines, asyncio, os
import argparse # argparse をインポート
from dotenv import load_dotenv

load_dotenv("/app/.env", override=True)        # OPENAI_API_KEY を読み込む

# %%
# 1️⃣  出力フォーマット
class QAPair(BaseModel):
    question: str
    answer: str

# 2️⃣  WebSearchTool だけを持つエージェント
# qa_agent の定義を collect_qa 関数内に移動します。

# 3️⃣  Runner で実行するユーティリティ関数
async def collect_qa(domain: str, outfile: str, model_name: str):
    qa_agent = Agent(
        name        = "Web QA Collector",
        instructions=(
            "You are a knowledge extraction assistant.\\n"
            "1. Use the WebSearchTool to search ONLY the specified domain.\\n"
            "2. Open promising links and read the page.\\n"
            "3. From each page, extract up to 3 question-answer pairs that would "
            "be genuinely helpful for an FAQ.\\n"
            "4. Avoid duplicate / trivial questions.\\n"
            "Return the result as List[QAPair]."
        ),
        tools       = [WebSearchTool(search_context_size="high")],
        output_type = List[QAPair],      # ← これが返るまで自動的にループ
        model       = model_name
    )
    # site 検索に絞るとクロール範囲を自然に制限できる
    query = f"site:{domain}"
    result = await Runner.run(qa_agent, input=query)
    with jsonlines.open(outfile, "w") as writer:
        for qa in result.final_output:
            writer.write(qa.model_dump())
    print(f"✨ {len(result.final_output)} 件を書き出しました → {outfile}")

# %%
# 4️⃣  実行
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebページからQ&Aペアを収集します。")
    parser.add_argument(
        "--domain",
        type=str,
        default="docs.python.org",
        help="検索対象のドメイン名 (例: example.com)"
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
        default="gpt-4.1-nano",
        help="使用するモデル名 (例: gpt-4o-mini, gpt-4.1)"
    )
    args = parser.parse_args()

    asyncio.run(collect_qa(args.domain, args.outfile, args.model))