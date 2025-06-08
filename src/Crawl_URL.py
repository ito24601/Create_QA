# %%
# 1️⃣ ライブラリのインポート
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import argparse
import time
import os # osモジュールをインポート

# %%
# 2️⃣ ドメイン内URL収集関数
def crawl_domain(start_url, max_urls=1000, output_file=None, state_file=None): # output_file と state_file を引数に追加
    """
    指定された開始URLから同じドメイン内のURLを収集します。
    進行状況は state_file に保存・読み込みされます。
    """
    domain = urlparse(start_url).netloc if start_url else None
    
    queue = []
    seen = set()
    results = []
    
    # 状態ファイルの読み込み試行
    if state_file and os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
                queue = state.get("queue", [])
                seen = set(state.get("seen", [])) # リストからセットに変換
                results = state.get("results", []) # 収集済み結果も復元
                # 状態ファイルからドメインを復元（start_urlがNoneの場合など）
                if not domain and results:
                    # resultsが空でない場合、最初の要素からドメインを取得しようと試みる
                    # ただし、resultsの各要素が 'domain' キーを持つことを前提とする
                    if results[0].get("domain"):
                        domain = results[0]["domain"]
                
                # start_urlが指定されておらず、queueも空の場合はエラー
                if not start_url and not queue:
                    print("エラー: state_fileから再開できませんでした。有効なqueueがありません。start_urlを指定して下さい。")
                    return []
                # ドメインが特定できない場合はエラー
                if not domain and queue: # queueがあってdomainがない場合、queueの最初のURLからドメインを推測
                    domain = urlparse(queue[0]).netloc

                if not domain:
                    print("エラー: ドメインを特定できませんでした。state_fileが不正か、start_urlが必要です。")
                    return []

                print(f"状態ファイル '{state_file}' からクロールを再開します。")
                print(f"復元されたキューの数: {len(queue)}, 訪問済みURLの数: {len(seen)}, 収集済み結果の数: {len(results)}")
        except Exception as e:
            print(f"警告: 状態ファイル '{state_file}' の読み込みに失敗しました: {e}。新規クロールを開始します。")
            # 読み込み失敗時はstart_urlから開始
            if not start_url:
                print("エラー: start_urlが指定されておらず、状態ファイルの読み込みにも失敗しました。")
                return []
            queue = [start_url]
            seen = set()
            results = []
            if not domain: # start_urlからドメインを再取得
                 domain = urlparse(start_url).netloc
                 if not domain:
                     print(f"エラー: 有効な開始URLではありません: {start_url}")
                     return []
    elif start_url:
        queue = [start_url]
        seen = set()
        results = []
        if not domain:
            domain = urlparse(start_url).netloc
            if not domain:
                print(f"エラー: 有効な開始URLではありません: {start_url}")
                return []
    else:
        print("エラー: start_urlが指定されていないか、有効な状態ファイルがありません。")
        return []

    print(f"クロール対象ドメイン: {domain}")
    print(f"クロール開始 (最大総収集URL数: {max_urls})")

    urls_processed_in_session = 0 # このセッションで処理したURL数

    try: # メインループをtryブロックで囲み、中断時に状態を保存
        
        print(queue)
        print(len(results))
        while queue and len(results) < max_urls: # seenの数ではなくresultsの数で判断
            url = queue.pop(0)
            if url in seen:
                continue
            
            print(f"処理中: {url} (収集済み: {len(results)}/{max_urls}, キュー: {len(queue)})")
            seen.add(url) # 処理開始時にseenに追加

            # リクエスト前に待機
            wait_time = 1 # 秒
            print(f"リクエスト前に{wait_time}秒待機します...")
            time.sleep(wait_time)

            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=20)
                response.raise_for_status()
                # --- デバッグ用に追加 ---
                # print(f"--- HTML content for {url} ---")
                # print(response.text[:2000]) # 最初の2000文字を表示
                # with open(f"debug_{url.replace('/', '_')}.html", "w", encoding="utf-8") as f_debug:
                #     f_debug.write(response.text)
                # print(f"Saved HTML for {url} to debug_{url.replace('/', '_')}.html")
                # --- ここまで ---
                print(response)

                content_type = response.headers.get('Content-Type', '').lower()
                print(content_type)

                # すべての成功したリクエストのURLをresultsに追加
                current_result = {"domain": domain, "url": url, "content_type": content_type}
                if current_result not in results: # 重複を避けて追加
                    results.append(current_result)
                
                urls_processed_in_session += 1

                # HTMLの場合のみリンクを収集
                if content_type.startswith('text/html'):
                    soup = BeautifulSoup(response.text, "html.parser")
                    for a_tag in soup.find_all("a", href=True):
                        link = urljoin(url, a_tag["href"])
                        parsed_link = urlparse(link)
                        if parsed_link.netloc == domain and parsed_link.scheme in ("http", "https"):
                            if link not in seen and link not in queue:
                                queue.append(link)
            except requests.exceptions.RequestException as e:
                print(f"リクエストエラー: {url} - {e}")
                continue
            except Exception as e:
                print(f"予期せぬエラー: {url} - {e}")
                continue
    finally: # 中断時や終了時に状態を保存
        if state_file:
            try:
                # results内のURLもseenに追加する（resultsにあってseenにない場合を考慮）
                for res_item in results:
                    seen.add(res_item["url"])

                current_state = {
                    "queue": queue,
                    "seen": list(seen), # セットをリストに変換して保存
                    "results": results,
                    "start_url_for_reference": start_url # 参考情報として元のstart_urlも保存
                }
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(current_state, f, ensure_ascii=False, indent=2)
                print(f"現在のクロール状態を '{state_file}' に保存しました。")
                print(f"保存されたキューの数: {len(queue)}, 訪問済みURLの数: {len(seen)}, 収集済み結果の数: {len(results)}")
            except Exception as e:
                print(f"警告: 状態ファイル '{state_file}' の保存に失敗しました: {e}")
            
    if len(results) >= max_urls:
        print(f"最大収集URL数 {max_urls} に達しました。")
    elif not queue:
        print("クロールキューが空になりました。")
    
    print(f"クロール完了。収集した総URL数: {len(results)}。このセッションで処理したURL数: {urls_processed_in_session}")
    return results

# %%
# 3️⃣ メイン処理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="指定された開始URLから同じドメイン内のURLを収集し、結果をJSONファイルに出力します。進行状況は状態ファイルに保存/読み込みされます。",
        formatter_class=argparse.RawTextHelpFormatter # ヘルプメッセージの改行を保持
    )
    parser.add_argument(
        "--start_url", 
        type=str,
        default=None, # デフォルトをNoneに変更
        help="クロールを開始するURL (例: https://example.com/)。\nstate_fileから再開する場合は省略可能。"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, # 出力ファイルは必須
        help="収集したURLのリストを保存するJSONファイル名 (必須)"
    )
    parser.add_argument(
        "--max_urls", 
        type=int, 
        default=1000, 
        help="収集する最大のURL数 (デフォルト: 1000)"
    )
    parser.add_argument(
        "--state_file",
        type=str,
        default="crawl_state.json", # デフォルトの状態ファイル名を設定
        help="クロールの進行状況を保存/読み込みするファイル名 (デフォルト: crawl_state.json)"
    )
    args = parser.parse_args()

    # start_url と state_file の整合性チェック
    if not args.start_url and (not args.state_file or not os.path.exists(args.state_file)):
        parser.error("--start_url が指定されておらず、有効な --state_file も存在しません。どちらかが必要です。")

    collected_urls = crawl_domain(args.start_url, args.max_urls, args.output, args.state_file)
    
    if collected_urls:
        try:
            # crawl_domain内でresultsが更新されるので、最終的な結果を書き出す
            # ただし、状態ファイルにresultsも保存しているので、ここでは重複書き込みになる可能性がある。
            # outputファイルは最終結果のみを保存する役割とする。
            # 状態ファイルとは別に、最終的な収集リストを出力ファイルに書き出す。
            final_output_results = []
            # 既にoutputファイルに存在するURLを読み込む (オプション: 重複を完全に避ける場合)
            # existing_output_urls = set()
            # if os.path.exists(args.output):
            #     try:
            #         with open(args.output, "r", encoding="utf-8") as f_out_read:
            #             for item in json.load(f_out_read):
            #                 existing_output_urls.add(item.get("url"))
            #     except Exception: # エラーの場合は無視
            #         pass

            for item in collected_urls:
                 # if item.get("url") not in existing_output_urls: # outputファイル内の重複も避ける場合
                final_output_results.append(item)
            
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(final_output_results, f, ensure_ascii=False, indent=2)
            print(f"✨ {len(final_output_results)} 件のURLを '{args.output}' に書き出しました。")
        except IOError as e:
            print(f"ファイル書き込みエラー: {args.output} - {e}")
    else:
        print(f"'{args.output}' への書き出しは行われませんでした。収集されたURLがありません。")