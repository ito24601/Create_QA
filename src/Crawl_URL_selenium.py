# %%
# 1️⃣ ライブラリのインポート
import argparse
import json
import time
import os
from urllib.parse import urlparse, urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import ChromiumOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException

# %%
# 2️⃣ ドメイン内URL収集関数
def crawl_domain(start_url, max_urls=1000, output_file=None, state_file=None):
    """
    Seleniumを使って指定された開始URLから同じドメイン内のURLを収集します。
    進行状況は state_file に保存・読み込みされます。
    """
    domain = urlparse(start_url).netloc if start_url else None
    queue = []
    seen = set()
    results = []

    # ドライバ設定（ヘッドレス）
    options = ChromiumOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Remote(command_executor='http://selenium:4444/wd/hub',
                            options=options)

    # 状態ファイルの読み込み
    if state_file and os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            queue = state.get('queue', [])
            seen = set(state.get('seen', []))
            results = state.get('results', [])
            if not domain and results:
                if results[0].get('domain'):
                    domain = results[0]['domain']
            if not start_url and not queue:
                print('エラー: state_fileから再開できませんでした。有効なqueueがありません。start_urlを指定して下さい。')
                driver.quit()
                return []
            if not domain and queue:
                domain = urlparse(queue[0]).netloc
            if not domain:
                print('エラー: ドメインを特定できませんでした。')
                driver.quit()
                return []
            print(f"状態ファイル '{state_file}' からクロールを再開します。")
            print(f"復元されたキューの数: {len(queue)}, 訪問済みURLの数: {len(seen)}, 収集済み結果の数: {len(results)}")
        except Exception as e:
            print(f"警告: 状態ファイル '{state_file}' の読み込みに失敗しました: {e}。新規クロールを開始します。")
            if not start_url:
                print('エラー: start_urlが指定されておらず、状態ファイルの読み込みにも失敗しました。')
                driver.quit()
                return []
            queue = [start_url]
            seen = set()
            results = []
            domain = urlparse(start_url).netloc
    elif start_url:
        queue = [start_url]
        seen = set()
        results = []
        if not domain:
            print(f"エラー: 有効な開始URLではありません: {start_url}")
            driver.quit()
            return []
    else:
        print('エラー: start_urlが指定されていないか、有効な状態ファイルがありません。')
        driver.quit()
        return []

    print(f"クロール対象ドメイン: {domain}")
    print(f"クロール開始 (最大総収集URL数: {max_urls})")

    urls_processed_in_session = 0
    try:
        while queue and len(results) < max_urls:
            url = queue.pop(0)
            if url in seen:
                continue
            print(f"処理中: {url} (収集済み: {len(results)}/{max_urls}, キュー: {len(queue)})")
            seen.add(url)

            # ページ読み込み
            try:
                driver.get(url)
            except WebDriverException as e:
                print(f"WebDriverエラー: {url} - {e}")
                continue

            # 明示的待機: 最低1つリンクが出現するまで最大10秒
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'a'))
                )
            except TimeoutException:
                print(f"タイムアウト: リンク要素が見つかりませんでした - {url}")

            # 無限スクロール例 (1回)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            # スクロール後のリンク待機 (最大5秒)
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'a'))
                )
            except TimeoutException:
                pass

            # URL結果に追加
            current = {'domain': domain, 'url': url}
            if current not in results:
                results.append(current)
            urls_processed_in_session += 1

            # リンク抽出
            elems = driver.find_elements(By.TAG_NAME, 'a')
            for elem in elems:
                href = elem.get_attribute('href')
                if not href:
                    continue
                href = urljoin(url, href)
                parsed = urlparse(href)
                if parsed.scheme in ('http', 'https') and parsed.netloc == domain:
                    if href not in seen and href not in queue:
                        queue.append(href)
    finally:
        driver.quit()
        if state_file:
            try:
                for item in results:
                    seen.add(item['url'])
                state = {'queue': queue, 'seen': list(seen), 'results': results}
                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                print(f"現在のクロール状態を '{state_file}' に保存しました。")
            except Exception as e:
                print(f"警告: 状態ファイルの保存に失敗しました: {e}")

    if len(results) >= max_urls:
        print(f"最大収集URL数 {max_urls} に達しました。")
    elif not queue:
        print('クロールキューが空になりました。')

    print(f"クロール完了。収集した総URL数: {len(results)}。このセッションで処理したURL数: {urls_processed_in_session}")
    return results

# %%
# 3️⃣ メイン処理
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('指定された開始URLから同じドメイン内のURLをSeleniumで収集し、'
                     '結果をJSONファイルに出力します。進行状況は状態ファイルに保存/読み込みされます。'),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--start_url',
        type=str,
        default=None,
        help='クロールを開始するURL (例: https://example.com/)。state_fileから再開する場合は省略可。'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='収集したURLリストを保存するJSONファイル名 (必須)'
    )
    parser.add_argument(
        '--max_urls',
        type=int,
        default=1000,
        help='収集する最大のURL数 (デフォルト: 1000)'
    )
    parser.add_argument(
        '--state_file',
        type=str,
        default='crawl_state.json',
        help='状態保存ファイル名 (デフォルト: crawl_state.json)'
    )
    args = parser.parse_args()

    if not args.start_url and (not args.state_file or not os.path.exists(args.state_file)):
        parser.error('--start_url が指定されておらず、有効な --state_file も存在しません。')

    collected = crawl_domain(
        args.start_url,
        max_urls=args.max_urls,
        output_file=args.output,
        state_file=args.state_file
    )

    if collected:
        final = []
        for item in collected:
            final.append(item)
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(final, f, ensure_ascii=False, indent=2)
            print(f"✨ {len(final)} 件のURLを '{args.output}' に書き出しました。")
        except IOError as e:
            print(f"ファイル書き込みエラー: {args.output} - {e}")
