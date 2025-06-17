#!/usr/bin/env python3
"""
Crawl_URL_with_response.py

指定された開始URLから同一ドメイン内のURLを収集し、HTTPレスポンスボディも含めてJSON Lines形式で保存するスクリプト。
進行状況は任意の状態ファイルに保存/再開できます。
"""
import argparse
import json
import os
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup  # HTMLタグ除去用に追加

def crawl_domain_with_response(start_url, max_urls=1000, output_file=None, state_file=None):
    domain = urlparse(start_url).netloc if start_url else None
    queue = []
    seen = set()
    results = []

    # 状態ファイルから復元
    if state_file and os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            queue = state.get('queue', [])
            seen = set(state.get('seen', []))
            results = state.get('results', [])
            if not start_url and not queue:
                print('state_fileから再開できません。start_urlを指定してください。')
                return []
        except Exception as e:
            print(f'状態ファイル読み込み失敗: {e}\n新規クロールを開始します。')
            queue = [start_url]
    elif start_url:
        queue = [start_url]
    else:
        print('start_urlが指定されていません。')
        return []

    print(f'クロール対象ドメイン: {domain}')
    print(f'最大 {max_urls} 件まで収集します。')
    print(f'状態ファイルから復元: 収集済み {len(results)} 件、キューに {len(queue)} 件') # 追加

    try:
        while queue and len(results) < max_urls:
            url = queue.pop(0)
            if url in seen:
                continue
            # 変更: 収集済み件数とキューの残り件数を表示
            print(f'処理中: {url} (収集済み {len(results)}/{max_urls}, キュー残り {len(queue)} 件)')
            seen.add(url)

            # リクエスト
            time.sleep(1)
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                resp = requests.get(url, headers=headers, timeout=20)
                resp.raise_for_status()
            except Exception as e:
                print(f'リクエストエラー: {url} - {e}')
                continue

            content_type = resp.headers.get('Content-Type', '').lower()
            body = ''
            # テキストコンテンツのみ抽出、HTMLはタグを除去してテキスト化
            if content_type.startswith('text/html'):
                try:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    # 改行区切りでプレーンテキストを取得
                    body = soup.get_text(separator='\n', strip=True)
                except Exception as e:
                    print(f'HTMLテキスト抽出エラー: {url} - {e}')
                    body = ''
            elif content_type.startswith('text/'):
                body = resp.text

            record = {
                'domain': domain,
                'url': url,
                'content_type': content_type,
                'response_body': body
            }
            results.append(record)

            # HTMLならリンクを抽出
            if content_type.startswith('text/html'):
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, 'html.parser')
                for a in soup.find_all('a', href=True):
                    href = urljoin(url, a['href'])
                    parsed = urlparse(href)
                    if parsed.scheme in ('http', 'https') and parsed.netloc == domain:
                        if href not in seen and href not in queue:
                            queue.append(href)

            # 状態保存
            if state_file:
                try:
                    current_state = {
                        'queue': queue,
                        'seen': list(seen),
                        'results': results
                    }
                    with open(state_file, 'w', encoding='utf-8') as sf:
                        json.dump(current_state, sf, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f'状態ファイル保存失敗: {e}')

    except KeyboardInterrupt:
        print('Interrupted. 終了します。')
    finally:
        pass

    # 結果書き出し
    if output_file:
        try:
            # JSON Lines形式で追記
            with open(output_file, 'w', encoding='utf-8') as f:
                for rec in results:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f'{len(results)} 件を書き出しました: {output_file}')
        except Exception as e:
            print(f'出力ファイル書き込み失敗: {e}')
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ドメイン内URLとレスポンスボディを収集します。')
    parser.add_argument('--start_url', required=True, help='クロール開始URL')
    parser.add_argument('--output', required=True, help='結果をJSONL形式で保存するファイル')
    parser.add_argument('--max_urls', type=int, default=1000, help='最大URL収集数')
    parser.add_argument('--state_file', default=None, help='状態ファイル (省略可)')
    args = parser.parse_args()

    crawl_domain_with_response(
        start_url=args.start_url,
        max_urls=args.max_urls,
        output_file=args.output,
        state_file=args.state_file
    )
