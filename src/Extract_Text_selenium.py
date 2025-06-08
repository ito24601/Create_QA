\
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.options import ChromiumOptions
from selenium.common.exceptions import WebDriverException, TimeoutException as SeleniumTimeoutException
from bs4 import BeautifulSoup, NavigableString, Tag
import time
import re

def init_driver():
    options = ChromiumOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    
    try:
        driver = webdriver.Remote(
            command_executor='http://selenium:4444/wd/hub',
            options=options
        )
        driver.set_page_load_timeout(30) 
        return driver
    except Exception as e:
        print(f"WebDriverの初期化に失敗しました: {e}")
        raise

def get_structured_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for element in soup(["script", "style", "meta", "link", "header", "footer", "nav", "aside", "form", "button", "iframe", "img", "svg", "noscript", "select", "input", "textarea", "figure", "figcaption"]):
        if element:
            element.decompose()

    texts = []

    def _extract_recursive(element, list_level=0):
        if isinstance(element, NavigableString):
            text = element.string.strip() if element.string else ""
            if text:
                texts.append(text + " ") # 単語間のスペースを確保するため末尾にスペース
            return

        if not isinstance(element, Tag):
            return

        tag_name = element.name
        
        if re.match(r'h[1-6]', tag_name):
            level = int(tag_name[1])
            heading_text = element.get_text(separator=' ', strip=True)
            if heading_text:
                texts.append("\\n" + "#" * level + " " + heading_text + "\\n")
        elif tag_name == 'p':
            para_text = element.get_text(separator=' ', strip=True)
            if para_text:
                texts.append(para_text + "\\n")
        elif tag_name == 'ul':
            texts.append("\\n")
            for item_element in element.find_all('li', recursive=False):
                texts.append("  " * list_level + "- ")
                for child_item in item_element.children:
                    _extract_recursive(child_item, list_level=list_level + 1)
                if texts and not texts[-1].endswith("\\n"): texts.append("\\n")
            if list_level == 0: texts.append("\\n") # トップレベルのリストの後に改行
        elif tag_name == 'ol':
            texts.append("\\n")
            for i, item_element in enumerate(element.find_all('li', recursive=False)):
                texts.append("  " * list_level + f"{i + 1}. ")
                for child_item in item_element.children:
                    _extract_recursive(child_item, list_level=list_level + 1)
                if texts and not texts[-1].endswith("\\n"): texts.append("\\n")
            if list_level == 0: texts.append("\\n")
        elif tag_name == 'table':
            texts.append("\\n--- Table ---\\n")
            for row in element.find_all('tr'):
                cols = [col.get_text(separator=' ', strip=True) for col in row.find_all(['th', 'td'])]
                if any(c.strip() for c in cols):
                    texts.append("| " + " | ".join(cols) + " |\\n")
            texts.append("--- End Table ---\\n\\n")
        elif tag_name == 'blockquote':
            texts.append("\\n> ")
            block_text = element.get_text(separator='\\n', strip=True)
            if block_text:
                texts.append(block_text.replace('\\n', '\\n> ') + "\\n\\n")
        elif tag_name == 'pre':
            texts.append("\\n```\\n")
            texts.append(element.get_text(strip=False))
            texts.append("\\n```\\n\\n")
        elif tag_name == 'hr':
            texts.append("\\n---\\n\\n")
        elif tag_name == 'br':
            if texts and not texts[-1].endswith("\\n"):
                 texts.append("\\n")
        else: # div, span, article, section, main etc.
            for child in element.children:
                _extract_recursive(child, list_level)
            if tag_name in ['article', 'section', 'main', 'div'] and element.get_text(strip=True):
                if texts and not texts[-1].endswith("\\n\\n") and not texts[-1].endswith("\\n"):
                    texts.append("\\n")

    main_content_selectors = ['main', 'article', '[role="main"]', '.content', '#content', '.main-content', '#main-content', '.post-body', '.entry-content', 'body']
    main_content_area = None
    for selector in main_content_selectors:
        if selector == 'body': # fallback
            main_content_area = soup.body
            break
        main_content_area = soup.select_one(selector)
        if main_content_area:
            break
    
    if not main_content_area:
         return "主要コンテンツ領域が見つかりませんでした。"

    for child in main_content_area.children:
        _extract_recursive(child)
        
    result_text = "".join(texts)
    
    result_text = re.sub(r' (\n)', r'\g<1>', result_text) # 行末の不要なスペースを削除
    result_text = re.sub(r'\n\s*\n', '\\n\\n', result_text) 
    result_text = re.sub(r'(\\n\\n)+', '\\n\\n', result_text)
    result_text = re.sub(r'^\\s+', '', result_text)
    return result_text.strip()

def main():
    parser = argparse.ArgumentParser(description="指定されたURLから構造化されたテキストを抽出します。")
    parser.add_argument("url", help="テキストを抽出する対象のURL")
    parser.add_argument("--output", help="抽出したテキストを保存するファイル名 (省略時はコンソールに出力)")
    parser.add_argument("--wait_time", type=int, default=5, help="ページロード後の待機時間 (秒、デフォルト: 5)")

    args = parser.parse_args()

    driver = None
    try:
        driver = init_driver()
        print(f"アクセス中: {args.url}")
        driver.get(args.url)
        
        print(f"{args.wait_time}秒待機します (JavaScriptによる動的コンテンツ読み込みのため)...")
        time.sleep(args.wait_time)

        html_content = driver.page_source
        print("HTMLコンテンツを取得しました。テキスト抽出を開始します...")
        extracted_text = get_structured_text_from_html(html_content)
        print("テキスト抽出が完了しました。")

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"抽出したテキストを '{args.output}' に保存しました。")
        else:
            print("\\n--- 抽出されたテキスト ---")
            print(extracted_text)
            print("--- テキスト終 ---")

    except SeleniumTimeoutException:
        print(f"ページロードがタイムアウトしました: {args.url}")
    except WebDriverException as e:
        print(f"WebDriverエラーが発生しました: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
    finally:
        if driver:
            try:
                driver.quit()
                print("WebDriverを終了しました。")
            except Exception as e:
                print(f"WebDriver終了時にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
