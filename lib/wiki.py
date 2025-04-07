import hashlib
import wikitextparser
import xml.etree.ElementTree as etree
from dateutil import parser
import pypandoc
import re

URL_BASE = 'https://en.wikipedia.org/wiki/'
URL_COMMONS = 'https://commons.wikimedia.org/wiki/'
URL_FILE = 'https://upload.wikimedia.org/wikipedia/commons/'
PATTERN_A = r'<a[^>]*>(.+?)</a>'
PATTERN_ALT = r'alt=["\'](.+?)["\']'
PATTERN_CLASS = r'class=["\'](.+?)["\']'
PATTERN_HREF = r'href=["\'](.+?)["\']'
PATTERN_IMG_HTML = r'<img[^>]*>'
PATTERN_IMG_MD = r'!\[((?:[^\[\]]|(?:\[[^\[\]]*\]))*)\]\(((?:[^()]|(?:\([^()]*\)))*)\)'
PATTERN_TITLE = r'title=["\'](.+?)["\']'
PATTERN_SRC = r'src=["\'](.+?)["\']'
PROTOCOLS = ('http://', 'https://')


def is_url(text):
    return text.startswith(PROTOCOLS)


def replacer_a(match):
    link_text = parse_inline_html_style(match.group(1))
    href_match = re.search(PATTERN_HREF, match[0])
    title_match = re.search(PATTERN_TITLE, match[0])
    class_match = re.search(PATTERN_CLASS, match[0])
    href = ori_href = href_match.group(1) if href_match else None
    title = title_match.group(1) if title_match else None
    sclass = class_match.group(1) if class_match else None
    if sclass == 'wikilink' and not ori_href.startswith(('http://', 'https://')):
        href = parse_wikilinks(ori_href)
    text = link_text or title or ori_href or sclass
    return f'[{text}]({href} "{text}")'


def replacer_img_html(match):
    src_match = re.search(PATTERN_SRC, match[0])
    title_match = re.search(PATTERN_TITLE, match[0])
    alt_match = re.search(PATTERN_ALT, match[0])
    src = src_match.group(1) if src_match else None
    title = title_match.group(1) if title_match else None
    alt = alt_match.group(1) if alt_match else None
    if not src:
        return match[0]
    full_src = parse_image_path(src)
    text = title or alt or src
    return f'[![{text}]({full_src[0]})]({full_src[1]} "{text}")'


def replacer_img_md(match):
    image_path = match.group(2).split(' ')[0]
    if not is_url(image_path):
        image_path = match[0].replace(
            image_path, parse_image_path(image_path)[0])
    return image_path


def parse_inline_html_style(text):
    text = re.sub(r'<em>(.*?)</em>', r'*\1*', text)
    text = re.sub(r'<strong>(.*?)</strong>', r'**\1**', text)
    text = re.sub(r'<strike>(.*?)</strike>', r'~~\1~~', text)
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text)
    return text


def parse_filename(text):
    return text.replace(' ', '_')


def parse_wikilinks(text, URL_BASE=URL_BASE, URL_COMMONS=URL_COMMONS):
    media_namespaces = ['file', 'image', 'media', 'timedtext']
    if text.startswith('c:'):
        return f"{URL_COMMONS}{parse_filename(text[2:])}"
    if ':' in text:
        namespace, page_name = text.split(':', 1)
        namespace = namespace.lower()
        if namespace == 'commons':
            if ':' in page_name:
                return f"{URL_COMMONS}{parse_filename(page_name)}"
            else:
                return f"{URL_COMMONS}{parse_filename(text)}"
        if namespace in media_namespaces:
            return f"{URL_COMMONS}File:{parse_filename(page_name)}"
        elif namespace in ['category']:
            return f"{URL_BASE}Category:{parse_filename(page_name)}"
        return f"{URL_BASE}{parse_filename(text)}"
    return f"{URL_BASE}{parse_filename(text)}"


def parse_a(text):
    return re.sub(PATTERN_A, replacer_a, text)


def parse_img_html(text):
    return re.sub(PATTERN_IMG_HTML, replacer_img_html, text)


def parse_img_md(text):
    return re.sub(PATTERN_IMG_MD, replacer_img_md, text)


def parse_image_path(src):
    if is_url(src):
        return (src, src)
    filename = parse_filename(src)
    md5hash = hashlib.md5(filename.encode('utf-8')).hexdigest()
    return (
        f"{URL_FILE}{md5hash[0]}/{md5hash[0:2]}/{filename}",
        f'{URL_COMMONS}File:{filename}'
    )


def post_process(text):
    try:
        text = parse_a(text)
    except Exception as e:
        print(f"Error: {e}, handlled!")
    try:
        text = parse_img_html(text)
    except Exception as e:
        print(f"Error: {e}, handlled!")
    try:
        text = parse_img_md(text)
    except Exception as e:
        print(f"Error: {e}, handlled!")
    return text


def wikitext_to_markdown(text):
    parsed = wikitextparser.parse(text)
    for template in parsed.templates:
        pass
    try:
        markdown = pypandoc.convert_text(
            str(parsed), 'gfm', format='mediawiki', extra_args=['--wrap=none']
        )
    except Exception as e:
        print(f"Error: {e}")
        markdown = str(parsed)
    return post_process(markdown)


def parse_wikipedia_dump(file_path, callback=None, check_exist=None):

    def strip_tag_name(t):
        t = elem.tag
        idx = t.rfind("}")
        if idx != -1:
            t = t[idx + 1:]
        return t

    namespaces = {'0': 'Page'}
    item = {}
    should_continue = False

    for event, elem in etree.iterparse(file_path, events=('start', 'end')):
        tname = strip_tag_name(elem.tag)
        if event == 'start':
            if tname == 'page':
                item = {
                    'id': None,
                    'title': '',
                    'url': '',
                    'redirect': '',
                    'redirecturl': '',
                    'contributor': {},
                    'revisionid': None,
                    'parentid': None,
                    'namespace': '',
                    'timestamp': None,
                    'comment': '',
                    'origin': None,
                    'model': None,
                    'format': None,
                    'sha1': None,
                    'text': '',
                }
                in_group = {}
                should_continue = False
            elif should_continue:
                continue
            elif tname == 'namespace' and elem.get('key') != None and elem.text != None:
                namespaces[elem.get('key')] = elem.text
            elif tname in ['revision', 'contributor']:
                in_group[tname] = True
            elif tname == 'id' and elem.text != None:
                id = int(elem.text)
                if in_group.get('contributor'):
                    item['contributor']['id'] = id
                elif in_group.get('revision'):
                    item['revisionid'] = id
                else:
                    item['id'] = id
                    if check_exist and check_exist(item['id']):
                        should_continue = True
                        continue
            elif tname == 'username' and elem.text != None:
                if in_group['contributor']:
                    item['contributor']['username'] = elem.text
            elif tname == 'redirect':
                item['redirect'] = (elem.get('title') or '').strip()
                item['redirecturl'] = f"{URL_BASE}{parse_filename(item['redirect'])}"
            elif tname == 'ns' and elem.text != None:
                item['namespace'] = namespaces.get(elem.text) \
                    or namespaces['0']
            elif tname == 'parentid' and elem.text != None:
                item['parentid'] = int(elem.text)
            elif tname in ['origin', 'model', 'format', 'sha1'] \
                    and elem.text != None:
                item[tname] = elem.text.strip()
            elif tname == 'timestamp' and elem.text != None:
                item['timestamp'] = parser.parse(elem.text)
            elif tname == 'title' and elem.text != None:
                item[tname] = elem.text.strip()
                item['url'] = f"{URL_BASE}{parse_filename(elem.text)}"
            elif tname in ['comment', 'text'] and elem.text != None:
                item[tname] = wikitext_to_markdown(elem.text).strip()
        elif tname == 'page':
            if should_continue:
                continue
            elif callback:
                callback(item)
            else:
                print(item)
        elem.clear()


__all__ = [
    'parse_wikipedia_dump',
]
