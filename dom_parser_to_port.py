import re
import base64
from bs4 import BeautifulSoup, Comment

# Pattern for 6-character alphanumeric codes
CODE_PATTERN = re.compile(r'\b([A-Z0-9]{6})\b')

# Common false positives to ignore - words that match 6-char pattern but aren't codes
FALSE_POSITIVES = {
    'DEVICE', 'VIEWPORT', 'SCRIPT', 'BUTTON', 'SUBMIT', 'ACCEPT',
    'COOKIE', 'SCROLL', 'HIDDEN', 'STYLES', 'WINDOW', 'SCREEN',
    'CHROME', 'WEBKIT', 'SAFARI', 'MOBILE', 'TABLET', 'ROBOTS',
    'NOINDEX', 'FOLLOW', 'NOFOLL', 'WIDTHD', 'HEIGHT', 'MARGIN',
    'FILLER', 'MOVING', 'LOADED', 'REVEAL', 'CHOICE', 'BEATAE',
    'TEMPOR', 'FUGIAT', 'ALIQUA', 'OPTION', 'DIALOG', 'ANSWER',
    'SELECT', 'DOLORE', 'MOLLIT', 'VENIAM', 'CILLUM', 'PLEASE',
    'LABORE', 'CONTENT', 'SECTION', 'HEADER', 'FOOTER', 'BORDER',
    'COLORS', 'IMAGES', 'CANCEL', 'RETURN', 'SUBMIT', 'CHANGE',
    'UPDATE', 'DELETE', 'CREATE', 'SEARCH', 'FILTER', 'NOTICE',
    'ALERTS', 'ERRORS', 'STATUS', 'RESULT', 'OUTPUT', 'INPUTS',
    '1500MS', '2500MS', '3500MS', '500PX0', 'BEFORE', 'AFTER0',
    'APPEAR', 'STICKY', 'NORMAL', 'INLINE', 'CENTER', 'BOTTOM',
    'SHADOW', 'CURSOR', 'ZINDEX', 'EASING', 'ROTATE', 'SMOOTH',
    'LAYOUT', 'RENDER', 'EFFECT', 'TOGGLE', 'HANDLE', 'CUSTOM',
    'PIXELS', 'POINTS', 'WEIGHT', 'SOURCE', 'TARGET', 'ORIGIN',
    'OBJECT', 'STRING', 'NUMBER', 'PROMPT', 'ACCESS', 'GLOBAL',
    'EXPORT', 'IMPORT', 'MODULE', 'SHOULD', 'UNSAFE', 'STRICT',
    'SIGNAL', 'STREAM', 'BUFFER', 'PARSED', 'THROWS', 'FIELDS',
    'CHOOSE', 'LABELS', 'BUTTON', 'CLOSER', 'SCROLL', 'TRICKS',
    'FAKING', 'PRIZES', 'MODALS', 'RADIOS', 'DECOYS', 'PROCED',
    'FILLED', 'PIECES', 'SIGNUP', 'BLOCKS', 'CHARTS', 'THINGS',
    'SAMPLE', 'VERIFY', 'PARAMS', 'EVENTS', 'CHECKS', 'CODING',
    'SINGLE', 'DOUBLE', 'EXPAND', 'UNIQUE', 'RECENT', 'ACTIVE',
    'RANDOM', 'CLOSED', 'OPENED', 'MARKED', 'CALLED', 'PASSED',
    'FAILED', 'PAUSED', 'LISTED', 'VALUED', 'STORED', 'POSTED',
    'COVERS', 'TIMERS', 'COUNTS', 'YELLOW', 'SECCND', 'BLACKS',
    'WHITES', 'GREENS', 'SPACES', 'SECOND', 'MINUTE', 'STARTS',
    'MEMORY', 'LOADED', 'BLOCKS', 'REMAIN', 'SIMPLE', 'NEEDED',
    'EXTEND', 'INFORM', 'PICKED', 'OPTION', 'CHOICE', 'CHOSEN',
    'CANVAS', 'STROKE', 'DRAWIN', 'DRAWAN', 'LISTEN', 'COMPLT',
    'TIMING', 'FRAMES', 'CAPTUR', 'PUZZLE', 'ROTATE', 'SCROLL',
    'MULTIT', 'TABBED', 'REVEAL', 'HIDDEN', 'DECODE', 'BASE64',
    'PLAYED', 'ESCAPE', 'ALMOST', 'INSIDE', 'SEQUEN', 'PROGRE',
    'CLICKM', 'SCROLL', 'FILLER', 'SQUARE', 'CIRCLE', 'DRAWIN',
    'GESTUR', 'SOLVED', 'PAGEGO', 'MEPICK', 'ONETHE',
    'CACHED', 'SERVIC', 'LAYERS', 'LEVELS', 'NESTED', 'SERVER',
    'SHADOW', 'SOCKET', 'CONNEC', 'HEREGO', 'IFRAME', 'BWRONG',
    'ONNEXT', 'MUTATI', 'DEEPER', '1WRONG',
}


def extract_hidden_codes(html: str) -> list[str]:
    """Extract potential 6-character codes from HTML (hidden AND visible)."""
    codes = set()
    soup = BeautifulSoup(html, 'html.parser')

    # 0. First, find ALL 6-char alphanumeric codes in visible text
    # This catches codes displayed after scroll reveals, etc.
    all_text = soup.get_text(separator=' ')
    codes.update(CODE_PATTERN.findall(all_text.upper()))

    # 1. Check data-* attributes
    for elem in soup.find_all(True):
        for key, value in elem.attrs.items():
            if key.startswith('data-') and isinstance(value, str):
                codes.update(CODE_PATTERN.findall(value.upper()))

    # 2. Check aria-* attributes
    for elem in soup.find_all(True):
        for key, value in elem.attrs.items():
            if key.startswith('aria-') and isinstance(value, str):
                codes.update(CODE_PATTERN.findall(value.upper()))

    # 3. Check hidden elements (display:none, visibility:hidden, hidden attribute)
    for elem in soup.find_all(style=re.compile(r'display:\s*none|visibility:\s*hidden')):
        text = elem.get_text()
        codes.update(CODE_PATTERN.findall(text.upper()))

    for elem in soup.find_all(attrs={'hidden': True}):
        text = elem.get_text()
        codes.update(CODE_PATTERN.findall(text.upper()))

    # 4. Check HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        codes.update(CODE_PATTERN.findall(str(comment).upper()))

    # Also search raw HTML for comments (backup)
    comment_pattern = re.compile(r'<!--(.*?)-->', re.DOTALL)
    for match in comment_pattern.findall(html):
        codes.update(CODE_PATTERN.findall(match.upper()))

    # 5. Check meta tags
    for meta in soup.find_all('meta'):
        content = meta.get('content', '')
        if isinstance(content, str):
            codes.update(CODE_PATTERN.findall(content.upper()))

    # 6. Check title attribute
    for elem in soup.find_all(attrs={'title': True}):
        title = elem.get('title', '')
        if isinstance(title, str):
            codes.update(CODE_PATTERN.findall(title.upper()))

    # 7. Decode Base64 strings in the page (may contain hidden codes)
    b64_pattern = re.compile(r'[A-Za-z0-9+/]{8,}={0,2}')
    all_text = soup.get_text(separator=' ') + ' ' + html
    for b64_match in b64_pattern.findall(all_text):
        try:
            decoded = base64.b64decode(b64_match).decode('utf-8', errors='ignore')
            if decoded:
                codes.update(CODE_PATTERN.findall(decoded.upper()))
        except Exception:
            pass
    # Also check data attributes for base64
    for elem in soup.find_all(True):
        for key, value in elem.attrs.items():
            if key.startswith('data-') and isinstance(value, str) and len(value) >= 8:
                try:
                    decoded = base64.b64decode(value).decode('utf-8', errors='ignore')
                    if decoded:
                        codes.update(CODE_PATTERN.findall(decoded.upper()))
                except Exception:
                    pass

    # Filter out false positives
    codes = {c for c in codes if c not in FALSE_POSITIVES}

    # Filter out CSS measurement artifacts (e.g. 6480PX, 2158PX, 100VH, 50VW, 200REM)
    codes = {c for c in codes if not re.match(r'^\d+(?:PX|VH|VW|EM|REM|CH|EX|PC|PT|MM|CM|IN|MS|FR)$', c)}

    # Filter out pure numeric strings (not likely codes)
    codes = {c for c in codes if not c.isdigit()}

    # Sort: codes with numbers first (more likely to be real codes like "TWA8Q7")
    def has_number(s):
        return any(c.isdigit() for c in s)

    sorted_codes = sorted(codes, key=lambda c: (not has_number(c), c))
    return sorted_codes


def find_real_next_button(html: str) -> str | None:
    """Find the selector for the real navigation button among decoys."""
    soup = BeautifulSoup(html, 'html.parser')

    # Look for buttons/links with navigation-related onclick or href
    for elem in soup.find_all(['button', 'a']):
        onclick = elem.get('onclick', '')
        href = elem.get('href', '')

        # Check if it actually navigates
        if 'step' in href.lower() or 'next' in onclick.lower():
            if elem.get('id'):
                return f"#{elem['id']}"
            if elem.get('class'):
                return f".{elem['class'][0]}"

    return None