import os
import webbrowser

_OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'recordings')

if __name__ == '__main__':
    htmls = sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(_OUTPUT_BASE)
        for f in files
        if f.endswith('.html')
    )
    if not htmls:
        print(f'No HTML files found under {_OUTPUT_BASE}')
    for path in htmls:
        print(f'Opening {path}')
        webbrowser.open(f'file://{os.path.abspath(path)}')
