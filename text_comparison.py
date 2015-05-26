#!/usr/bin/env python
'''
compares two text files for differences. text files should be located in the
same directory as the script and called 'comp1.txt' and 'comp2.txt'
'''

import difflib
from progressbar import ProgressBar
from flask import Flask
import webbrowser

app = Flask(__name__)

file1 = 'comp1.txt'
file2 = 'comp2.txt'

webbrowser.open('http://127.0.0.1:5000')

@app.route("/")
def make_html():
    pbar = ProgressBar().start()

    with open(file1) as f:
        f1_text = f.read()
    with open(file2) as f:
        f2_text = f.read()

    html = difflib.HtmlDiff()
    raw_html = html.make_file(f1_text.splitlines(), f2_text.splitlines())

    pbar.finish()

    return raw_html


if __name__ == "__main__":
    app.run()
