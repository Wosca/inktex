from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenizer.latex_tokenizer import LatexTokenizer, tokenize_latex
from tokenizer.standardize import standardize_latex

SHARDS = Path(r"E:\math-recogniser-training\colab-shards")
REAL_SHARD = SHARDS / "stage2_train" / "shard-00000.tar"
TOKENIZER = SHARDS / "tokenizer.json"


@pytest.mark.parametrize(("raw", "expected"), [
    (r"a\lt b", "a<b"),
    (r"a\gt b", "a>b"),
    (r"1,\cdots,n", r"1,\ldots,n"),
    (r"1,\dots,n", r"1,\ldots,n"),
    (r"x_{1}+\dotsb+x_{n}", r"x_{1}+\ldots+x_{n}"),
    (r"\sum\limits_{i=1}^{n}i", r"\sum_{i=1}^{n}i"),
    (r"\lim\nolimits_{x\to0}x", r"\lim_{x\rightarrow0}x"),
    (r"\displaystyle\frac{1}{2}", r"\frac{1}{2}"),
    (r"A\longrightarrow B", r"A\rightarrow B"),
    (r"A\longleftarrow B", r"A\leftarrow B"),
    (r"p\iff q", r"p\Leftrightarrow q"),
    (r"p\implies q", r"p\Rightarrow q"),
    (r"\bigwedge_{i}p_{i}", r"\wedge_{i}p_{i}"),
    (r"\bigvee p", r"\vee p"),
])
def test_aliases_have_one_spelling(raw, expected):
    assert standardize_latex(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    (r"\mbox{C}", "C"),
    (r"\mbox{j}", "j"),
    (r"\mathrm{d}x", "dx"),
    (r"\mathrm{card}(A)", "card(A)"),
    (r"x\mbox{and}y", "xandy"),
    (r"\mbox{ x }", "x"),
    (r"\mbox{}x", "x"),
    (r"\mbox{sin}x", r"\sin x"),
    (r"\text{lim}_{n}a_{n}", r"\lim_{n}a_{n}"),
    (r"\operatorname{arg}z", r"\arg z"),
    (r"\mathit{log}x", r"\log x"),
    (r"\mathrm{\Delta}x", r"\Delta x"),
    (r"\mbox{if x}", r"\mbox{if x}"),
    (r"\text{if x}", r"\mbox{if x}"),
    (r"\operatorname{arg max}f", r"\mbox{arg max}f"),
    (r"x_{\mathrm{max}}", r"x_{\max}"),
])
def test_text_and_font_wrappers(raw, expected):
    assert standardize_latex(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    (r"exp-\frac{v}{k}", r"\exp-\frac{v}{k}"),
    ("ln-1", r"\ln-1"),
    ("cos+1", r"\cos+1"),
    ("cosine-1", "cosine-1"),
    ("index+1", "index+1"),
    ("xsin-1", "xsin-1"),
    (r"\mbox{exp-x}", r"\mbox{exp-x}"),
])
def test_bare_functions_before_signs(raw, expected):
    assert standardize_latex(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("A^{2}_{k}", "A_{k}^{2}"),
    (r"\int\limits^{a}_{0}f", r"\int_{0}^{a}f"),
    (r"\int^{a}_{0}f", r"\int_{0}^{a}f"),
    (r"p^{\beta_{1}}_{1}", r"p_{1}^{\beta_{1}}"),
    ("x^{a^{2}_{b}}_{c}", "x_{c}^{a_{b}^{2}}"),
    ("f'_{1}", r"f_{1}^{\prime}"),
    ("A_{k}^{2}", "A_{k}^{2}"),
])
def test_subscript_before_superscript(raw, expected):
    assert standardize_latex(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("{-3}", "-3"),
    (r"{\ldots-\sqrt{l}}", r"\ldots-\sqrt{l}"),
    ("{x^{2}}+1", "x^{2}+1"),
    ("{C_{n}^{k}}{x^{n-k}}", "C_{n}^{k}x^{n-k}"),
    ("{(a+b)}+c", "(a+b)+c"),
    ("{{a}+b}", "a+b"),
    (r"{\gamma}_{\mbox{c}}", r"\gamma_{c}"),
    ("{(x^{1})}^{2}", "(x^{1})^{2}"),
    ("{(x+y)}^{2}", "(x+y)^{2}"),
    ("{|p|}^{2}", "|p|^{2}"),
    (r"{\vec{x}}_{i}", r"\vec{x}_{i}"),
    (r"{\frac{a}{b}}^{2}", r"\frac{a}{b}^{2}"),
    ("{a^{b}}^{c}", "{a^{b}}^{c}"),
    ("{a+b}^{2}", "{a+b}^{2}"),
    ("{12}^{2}", "{12}^{2}"),
    ("{(a)(b)}^{2}", "{(a)(b)}^{2}"),
    ("{}^{14}_{6}C", "{}_{6}^{14}C"),
    (r"\frac{1}{2}", r"\frac{1}{2}"),
    (r"\frac{{a}}{{b}}", r"\frac{a}{b}"),
    (r"\sqrt{x}", r"\sqrt{x}"),
    (r"\sqrt[3]{x}", r"\sqrt[3]{x}"),
    (r"\overline{x}", r"\overline{x}"),
    (r"\mathbb{R}", r"\mathbb{R}"),
    (r"\mathcal{L}", r"\mathcal{L}"),
    (r"\mathbf{x}", r"\mathbf{x}"),
    (r"\hat{x}", r"\hat{x}"),
    (r"\overset{a}{=}", r"\overset{a}{=}"),
    (r"\begin{array}{cc}a&b\end{array}", r"\begin{array}{cc}a&b\end{array}"),
    (r"\begin{matrix}{a&b}\end{matrix}", r"\begin{matrix}{a&b}\end{matrix}"),
])
def test_redundant_braces_only(raw, expected):
    assert standardize_latex(raw) == expected


def test_stripped_output_stays_balanced_tex():
    for raw in ("{{{x}}}", r"{\frac{{1}}{{2}}}^{{2}}", "{a}{b}{c}", r"\sqrt{{x+1}}", "{x_{1}}_{2}"):
        out = standardize_latex(raw)
        tokens = tokenize_latex(out)
        depth = 0
        for token in tokens:
            depth += (token == "{") - (token == "}")
            assert depth >= 0, out
        assert depth == 0, out


@pytest.mark.parametrize(("raw", "expected"), [
    ("a·b", r"a\cdot b"),
    ("a±b", r"a\pm b"),
    ("f′", r"f^{\prime}"),
    ("x−1", "x-1"),
    ("α+β", r"\alpha+\beta"),
])
def test_unicode_math_is_mapped(raw, expected):
    assert standardize_latex(raw) == expected


@pytest.mark.parametrize("raw", ["1+3\\rightarrow 第9行", "50。", "、BE=3", r"\mbox{été}", "x\u200b+1", "", "{}",
                                 "{x", "x}", r"\frac{1}"])
def test_non_ascii_and_malformed_are_rejected(raw):
    assert standardize_latex(raw) is None


CORPUS = [
    r"\frac{-b\pm\sqrt{b^{2}-4ac}}{2a}", r"\sum_{i=1}^{n}i^{2}=\frac{n(n+1)(2n+1)}{6}", r"\int_{0}^{\infty}e^{-x}dx",
    r"\lim_{x\to0}\frac{\sin x}{x}=1", r"\left(\frac{a}{b}\right)^{2}", r"a\lt b\gt c", r"\mbox{if }x\geq0",
    r"{{\mbox{b}}_{Y}+8.5}", r"\sqrt{{{-m}+\mbox{t}}}", r"\binom{n}{k}", r"x'' + y'", r"\vec{v}\cdot\vec{w}",
    r"\begin{pmatrix}a&b\\c&d\end{pmatrix}", r"\overline{AB}", r"{x^{y}}-{y^{x}}\mbox{xl}f(v,t)", "1,2,...,n",
    r"\mathrm{2ln}(x)", r"\dfrac{1}{2}", r"x \over y", r"\mathbb{R}^{n}_{+}", r"{\sum}_{i}", r"e^{i\pi}+1=0",
]


def test_idempotent_on_corpus():
    for raw in CORPUS:
        once = standardize_latex(raw)
        assert once is not None, raw
        assert standardize_latex(once) == once, (raw, once)


@pytest.mark.skipif(not REAL_SHARD.is_file() or not TOKENIZER.is_file(), reason="real shards unavailable")
def test_real_labels_are_idempotent_and_in_vocabulary():
    from data.bimodal_shards import iter_shard

    tokenizer = LatexTokenizer.load(TOKENIZER)
    assert len(tokenizer) == 907
    vocabulary = tokenizer.token_to_id
    checked = rejected = 0
    for record in iter_shard(REAL_SHARD, load_npz=False):
        raw = json.loads(record["json"])["normalized_latex"]
        out = standardize_latex(raw)
        if out is None:
            rejected += 1
            assert not raw.isascii() or not raw.strip(), raw
        else:
            assert standardize_latex(out) == out, (raw, out)
            unknown_before = {token for token in tokenize_latex(raw) if token not in vocabulary}
            unknown_after = {token for token in tokenize_latex(out) if token not in vocabulary}
            assert unknown_after <= unknown_before, (raw, out, unknown_after - unknown_before)
        checked += 1
        if checked >= 500:
            break
    assert checked == 500 and rejected <= 5
