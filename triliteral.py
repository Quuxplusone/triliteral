import argparse
import collections
import dataclasses
import itertools
import os.path
import re
import sys

# Latin has trouble with digraphs: "ushah" means "ušah", not "us-hah",
# and will be treated that way by the instruction unpacker and
# when recoding into Hebrew and/or Arabic.
# To disambiguate a pair of consonants that would otherwise make one
# digraph, separate them with an ignored character such as "-".
# Likewise, the recoder will encode Hebrew "צ" into Latin "ts"
# but Hebrew "טס" into Latin "t-s".
#
# Also, consider letters that are homophonous in Latin: Latin "s" and "x"
# both encode into Hebrew "ס". So "t-x-s" and "t-x-x" denote different
# variables in Latin but denote the same variable "ט-ס-ס" in Hebrew.

args = argparse.Namespace()
variables = collections.defaultdict(int)
script = None
LATIN = dict(
    values=dict(
        A=1, B=2, G=3, D=4, H=5, E=5, U=6, V=6, W=6, Z=7, C=8, Ch=8, T=9,
        I=10, J=10, Y=10, K=20, L=30, M=40, N=50, S=60, X=60, O=70, F=80, P=80, Ph=80, Ts=90, Tz=90,
        Q=100, R=200, Sh=300, Th=400, K_=500, M_=600, N_=700, P_=800, Ph_=800, Ts_=900, Tz_=900,
    ),
    vowels={'': 0, '-': 0, 'A': 1, 'E': 1, 'U': 2, 'I': 3},
    ext='.tlt',
)
ARABIC=dict(
    values=dict(
        ا=1, ب=2, ج=3, د=4, ه=5, و=6, ز=7, ح=8, ط=9,
        ي=10, ك=20, ل=30, م=40, ن=50, س=60, ع=70, ف=80, ص=90,
        ق=100, ر=200, ش=300, ت=400, ث=500, خ=600, ذ=700, ض=800, ظ=900, غ=1000,
    ),
    vowels={'': 0, 'ا': 1, 'و': 2, 'ي': 3},
    ext='.ثلث',
)
HEBREW=dict(
    values=dict(
        א=1, ב=2, ג=3, ד=4, ה=5, ו=6, ז=7, ח=8, ט=9,
        י=10, כ=20, ל=30, מ=40, נ=50, ס=60, ע=70, פ=80, צ=90,
        ק=100, ר=200, ש=300, ת=400, ך=500, ם=600, ן=700, ף=800, ץ=900,
    ),
    vowels={'': 0, 'א': 1, 'ה': 1, 'ו': 2, 'י': 3},
    ext='.תלת',
)


def trace(s):
    if args.trace:
        print(s)


def parse(path):
    with open(path, 'r') as f:
        return [w for line in f for w in line.split()]


def unpack(word):
    values, vowels = script['values'], script['vowels']
    gotvowel = False
    root = ''
    stem = ''
    word = word.replace('ـ', '').upper()
    while word:
        if word[0] in vowels:
            stem += str(vowels[word[0]])
            word = word[1:]
            gotvowel = True
        else:
            if not gotvowel:
                stem += '0'
            gotvowel = False
            for i in 3, 2, 1:
                c = word[:i].title()
                if c in values:
                    break
            root += ('-' if root else '') + c.upper()
            word = word[i:]
    return root, int(stem[::-1], 4)


def gem(word):
    acc = 0
    word = word.replace('ـ', '').upper() + '_'
    values = script['values']
    while word and word != '_':
        if word[0] == '-':
            word = word[1:]
            continue
        for i in 3, 2, 1:
            c = word[:i].title()
            if c in values:
                acc += values[c]
                break
        word = word[i:]
    return acc


def degem(n):
    sv, sw = script['values'], script['vowels']
    vowels, consonants = [], []
    m = n
    while m > 0:
        for c, v in reversed(sv.items()):
            if (c[-1] != '_' or not consonants) and (c + '_' not in sv or consonants) and v <= m:
                m -= v
                (vowels if c in sw else consonants)[:0] = (c[:-1] if c[-1] == '_' else c)
                break
    r = ''.join(itertools.chain.from_iterable(itertools.zip_longest(vowels, consonants, fillvalue='')))
    if script is LATIN:
        r = r.replace('TSh', 'CASh')
    assert gem(r) == n, (n, r, gem(r))
    return r


def recode(word, to):
    out = ''
    sv, sw = script['values'], script['vowels']
    tv, tw = to['values'], to['vowels']
    word = word.upper() + '_'
    while word and word != '_':
        for i in 3, 2, 1:
            c = word[:i].title()
            if c in sv:
                break
        word = word[i:]
        if (c in sv) and (c != '-'):
            x = sv[c]
            y = sw.get(c, 0)
            try:
                recoded = next(k for k, v in tv.items() if v == x and tw.get(k, 0) == y).strip('_')
                if len(out) and (out[-1] + recoded).title() in tv:
                    out += '-'
                out += recoded.lower()
            except StopIteration:
                raise Exception(f"can't recode '{c}' to {args.recode}")
    return out


def recode_to_code(word):
    root, stem = unpack(word)
    op = OPS[stem]
    if (root.count('-') == 2) and (op is not None):
        return '(%s %s)' % (op.__name__.upper().strip('_'), root.lower())
    else:
        return '(%d)' % gem(word)

def test_recode():
    global script
    script = LATIN
    assert recode_to_code('tsc') == '(98)'
    assert recode_to_code('itst') == '(109)'
    assert recode_to_code('it-st') == '(WITH t-s-t)'
    assert recode_to_code('utasat') == '(NOT t-s-t)'
    assert recode_to_code('atust') == '(MUL t-s-t)'
    assert recode_to_code('utsat') == '(106)'
    assert recode_to_code('ut-sat') == '(EQ t-s-t)'
    assert recode_to_code('inumm') == '(MOD n-m-m)'
    assert recode_to_code('askp') == '(QUOT s-k-p)'
    assert recode_to_code('it-sp') == '(WITH t-s-p)'
    assert recode_to_code('itxs') == '(WITH t-x-s)'
    assert recode_to_code('itxx') == '(WITH t-x-x)'

    assert recode('tsc', HEBREW) == 'צח'
    assert recode('itst', HEBREW) == 'יצט'
    assert recode('it-st', HEBREW) == 'יטסט'
    assert recode('utasat', HEBREW) == 'וטאסאט'
    assert recode('atust', HEBREW) == 'אטוסט'
    assert recode('utsat', HEBREW) == 'וצאט'
    assert recode('ut-sat', HEBREW) == 'וטסאט'
    assert recode('inumm', HEBREW) == 'ינומם'
    assert recode('askp', HEBREW) == 'אסכף'
    assert recode('it-sp', HEBREW) == 'יטסף'

    # "S" and "X" both encode to "ס".
    assert recode('itxs', HEBREW) == 'יטסס'
    assert recode('itxx', HEBREW) == 'יטסס'

    script = HEBREW
    assert recode_to_code('צח') == '(98)'
    assert recode_to_code('יצט') == '(109)'
    assert recode_to_code('יטסט') == '(WITH ט-ס-ט)'
    assert recode_to_code('יצסט') == '(WITH צ-ס-ט)'
    assert recode_to_code('וטאסאט') == '(NOT ט-ס-ט)'
    assert recode_to_code('וצאסאט') == '(NOT צ-ס-ט)'
    assert recode_to_code('אטוסט') == '(MUL ט-ס-ט)'
    assert recode_to_code('אצוסט') == '(MUL צ-ס-ט)'
    assert recode_to_code('וצאט') == '(106)'
    assert recode_to_code('וטסאט') == '(EQ ט-ס-ט)'
    assert recode_to_code('וצסאט') == '(EQ צ-ס-ט)'
    assert recode_to_code('ינומם') == '(MOD נ-מ-ם)'
    assert recode_to_code('אסכף') == '(QUOT ס-כ-ף)'
    assert recode_to_code('יטסף') == '(WITH ט-ס-ף)'
    assert recode_to_code('יטסס') == '(WITH ט-ס-ס)'

    assert recode('צח', LATIN) == 'tsc'
    assert recode('יצט', LATIN) == 'itst'
    assert recode('יטסט', LATIN) == 'it-st'
    assert recode('יצסט', LATIN) == 'itsst'
    assert recode('וטאסאט', LATIN) == 'utasat'
    assert recode('וצאסאט', LATIN) == 'utsasat'
    assert recode('אטוסט', LATIN) == 'atust'
    assert recode('אצוסט', LATIN) == 'atsust'
    assert recode('וצאט', LATIN) == 'utsat'
    assert recode('וטסאט', LATIN) == 'ut-sat'
    assert recode('וצסאט', LATIN) == 'utssat'
    assert recode('ינומם', LATIN) == 'inumm'
    assert recode('אסכף', LATIN) == 'askp'
    assert recode('יטסף', LATIN) == 'it-sp'
    assert recode('יטסס', LATIN) == 'it-ss'


class State:
    def __init__(self, program):
        self.code = program
        self.wv = ''
        self.root = ''
        self.pc = 0
        self.vs = collections.defaultdict(str)

    def eval(self):
        while self.pc < len(self.code):
            word, self.pc = self.code[self.pc], self.pc + 1
            root, stem = unpack(word)
            op = OPS[stem]
            trace(f"{word=}: {None if op is None else op.__name__}({root}={self.vs[root]})")
            if op is not None:
                self.root = root
                op(self)
            if op is not with_:
                self.wv = ''

    def get(self):
        return self.vs[self.root]

    def get2(self):
        return self.vs[self.wv or self.root], self.vs[self.root]

    def set(self, x):
        self.vs[self.root] = x


def quot(state):
    state.set(state.code[state.pc])
    state.pc += 1


def clr(state):
    state.set('')


def with_(state):
    state.wv = state.root


def load(state):
    ww, _ = state.get2()
    state.set(ww)


def store(state):
    state.vs[state.wv or state.root] = state.get()


def swap(state):
    state.vs[state.root], state.vs[state.wv or state.root] = state.get2()


def cat(state):
    ww, cw = state.get2()
    if script is LATIN and cw[-1:].upper() == 'T' and ww[:1].upper() in {'S', 'Z'}:
        y = cw[:-1] + 'CA' + ww
    else:
        y = cw + ww
    assert gem(y) == gem(cw) + gem(ww), (cw, ww)
    state.set(y)


def sub(state):
    ww, cw = state.get2()
    state.set(degem(max(0, gem(cw) - gem(ww))))


def mul(state):
    ww, cw = state.get2()
    state.set(degem(gem(cw) * gem(ww)))


def div(state):
    ww, cw = state.get2()
    state.set(degem(gem(cw) // gem(ww)))


def mod(state):
    ww, cw = state.get2()
    state.set(degem(gem(cw) % gem(ww)))


def hop(state):
    word = state.get()
    n = gem(word)
    state.pc = max(0, state.pc - n)
    trace(f"-- hop {n=}: {state.pc=}")


def skip(state):
    word = state.get()
    n = gem(word)
    state.pc += n
    trace(f"-- skip {n=}: {state.pc=}")


def jump(state):
    word = state.get()
    n = gem(word)
    state.pc = n
    trace(f"-- jump {n=}")


def land(state):
    state.set(degem(state.pc))


def gt(state):
    ww, cw = state.get2()
    state.set(degem(1 if gem(cw) > gem(ww) else 0))


def lt(state):
    ww, cw = state.get2()
    state.set(degem(1 if gem(cw) < gem(ww) else 0))


def eq(state):
    ww, cw = state.get2()
    state.set(degem(1 if gem(cw) == gem(ww) else 0))


def neq(state):
    ww, cw = state.get2()
    state.set(degem(1 if gem(cw) != gem(ww) else 0))


def inc(state):
    cw = state.get()
    state.set(degem(gem(cw) + 1))


def dec(state):
    cw = state.get()
    state.set(degem(max(0, gem(state.get()) - 1)))


def not_(state):
    state.set(degem(0 if state.get() else 1))


def and_(state):
    ww, cw = state.get2()
    state.set(degem(1 if cw and ww else 0))


def peek(state):
    word = state.get()
    n = gem(word)
    trace(f"peek {n=}")
    state.set(state.code[n] if n < len(state.code) else '')


def poke(state):
    ww, cw = state.get2()
    n = gem(ww)
    if n < len(state.code):
        trace(f"poke {n=}: {state.code[n]}->{cw}")
        state.code[n] = cw
        state.set(degem(1))
    else:
        trace(f"failed poke {n=}")
        state.set(degem(0))


def rint(state):
    n = int(input())
    state.set(degem(n))


def wint(state):
    word = state.get()
    n = gem(word)
    print(n, flush=True)


def rword(state):
    word = input()
    state.set(word)


def wword(state):
    word = state.get()
    print(word, flush=True)


def rchar(state):
    n = sys.stdin.read(1)
    state.set(degem(ord(n) if n else 0))


def wchar(state):
    word = state.get()
    n = gem(word)
    sys.stdout.write(chr(n))
    sys.stdout.flush()


OPS = [
    None, quot, clr, with_,
    load, store, swap, cat,
    sub, mul, div, mod,
    hop, skip, jump, land,
    gt, lt, eq, neq,
    inc, dec, not_, and_,
    peek, poke, rint, wint,
    rword, wword, rchar, wchar,
] + [None, None, None, None] * 8


def recode_to_code_p(program):
    out = sys.stdout
    line = []
    for word in program:
        w = recode_to_code(word)
        if len(' '.join(line + [w])) > 120:
            out.write(' '.join(line) + '\n')
            line = []
        line += [w]
    out.write(' '.join(line) + '\n')


def justify_with_spaces(ww, linelen):
    x = linelen - len(' '.join(ww))
    q, r = divmod(x, len(ww) - 1)
    def spaces(i):
        return (0 if i == 0 else q + 1 if i <= (r + 1) else q)
    return [' ' * spaces(i) + w for i, w in enumerate(ww)]


def justify_with_tatwil(ww, linelen):
    # This function is extremely incorrect; it doesn't take word breaks into account,
    # doesn't prioritize the places to add tatwil, etc.
    # See https://www.khtt.net/en/page/1821/the-big-kashida-secret
    unjustified_line = ' '.join(ww)
    x = linelen - len(unjustified_line)
    placements = set([
        # 2. after a non-final SEEN or SAD
        i + 1 for i, c in enumerate(unjustified_line) if c in ['س', 'ص']
    ] + [
        # 3. before a final TEH MARBUTA, HAH, DAL
        i for i, c in enumerate(unjustified_line) if c in ['ﺔ', 'ﻪ', 'ﺪ'] or c in ['ة', 'ح', 'د']
    ] + [
        # 4. before a final ALEF, TEH, LAM, KAF, QAF
        i for i, c in enumerate(unjustified_line) if c in ['ﺎ', 'ـت', 'ﻞ', 'ﻚ', 'ﻖ'] or c in ['ا', 'ت', 'ل', 'ك', 'ق']
    ] + [
        # 5. before the preceding medial BEH of REH, YEH, ALEF MAKSURA
        i for i, c in enumerate(unjustified_line) if c in ['', '', '']
    ] + [
        # 6. before final WAW, AIN, QAF, FEH
        i for i, c in enumerate(unjustified_line) if c in ['ﻮ', 'ﻊ', 'ﻖ', 'ﻒ'] or c in ['و', 'ع', 'ق', 'ف']
    ] + [
        # 7. before final form of other characters that can be connected.
    ])
    placements -= set([0, len(unjustified_line)])
    placements = sorted(placements)
    indices = zip([0] + placements, placements + [len(unjustified_line)])
    ww = [unjustified_line[i:j] for i, j in indices]
    q, r = divmod(x, len(ww) - 1)
    justified_line = ''.join([w + 'ـ' * (q + 1 if i <= r else q) for i, w in enumerate(ww)])
    return justified_line.split(' ')


def recode_p(program, base):
    to = {'arabic': ARABIC, 'hebrew': HEBREW, 'latin': LATIN}[args.recode]
    justify_line = justify_with_tatwil if (to is ARABIC) else justify_with_spaces
    if args.justify is False:
        justify_line = (lambda ww, _: ww)
    with open(base + to['ext'], 'w') as out:
        line = []
        for word in program:
            w = recode(word, to)
            if len(' '.join(line + [w])) > 120:
                out.write(' '.join(justify_line(line, 120)) + '\n')
                line = []
            line += [w]
        out.write(' '.join(line) + '\n')


def run(path):
    global script
    base, ext = os.path.splitext(path)
    script = {s['ext']: s for s in (ARABIC, HEBREW, LATIN)}[ext]
    program = parse(path)
    if args.recode:
        recode_p(program, base)
    elif args.show:
        recode_to_code_p(program)
    else:
        State(program).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--show", action='store_true')
    group.add_argument("--recode", choices=['arabic', 'hebrew', 'latin'])
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--justify", action=argparse.BooleanOptionalAction)
    parser.parse_args(namespace=args)
    run(args.script)


if __name__ == "__main__":
    test_recode()
    main()
