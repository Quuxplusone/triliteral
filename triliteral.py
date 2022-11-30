import argparse
import collections
import dataclasses
import itertools
import os.path
import re
import sys


args = argparse.Namespace()
returns = []
script = None
LATIN = dict(
    values=dict(
        A=1, B=2, G=3, D=4, H=5, E=5, U=6, V=6, W=6, Z=7, C=8, Ch=8, T=9,
        I=10, J=10, Y=10, K=20, L=30, M=40, N=50, S=60, X=60, O=70, F=80, P=80, Ph=80, Ts=90, Tz=90,
        Q=100, R=200, Sh=300, Th=400, K_=500, M_=600, N_=700, P_=800, Ph_=800, Ts_=900, Tz_=900,
    ),
    vowels={'': 0, 'A': 1, 'E': 1, 'U': 2, 'I': 3},
    ext='.trl',
)
ARABIC=dict(
    values=dict(
        ا=1, ب=2, ج=3, د=4, ه=5, و=6, ز=7, ح=8, ط=9,
        ي=10, ك=20, ل=30, م=40, ن=50, س=60, ع=70, ف=80, ص=90,
        ق=100, ر=200, ش=300, ت=400, ث=500, خ=600, ذ=700, ض=800, ظ=900, غ=1000,
    ),
    vowels={'': 0, 'ا': 1, 'و': 2, 'ي': 3},
    ext='.طرل',
)
HEBREW=dict(
    values=dict(
        א=1, ב=2, ג=3, ד=4, ה=5, ו=6, ז=7, ח=8, ט=9,
        י=10, כ=20, ל=30, מ=40, נ=50, ס=60, ע=70, פ=80, צ=90,
        ק=100, ר=200, ש=300, ת=400, ך=500, ם=600, ן=700, ף=800, ץ=900,
    ),
    vowels={'': 0, 'א': 1, 'ה': 1, 'ו': 2, 'י': 3},
    ext='.טרל',
)


@dataclasses.dataclass
class Return:
    home: list
    code: list
    pc: int


def parse(path):
    words = []
    letters_re = r'[^' + ''.join(k for k in script['values'].keys() if len(k) == 1) + r']+'
    with open(path, 'r') as f:
        for line in f:
            words += re.split(letters_re, line.strip(), flags=re.IGNORECASE)
    return words


def unpack(word):
    vowels = script['vowels']
    vowels_re = r'([^' + ''.join(vowels.keys()) + r'])'
    parts = re.split(vowels_re, word.upper())[:6]
    root = "".join(parts[1::2])
    stem = int("".join(str(vowels.get(c[:1], c[:1])) for c in parts[4::-2]), 4)
    return root, stem


def gem(word):
    acc = 0
    word = word.upper() + '_'
    values = script['values']
    while word and word != '_':
        for i in 3, 2, 1:
            c = word[:i].title()
            if c in values:
                break
        word = word[i:]
        acc += values[c]
    return acc


def degem(n, scr=None):
    if scr is None:
        scr = script
    sv, sw = script['values'], script['vowels']
    vowels, consonants = [], []
    while n > 0:
        for c, v in reversed(sv.items()):
            if (c[-1] != '_' or not consonants) and (c + '_' not in sv or consonants) and v <= n:
                n -= v
                (vowels if c in sw else consonants)[:0] = (c[:-1] if c[-1] == '_' else c)
    return ''.join(itertools.chain.from_iterable(itertools.zip_longest(vowels, consonants, fillvalue='')))


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
        x = sv[c]
        y = sw.get(c, 0)
        try:
            out += next(k for k, v in tv.items() if v == x and tw.get(k, 0) == y)
        except StopIteration:
            raise Exception(f"can't recode '{c}' to {args.recode}")
    return out


def quot(stack):
    r = returns[-1]
    word, r.pc = r.code[r.pc] if r.pc < len(r.code) else '', r.pc + 1
    stack.append(word)


def nquot(stack):
    word = stack.pop() if stack else ''
    n = gem(word)
    for _ in range(n):
        quot(stack)
    if args.trace:
        print(f"-- nquot {n=}: {stack[-n:]}")


def dup(stack):
    stack.append(stack[-1] if stack else '')


def drop(stack):
    if stack:
        stack.pop()


def swap(stack):
    pq = stack[-2:]
    pq[:0] = [''] * (2 - len(pq))
    p, q = pq
    stack[-2:] = [q, p]


def rot(stack):
    pqr = stack[-3:]
    pqr[:0] = [''] * (3 - len(pqr))
    p, q, r = pqr
    stack[-3:] = [q, r, p]


def pick(stack):
    word = stack.pop() if stack else ''
    n = gem(word)
    stack.append(word if n == 0 else stack[-n] if len(stack) >= n else '')


def poke(stack):
    word = stack.pop() if stack else ''
    n = gem(word)
    if stack:
        stack[-n] = stack[-1]


def push(stack):
    home = returns[-1].home
    stack.append(home.pop() if home else '')


def pull(stack):
    home = returns[-1].home
    home.append(stack.pop() if stack else '')


def add(stack):
    w = stack.pop() if stack else ''
    x = stack.pop() if stack else ''
    stack.append(degem(gem(w) + gem(x)))


def sub(stack):
    w = stack.pop() if stack else ''
    x = stack.pop() if stack else ''
    y = degem(gem(w) - gem(x))
    assert gem(y) == max(0, gem(w) - gem(x)), (w, x, y)
    stack.append(y)


def mult(stack):
    w = stack.pop() if stack else ''
    x = stack.pop() if stack else ''
    stack.append(degem(gem(w) * gem(x)))


def div(stack):
    w = stack.pop() if stack else ''
    x = stack.pop() if stack else ''
    stack.append(degem(gem(w) // gem(x)))


def gt(stack):
    w = stack.pop() if stack else ''
    x = stack.pop() if stack else ''
    stack.append(degem(1 if gem(w) > gem(x) else 0))


def eq(stack):
    w = stack.pop() if stack else ''
    x = stack.pop() if stack else ''
    stack.append(degem(1 if gem(w) == gem(x) else 0))


def not_(stack):
    w = stack.pop() if stack else ''
    stack.append(degem(1 if gem(w) == 0 else 0))


def mark(stack):
    returns[-1].home = stack


def call(stack):
    returns.append(Return(returns[-1].home, stack, 0))


def skip(stack):
    word = stack.pop() if stack else ''
    n = gem(word)
    r = returns[-1]
    r.pc += n
    if args.trace:
        print(f"-- skip {n=}: {r.pc=}")


def hop(stack):
    word = stack.pop() if stack else ''
    n = gem(word)
    r = returns[-1]
    r.pc = max(0, r.pc - n)
    if args.trace:
        print(f"-- hop {n=}: {r.pc=}")


def split(stack):
    word = stack.pop() if stack else ''
    stack.extend(word)
    stack.append(degem(len(word)))


def splat(stack):
    word = stack.pop() if stack else ''
    n = gem(word)
    if n == 0:
        cc = []
    else:
        cc, stack[-n:] = stack[-n:], []
    stack.append(''.join(c[:1] for c in cc))


def rint(stack):
    n = int(input())
    stack.append(degem(n))


def wint(stack):
    word = stack.pop() if stack else ''
    n = gem(word)
    print(n)


def rword(stack):
    word = input()
    stack.append(word)


def wword(stack):
    word = stack.pop() if stack else ''
    print(word)


def rchar(stack):
    n = sys.stdin.read(1)
    stack.append(degem(ord(n) if n else 0))


def wchar(stack):
    word = stack.pop() if stack else ''
    n = gem(word)
    sys.stdout.write(chr(n))


OPS = [
    None, quot, nquot, None,
    dup, drop, swap, rot,
    pick, poke, push, pull,
    add, sub, mult, div,
    gt, eq, not_, None,
    mark, call, skip, hop,
    split, splat, rint, wint,
    rword, wword, rchar, wchar,
] + [None, None, None, None] * 8


def run(path):
    global script
    base, ext = os.path.splitext(path)
    script = {s['ext']: s for s in (ARABIC, HEBREW, LATIN)}[ext]
    program = parse(path)
    if args.recode:
        to = {'arabic': ARABIC, 'hebrew': HEBREW, 'latin': LATIN}[args.recode]
        out = open(base + to['ext'], 'w')
        for word in program:
            print(recode(word, to), file=out)
        return
    stacks = collections.defaultdict(list)
    returns.append(Return(program, program, 0))
    while returns:
        r = returns[-1]
        if r.pc >= len(r.code):
            returns.pop()
        else:
            word, r.pc = r.code[r.pc], r.pc + 1
            root, stem = unpack(word)
            op = OPS[stem]
            stack = stacks[root]
            if stack == r.code:
                root, stack = None, r.home
            if args.trace:
                print(f"{word=}: {None if op is None else op.__name__}({root}={stack})")
            if op is not None:
                op(stack)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--recode", choices=['arabic', 'hebrew', 'latin'])
    parser.add_argument("--trace", action='store_true')
    parser.parse_args(namespace=args)
    run(args.script)


if __name__ == "__main__":
    main()