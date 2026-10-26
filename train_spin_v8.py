import argparse
import itertools
import os
import random
import time
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

START_STACK = 500
TOTAL_CHIPS = 1500
HANDS_PER_LEVEL = 14
BLINDS = [(10, 20), (15, 30), (20, 40), (25, 50), (30, 60),
          (40, 80), (50, 100), (60, 120), (80, 160), (100, 200)]
N_ACT = 7
RAISE_FRAC = {3: 0.5, 4: 1.0, 5: 2.0}
MAX_TOKENS = 48
TOK_DIM = 15
SCALAR_DIM = 41
MEM_TOKENS = 32
MEM_DIM = 29
MEM_OPP = 13
EFF_BINS = [4.0, 6.0, 8.0, 10.0, 13.0, 17.0, 22.0]

_COMBOS = list(itertools.combinations(range(7), 5))


def _rank5(cards):
    rs = [c >> 2 for c in cards]
    ss = [c & 3 for c in cards]
    cnt = Counter(rs)
    groups = sorted(cnt.items(), key=lambda kv: (-kv[1], -kv[0]))
    flush = len(set(ss)) == 1
    u = sorted(cnt, reverse=True)
    st = -1
    if len(u) == 5:
        if u[0] - u[4] == 4:
            st = u[0]
        elif u == [12, 3, 2, 1, 0]:
            st = 3
    if flush and st >= 0:
        cat, tb = 8, [st]
    elif groups[0][1] == 4:
        cat, tb = 7, [groups[0][0], groups[1][0]]
    elif groups[0][1] == 3 and groups[1][1] == 2:
        cat, tb = 6, [groups[0][0], groups[1][0]]
    elif flush:
        cat, tb = 5, sorted(rs, reverse=True)
    elif st >= 0:
        cat, tb = 4, [st]
    elif groups[0][1] == 3:
        cat, tb = 3, [groups[0][0]] + sorted((g[0] for g in groups[1:]), reverse=True)
    elif groups[0][1] == 2 and groups[1][1] == 2:
        hi, lo = max(groups[0][0], groups[1][0]), min(groups[0][0], groups[1][0])
        cat, tb = 2, [hi, lo, groups[2][0]]
    elif groups[0][1] == 2:
        cat, tb = 1, [groups[0][0]] + sorted((g[0] for g in groups[1:]), reverse=True)
    else:
        cat, tb = 0, sorted(rs, reverse=True)
    tb = (tb + [0, 0, 0, 0, 0])[:5]
    v = cat
    for x in tb:
        v = v * 13 + x
    return v


try:
    from phevaluator import evaluate_cards as _phe
except ImportError:
    _phe = None


def eval7_py(cards):
    return max(_rank5([cards[i] for i in c]) for c in _COMBOS)


def eval7(cards):
    if _phe is None:
        return eval7_py(cards)
    return -_phe(*cards)


ARCH = ["random", "aggro", "nit", "station", "tag"]
ARCH_TRAIN = ["aggro", "nit", "tag"]


def sample_params(rng):
    u = rng.random()
    if u < 0.03:
        return (1.05, 1.05, 0.0, 0.01, 0.0)
    if u < 0.06:
        return (0.0, 0.0, 1.0, 0.01, 1.0)
    return (rng.betavariate(2.0, 2.0) * 1.05,
            0.05 + rng.betavariate(2.0, 2.0) * 1.00,
            rng.betavariate(1.5, 4.0),
            0.01 + rng.betavariate(1.5, 6.0) * 0.49,
            rng.random())


def new_table(rng, n_pool=0, p_self=1.0, exploit=False, p_frozen=0.5):
    t = {"stacks": [START_STACK] * 3, "button": rng.randrange(3),
         "hands": 0, "hand": None, "winner": None, "seq": {},
         "mem": [deque(maxlen=MEM_TOKENS - 1) for _ in range(3)]}
    if exploit:
        hero = rng.randrange(3)
        t["roles"] = [("learn", None) if i == hero else ("frozen", 0)
                      for i in range(3)]
    elif rng.random() < p_self:
        t["roles"] = [("learn", None)] * 3
    else:
        hero = rng.randrange(3)
        roles = []
        for i in range(3):
            if i == hero:
                roles.append(("learn", None))
            elif n_pool and rng.random() < p_frozen:
                roles.append(("frozen", rng.randrange(n_pool)))
            else:
                roles.append(("param", sample_params(rng)))
        t["roles"] = roles
    return t


def alive(t):
    return [i for i in range(3) if t["stacks"][i] > 0]


def blinds_of(t):
    lv = min(t["hands"] // HANDS_PER_LEVEL, len(BLINDS) - 1)
    return BLINDS[lv], lv


def start_hand(t, rng):
    al = alive(t)
    n = len(al)
    while t["button"] not in al:
        t["button"] = (t["button"] + 1) % 3
    k = al.index(t["button"])
    order = [al[(k + j) % n] for j in range(n)]
    (sb_amt, bb_amt), lv = blinds_of(t)
    if n == 3:
        sb, bb, first = order[1], order[2], order[0]
    else:
        sb, bb, first = order[0], order[1], order[0]
    deck = rng.sample(range(52), 2 * n + 5)
    h = {"players": al, "order": order, "hole": {}, "board": deck[2 * n:],
         "street": 0, "committed": [0, 0, 0], "bet": [0, 0, 0],
         "folded": [False] * 3, "allin": [False] * 3, "log": [],
         "sb": sb, "bb": bb, "bb_amt": bb_amt, "level": lv,
         "min_raise": bb_amt, "acted": set(), "to_act": first,
         "dec": {}, "start_stacks": list(t["stacks"])}
    for j, i in enumerate(al):
        h["hole"][i] = [deck[2 * j], deck[2 * j + 1]]
    for seat, amt in ((sb, sb_amt), (bb, bb_amt)):
        p = min(amt, t["stacks"][seat])
        t["stacks"][seat] -= p
        h["bet"][seat] = p
        h["committed"][seat] = p
        if t["stacks"][seat] == 0:
            h["allin"][seat] = True
    h["cur"] = max(h["bet"])
    t["hand"] = h
    return h


def hand_strength(h, seat):
    vis = {0: 0, 1: 3, 2: 4, 3: 5}[h["street"]]
    hole = h["hole"][seat]
    if vis == 0 or _phe is None:
        r = sorted((c >> 2 for c in hole), reverse=True)
        s = r[0] * 0.045 + r[1] * 0.02
        if r[0] == r[1]:
            s += 0.35
        if (hole[0] & 3) == (hole[1] & 3):
            s += 0.08
        if 0 < r[0] - r[1] <= 2:
            s += 0.05
        return min(s, 1.0)
    return 1.0 - _phe(*(hole + h["board"][:vis])) / 7462.0


def param_action(pr, h, seat, mask, rng):
    legal = [i for i in range(N_ACT) if mask[i] > 0]
    tight, aggr, bluff, noise, size = pr
    if rng.random() < noise:
        return rng.choice(legal)
    s = hand_strength(h, seat)
    raises = [i for i in (2, 3, 4, 5, 6) if mask[i] > 0]
    if raises and (s > aggr or rng.random() < bluff):
        return raises[min(int(size * len(raises)), len(raises) - 1)]
    if h["cur"] - h["bet"][seat] == 0:
        return 1
    if s > tight:
        return 1
    return 0 if mask[0] > 0 else 1


def arch_action(kind, h, seat, mask, rng):
    legal = [i for i in range(N_ACT) if mask[i] > 0]
    if kind == "random" or rng.random() < 0.05:
        return rng.choice(legal)
    if kind == "station":
        return 1
    s = hand_strength(h, seat)
    to_call = h["cur"] - h["bet"][seat]
    raises = [i for i in (4, 5, 6, 3, 2) if mask[i] > 0]
    if kind == "tag":
        if s > 0.55 and raises:
            return raises[0]
        if to_call == 0:
            return 1
        if s > 0.45:
            return 1
        return 0 if mask[0] > 0 else 1
    if kind == "aggro":
        if s > 0.30 and raises:
            return raises[0] if rng.random() < 0.7 else rng.choice(raises)
        if to_call == 0 or rng.random() < 0.5:
            return 1
        return 0 if mask[0] > 0 else 1
    if s > 0.85 and raises:
        return rng.choice(raises)
    if to_call == 0:
        return 1
    if s > 0.60:
        return 1
    return 0 if mask[0] > 0 else 1


def contenders(h):
    return [i for i in h["players"] if not h["folded"][i]]


def actors(h):
    return [i for i in contenders(h) if not h["allin"][i]]


def round_closed(h):
    a = actors(h)
    if len(contenders(h)) <= 1:
        return True
    if not a:
        return True
    if len(a) == 1 and h["bet"][a[0]] >= h["cur"]:
        return True
    return all(i in h["acted"] and h["bet"][i] == h["cur"] for i in a)


def next_actor(h, seat):
    n = len(h["order"])
    k = h["order"].index(seat)
    for j in range(1, n + 1):
        c = h["order"][(k + j) % n]
        if not h["folded"][c] and not h["allin"][c]:
            return c
    return None


def first_postflop(h):
    ref = h["bb"] if len(h["players"]) == 2 else h["sb"]
    if ref not in h["players"]:
        ref = h["order"][0]
    if not h["folded"][ref] and not h["allin"][ref]:
        return ref
    return next_actor(h, ref)


def legal_mask(t, h, seat):
    m = np.zeros(N_ACT, dtype=np.float32)
    stack = t["stacks"][seat]
    to_call = h["cur"] - h["bet"][seat]
    m[1] = 1.0
    if to_call > 0:
        m[0] = 1.0
    if stack > to_call:
        m[6] = 1.0
        pot = sum(h["committed"])
        min_to = h["cur"] + h["min_raise"]
        maxi = stack + h["bet"][seat]
        seen = {maxi}
        if maxi > min_to:
            m[2] = 1.0
            seen.add(min_to)
        for a, f in RAISE_FRAC.items():
            tgt = h["cur"] + int(f * (pot + to_call))
            if min_to <= tgt < maxi and tgt not in seen:
                m[a] = 1.0
                seen.add(tgt)
    return m


def apply_action(t, h, seat, a):
    stack = t["stacks"][seat]
    to_call = h["cur"] - h["bet"][seat]
    pot = sum(h["committed"])
    if a == 0:
        h["folded"][seat] = True
        size = 0.0
    elif a == 1:
        p = min(to_call, stack)
        t["stacks"][seat] -= p
        h["bet"][seat] += p
        h["committed"][seat] += p
        if t["stacks"][seat] == 0:
            h["allin"][seat] = True
        size = p / max(pot, 1)
    else:
        if a == 6:
            tgt = stack + h["bet"][seat]
        elif a == 2:
            tgt = h["cur"] + h["min_raise"]
        else:
            tgt = h["cur"] + int(RAISE_FRAC[a] * (pot + to_call))
        tgt = min(tgt, stack + h["bet"][seat])
        p = tgt - h["bet"][seat]
        inc = tgt - h["cur"]
        if inc >= h["min_raise"]:
            h["min_raise"] = inc
            h["acted"] = set()
        t["stacks"][seat] -= p
        h["bet"][seat] = tgt
        h["committed"][seat] += p
        h["cur"] = max(h["cur"], tgt)
        if t["stacks"][seat] == 0:
            h["allin"][seat] = True
        size = p / max(pot, 1)
    h["acted"].add(seat)
    h["log"].append((seat, a, min(size, 4.0), h["street"]))


def advance(t, h):
    while round_closed(h):
        if len(contenders(h)) <= 1 or h["street"] == 3:
            return True
        if len(actors(h)) <= 1:
            h["street"] = 3
            return True
        h["street"] += 1
        h["bet"] = [0, 0, 0]
        h["cur"] = 0
        h["min_raise"] = h["bb_amt"]
        h["acted"] = set()
        h["to_act"] = first_postflop(h)
        if h["to_act"] is None:
            return True
    return False


def settle(t, h):
    con = contenders(h)
    if len(con) == 1:
        t["stacks"][con[0]] += sum(h["committed"])
        return
    board = h["board"]
    scores = {i: eval7(h["hole"][i] + board) for i in con}
    levels = sorted({h["committed"][i] for i in h["players"] if h["committed"][i] > 0})
    prev = 0
    for lv in levels:
        pot = sum(max(0, min(h["committed"][i], lv) - prev) for i in h["players"])
        elig = [i for i in con if h["committed"][i] >= lv]
        prev = lv
        if not elig or pot == 0:
            continue
        best = max(scores[i] for i in elig)
        wins = [i for i in elig if scores[i] == best]
        share = pot // len(wins)
        for i in wins:
            t["stacks"][i] += share
        t["stacks"][wins[0]] += pot - share * len(wins)


def end_hand(t):
    h = t["hand"]
    settle(t, h)
    t["hands"] += 1
    t["button"] = (t["button"] + 1) % 3
    t["hand"] = None
    al = alive(t)
    if len(al) == 1:
        t["winner"] = al[0]
    return h


def encode(t, h, seat):
    planes = np.zeros((6, 4, 13), dtype=np.float32)
    for c in h["hole"][seat]:
        planes[0, c & 3, c >> 2] = 1.0
    vis = {0: 0, 1: 3, 2: 4, 3: 5}[h["street"]]
    for k, c in enumerate(h["board"][:vis]):
        idx = 1 if k < 3 else (2 if k == 3 else 3)
        planes[idx, c & 3, c >> 2] = 1.0
        planes[4, c & 3, c >> 2] = 1.0
    for c in h["hole"][seat] + h["board"][:vis]:
        planes[5, c & 3, c >> 2] = 1.0

    toks = np.zeros((MAX_TOKENS, TOK_DIM), dtype=np.float32)
    mask = np.ones(MAX_TOKENS, dtype=bool)
    mask[0] = False
    for k, (s, a, size, st) in enumerate(h["log"][-(MAX_TOKENS - 1):], start=1):
        toks[k, (s - seat) % 3] = 1.0
        toks[k, 3 + a] = 1.0
        toks[k, 10] = size
        toks[k, 11 + st] = 1.0
        mask[k] = False

    mem = np.zeros((MEM_TOKENS, MEM_DIM), dtype=np.float32)
    mmask = np.ones(MEM_TOKENS, dtype=bool)
    mmask[0] = False
    hist = list(t["mem"][seat])[-(MEM_TOKENS - 1):]
    for k, v in enumerate(hist, start=1):
        mem[k] = v
        mmask[k] = False
    stats = [0.0] * 10
    if hist:
        arr = np.stack(hist)
        for oi in range(2):
            b = oi * MEM_OPP
            sh = arr[:, b + 12]
            stats[oi * 5] = min(len(hist) / 16.0, 1.0)
            stats[oi * 5 + 1] = float(arr[:, b + 7].mean())
            stats[oi * 5 + 2] = float(arr[:, b + 10].mean())
            stats[oi * 5 + 3] = float(arr[:, b + 9].mean())
            if sh.sum() > 0:
                stats[oi * 5 + 4] = float((arr[:, b + 11] * sh).sum() / sh.sum())

    bb = h["bb_amt"]
    others = [(seat + j) % 3 for j in (1, 2)]
    pot = sum(h["committed"])
    to_call = h["cur"] - h["bet"][seat]
    eff = min(t["stacks"][seat], max(t["stacks"][i] for i in contenders(h) if i != seat)) \
        if len(contenders(h)) > 1 else t["stacks"][seat]
    eff_bb = eff / bb
    lv = h["level"]
    sb_now, bb_now = BLINDS[lv]
    sb_nx, bb_nx = BLINDS[min(lv + 1, len(BLINDS) - 1)]
    left = 1.0 if lv == len(BLINDS) - 1 else (
        (HANDS_PER_LEVEL - t["hands"] % HANDS_PER_LEVEL) / HANDS_PER_LEVEL)
    own0 = h["start_stacks"][seat]
    m_now = min(own0 / (sb_now + bb_now) / 20.0, 1.0)
    m_next = min(own0 / (sb_nx + bb_nx) / 20.0, 1.0)
    bins = [0.0] * (len(EFF_BINS) + 1)
    k = 0
    while k < len(EFF_BINS) and eff_bb >= EFF_BINS[k]:
        k += 1
    bins[k] = 1.0
    sc = np.array([
        t["stacks"][seat] / bb / 25, eff_bb / 25,
        t["stacks"][others[0]] / bb / 25, t["stacks"][others[1]] / bb / 25,
        pot / bb / 25, to_call / bb / 25, h["cur"] / bb / 25,
        h["bet"][others[0]] / bb / 25, h["bet"][others[1]] / bb / 25,
        t["stacks"][seat] / TOTAL_CHIPS, len(h["players"]) / 3,
        1.0 if len(h["players"]) == 2 else 0.0,
        1.0 if seat == h["order"][0] else 0.0,
        1.0 if seat == h["sb"] else 0.0,
        1.0 if seat == h["bb"] else 0.0,
        h["level"] / len(BLINDS), left, m_now, m_next,
        *[1.0 if h["street"] == s else 0.0 for s in range(4)],
        *bins, *stats,
    ], dtype=np.float32)
    return planes, toks, mask, mem, mmask, sc


class Net(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 4 * 13, d), nn.ReLU())
        self.tok = nn.Linear(TOK_DIM, 128)
        self.pos = nn.Parameter(torch.zeros(1, MAX_TOKENS, 128))
        layer = nn.TransformerEncoderLayer(128, 4, 256, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.act_proj = nn.Linear(128, d)
        self.scal = nn.Sequential(nn.Linear(SCALAR_DIM, 128), nn.ReLU(),
                                  nn.Linear(128, 128), nn.ReLU())
        self.mtok = nn.Linear(MEM_DIM, 64)
        self.mpos = nn.Parameter(torch.zeros(1, MEM_TOKENS, 64))
        mlayer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True,
                                            norm_first=True)
        self.menc = nn.TransformerEncoder(mlayer, 2, enable_nested_tensor=False)
        self.mem_proj = nn.Linear(64, 128)
        self.fuse = nn.Sequential(nn.Linear(2 * d + 256, d), nn.ReLU())
        self.pi = nn.Linear(d, N_ACT)
        self.v = nn.Linear(d, 1)
        self.win = nn.Linear(d, 1)
        self.eq = nn.Linear(d, 1)
        self.opp = nn.Linear(d, 52)

    def forward(self, planes, toks, mask, mem, mmask, sc):
        c = self.conv(planes)
        x = self.tok(toks) + self.pos
        x = self.enc(x, src_key_padding_mask=mask)
        keep = (~mask).unsqueeze(-1).float()
        x = (x * keep).sum(1) / keep.sum(1).clamp(min=1)
        a = F.relu(self.act_proj(x))
        y = self.mtok(mem) + self.mpos
        y = self.menc(y, src_key_padding_mask=mmask)
        k2 = (~mmask).unsqueeze(-1).float()
        y = (y * k2).sum(1) / k2.sum(1).clamp(min=1)
        mm = F.relu(self.mem_proj(y))
        z = self.fuse(torch.cat([c, a, mm, self.scal(sc)], -1))
        return (self.pi(z), self.v(z).squeeze(-1), self.win(z).squeeze(-1),
                self.eq(z).squeeze(-1), self.opp(z))


def to_batch(obs, device):
    p = torch.from_numpy(np.stack([o[0] for o in obs])).to(device, non_blocking=True)
    tk = torch.from_numpy(np.stack([o[1] for o in obs])).to(device, non_blocking=True)
    mk = torch.from_numpy(np.stack([o[2] for o in obs])).to(device, non_blocking=True)
    mm = torch.from_numpy(np.stack([o[3] for o in obs])).to(device, non_blocking=True)
    mmk = torch.from_numpy(np.stack([o[4] for o in obs])).to(device, non_blocking=True)
    sc = torch.from_numpy(np.stack([o[5] for o in obs])).to(device, non_blocking=True)
    return p, tk, mk, mm, mmk, sc


def proceed(t, h, seat):
    if round_closed(h):
        return advance(t, h)
    nxt = next_actor(h, seat)
    if nxt is None:
        return advance(t, h)
    h["to_act"] = nxt
    return False


def hand_token(t, h, seat, before, show):
    v = np.zeros(MEM_DIM, dtype=np.float32)
    bb = h["bb_amt"]
    for oi, o in enumerate(((seat + 1) % 3, (seat + 2) % 3)):
        b = oi * MEM_OPP
        acts = [a for (sq, a, _, _) in h["log"] if sq == o]
        if acts:
            for a in acts:
                v[b + a] += 1.0
            v[b:b + 7] /= len(acts)
            v[b + 10] = 1.0 if any(a >= 2 for a in acts) else 0.0
        v[b + 7] = 1.0 if h["folded"][o] else 0.0
        v[b + 8] = 1.0 if (o in h["players"] and not h["folded"][o]
                           and h["street"] == 3) else 0.0
        v[b + 9] = float(np.clip((t["stacks"][o] - before[o]) / (10 * bb),
                                 -1.0, 1.0))
        if o in show:
            v[b + 11] = show[o]
            v[b + 12] = 1.0
    v[26] = h["street"] / 3.0
    v[27] = min(sum(h["committed"]) / (20.0 * bb), 1.0)
    v[28] = float(np.clip((t["stacks"][seat] - before[seat]) / (10 * bb),
                          -1.0, 1.0))
    return v


def close_hand(t, buf, r_win=0.0, r_bust=0.0):
    h = t["hand"]
    before = h["start_stacks"]
    con = [i for i in h["players"] if not h["folded"][i]]
    show = {}
    if len(con) >= 2:
        for i in con:
            show[i] = (1.0 - _phe(*(h["hole"][i] + h["board"])) / 7462.0
                       if _phe is not None else 0.5)
    end_hand(t)
    for seat in range(3):
        t["mem"][seat].append(hand_token(t, h, seat, before, show))
    if buf is None:
        return
    scores = {i: eval7(h["hole"][i] + h["board"]) for i in h["players"]}
    for seat, idxs in h["dec"].items():
        if not idxs:
            continue
        buf[idxs[-1]]["r"] += (t["stacks"][seat] - before[seat]) / START_STACK
        opp = [i for i in h["players"] if i != seat]
        top = max(scores[i] for i in opp)
        lab = 1.0 if scores[seat] > top else (0.5 if scores[seat] == top else 0.0)
        cards = tuple(c for i in opp for c in h["hole"][i])
        for j in idxs:
            buf[j]["eq"] = lab
            buf[j]["opp"] = cards
            buf[j]["has_eq"] = 1.0
    for seat, idxs in t["seq"].items():
        if not idxs:
            continue
        if seat in h["players"] and not h["dec"].get(seat):
            buf[idxs[-1]]["r"] += (t["stacks"][seat] - before[seat]) / START_STACK
        if before[seat] > 0 and t["stacks"][seat] == 0:
            buf[idxs[-1]]["r"] -= r_bust
        if t["winner"] == seat:
            buf[idxs[-1]]["r"] += r_win


def finish_tournament(t, buf, seqs):
    for seat, idxs in t["seq"].items():
        if not idxs:
            continue
        lab = 1.0 if seat == t["winner"] else 0.0
        for j in idxs:
            buf[j]["win"] = lab
            buf[j]["has_win"] = 1.0
        buf[idxs[-1]]["term"] = True
        seqs.append(idxs)


def truncate_open(tables, buf, seqs):
    for t in tables:
        if t["winner"] is not None:
            finish_tournament(t, buf, seqs)
            t["seq"] = {}
            continue
        for seat, idxs in t["seq"].items():
            if not idxs:
                continue
            share = t["stacks"][seat] / TOTAL_CHIPS
            for j in idxs:
                if buf[j]["has_win"] < 0.0:
                    buf[j]["win"] = share
                    buf[j]["has_win"] = 1.0
            seqs.append(idxs)
        t["seq"] = {}


def net_actions(model, entries, ks, device, amp):
    obs = [encode(entries[k][0], entries[k][1], entries[k][2]) for k in ks]
    p, tk, mk, mm, mmk, sc = to_batch(obs, device)
    msk = torch.from_numpy(np.stack([entries[k][3] for k in ks])).to(device)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
        logits, val = model(p, tk, mk, mm, mmk, sc)[:2]
    logits = logits.float().masked_fill(msk == 0, -1e9)
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    return (a.cpu().numpy(), dist.log_prob(a).cpu().numpy(),
            val.float().cpu().numpy(), obs)


def decide(entries, roles, net, frozen, device, amp, rng):
    out = [None] * len(entries)
    groups = {}
    for k, role in enumerate(roles):
        if role[0] == "arch":
            e = entries[k]
            out[k] = (arch_action(role[1], e[1], e[2], e[3], rng),
                      None, None, None)
        elif role[0] == "param":
            e = entries[k]
            out[k] = (param_action(role[1], e[1], e[2], e[3], rng),
                      None, None, None)
        else:
            groups.setdefault(role, []).append(k)
    for role, ks in groups.items():
        model = net if role[0] == "learn" else frozen[role[1]]
        a, lp, v, obs = net_actions(model, entries, ks, device, amp)
        for j, k in enumerate(ks):
            if role[0] == "learn":
                out[k] = (int(a[j]), float(lp[j]), float(v[j]), obs[j])
            else:
                out[k] = (int(a[j]), None, None, None)
    return out


def collect(net, frozen, tables, rng, device, n_steps, amp, args):
    buf = []
    seqs = []
    res = []
    for t in tables:
        t["seq"] = {}
        if t["hand"] is not None:
            t["hand"]["dec"] = {}
    t_env = t_fwd = 0.0
    collected = 0
    mix = {"self": 0, "league": 0}
    n_pool = len(frozen)
    while collected < n_steps:
        t0 = time.perf_counter()
        entries, roles, idx = [], [], []
        for ti, t in enumerate(tables):
            if t["winner"] is not None:
                for sq in t["seq"]:
                    res.append(int(sq == t["winner"]))
                finish_tournament(t, buf, seqs)
                tables[ti] = new_table(rng, n_pool, args.p_self,
                                       bool(args.exploit), args.p_frozen)
                continue
            if t["hand"] is None:
                start_hand(t, rng)
                if advance(t, t["hand"]):
                    close_hand(t, buf, args.r_win, args.r_bust)
                    continue
            h = t["hand"]
            seat = h["to_act"]
            entries.append((t, h, seat, legal_mask(t, h, seat)))
            roles.append(t["roles"][seat])
            idx.append(ti)
        t_env += time.perf_counter() - t0
        if not entries:
            continue

        t0 = time.perf_counter()
        out = decide(entries, roles, net, frozen, device, amp, rng)
        t_fwd += time.perf_counter() - t0

        t0 = time.perf_counter()
        for k, (t, h, seat, lm) in enumerate(entries):
            a, lp, v, obs = out[k]
            if roles[k][0] == "learn":
                buf.append({"obs": obs, "mask": lm, "a": a, "logp": lp, "v": v,
                            "r": 0.0, "win": 0.0, "has_win": -1.0,
                            "term": False, "eq": 0.0, "opp": (),
                            "has_eq": 0.0})
                j = len(buf) - 1
                h["dec"].setdefault(seat, []).append(j)
                t["seq"].setdefault(seat, []).append(j)
                collected += 1
                key = ("self" if all(r[0] == "learn" for r in t["roles"])
                       else "league")
                mix[key] += 1
            apply_action(t, h, seat, a)
            if proceed(t, h, seat):
                close_hand(t, buf, args.r_win, args.r_bust)
        t_env += time.perf_counter() - t0
    truncate_open(tables, buf, seqs)
    wr = float(np.mean(res)) if res else float("nan")
    tot = max(mix["self"] + mix["league"], 1)
    return buf, seqs, t_env, t_fwd, wr, mix["self"] / tot


def evaluate(net, device, amp, rng, n_games, pairs):
    res = {}
    for pair in pairs:
        tables = []
        for g in range(n_games):
            t = new_table(rng)
            hero = g % 3
            others = [i for i in range(3) if i != hero]
            roles = [None] * 3
            roles[hero] = ("learn", None)
            roles[others[0]] = ("arch", pair[0])
            roles[others[1]] = ("arch", pair[1])
            t["roles"] = roles
            t["hero"] = hero
            tables.append(t)
        wins = done = 0
        while done < n_games:
            entries, roles = [], []
            for t in tables:
                if t.get("fin"):
                    continue
                if t["winner"] is not None:
                    t["fin"] = True
                    done += 1
                    wins += int(t["winner"] == t["hero"])
                    continue
                if t["hand"] is None:
                    start_hand(t, rng)
                    if advance(t, t["hand"]):
                        close_hand(t, None)
                        continue
                h = t["hand"]
                seat = h["to_act"]
                entries.append((t, h, seat, legal_mask(t, h, seat)))
                roles.append(t["roles"][seat])
            if not entries:
                continue
            out = decide(entries, roles, net, [], device, amp, rng)
            for k, (t, h, seat, lm) in enumerate(entries):
                apply_action(t, h, seat, out[k][0])
                if proceed(t, h, seat):
                    close_hand(t, None)
        res["+".join(pair)] = wins / n_games
    return res


def versus(a_net, b_net, device, amp, rng, n_games):
    tables = []
    for g in range(n_games):
        t = new_table(rng)
        hero = g % 3
        t["roles"] = [("learn", None) if i == hero else ("frozen", 0)
                      for i in range(3)]
        t["hero"] = hero
        tables.append(t)
    wins = done = 0
    chips = []
    while done < n_games:
        entries, roles = [], []
        for t in tables:
            if t.get("fin"):
                continue
            if t["winner"] is not None:
                t["fin"] = True
                done += 1
                wins += int(t["winner"] == t["hero"])
                continue
            if t["hand"] is None:
                before = list(t["stacks"])
                start_hand(t, rng)
                t["before"] = before
                if advance(t, t["hand"]):
                    bb = t["hand"]["bb_amt"]
                    hero = t["hero"]
                    close_hand(t, None)
                    chips.append((t["stacks"][hero] - before[hero]) / bb)
                    continue
            h = t["hand"]
            seat = h["to_act"]
            entries.append((t, h, seat, legal_mask(t, h, seat)))
            roles.append(t["roles"][seat])
        if not entries:
            continue
        out = decide(entries, roles, a_net, [b_net], device, amp, rng)
        for k, (t, h, seat, lm) in enumerate(entries):
            apply_action(t, h, seat, out[k][0])
            if proceed(t, h, seat):
                bb = h["bb_amt"]
                hero = t["hero"]
                before = t["before"]
                close_hand(t, None)
                chips.append((t["stacks"][hero] - before[hero]) / bb)
    c = np.array(chips)
    return wins / n_games, float(c.mean() * 100), float(
        1.96 * c.std() / max(len(c), 1) ** 0.5 * 100)


def gae(buf, seqs, gamma=1.0, lam=0.95):
    adv = np.zeros(len(buf), dtype=np.float32)
    ret = np.zeros(len(buf), dtype=np.float32)
    keep = np.zeros(len(buf), dtype=bool)
    for idxs in seqs:
        if buf[idxs[-1]]["term"]:
            end, boot = len(idxs), 0.0
        else:
            end, boot = len(idxs) - 1, buf[idxs[-1]]["v"]
        last = 0.0
        for k in range(end - 1, -1, -1):
            j = idxs[k]
            nv = boot if k == end - 1 else buf[idxs[k + 1]]["v"]
            delta = buf[j]["r"] + gamma * nv - buf[j]["v"]
            last = delta + gamma * lam * last
            adv[j] = last
            ret[j] = last + buf[j]["v"]
            keep[j] = True
    return adv, ret, keep


def update(net, opt, buf, adv, ret, keep, device, args, amp):
    sel = [i for i in range(len(buf)) if keep[i]]
    n = len(sel)
    b = [buf[i] for i in sel]
    p, tk, mk, mm, mmk, sc = to_batch([x["obs"] for x in b], device)
    msk = torch.from_numpy(np.stack([x["mask"] for x in b])).to(device)
    a = torch.tensor([x["a"] for x in b], device=device)
    lp_old = torch.tensor([x["logp"] for x in b], device=device)
    v_old = torch.tensor([x["v"] for x in b], device=device)
    w = torch.tensor([max(x["win"], 0.0) for x in b], device=device)
    wm = torch.tensor([1.0 if x["has_win"] > 0 else 0.0 for x in b], device=device)
    eq = torch.tensor([x["eq"] for x in b], device=device)
    em = torch.tensor([x["has_eq"] for x in b], device=device)
    oc = np.zeros((n, 52), dtype=np.float32)
    for k, x in enumerate(b):
        for c in x["opp"]:
            oc[k, c] = 1.0
    oc = torch.from_numpy(oc).to(device)
    adv = torch.from_numpy(adv[sel]).to(device)
    ret = torch.from_numpy(ret[sel]).to(device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = None
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.minibatch):
            i = perm[s:s + args.minibatch]
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                logits, v, wl, eql, opl = net(p[i], tk[i], mk[i], mm[i],
                                              mmk[i], sc[i])
            logits = logits.float().masked_fill(msk[i] == 0, -1e9)
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(a[i])
            ratio = (lp - lp_old[i]).exp()
            ai = adv[i]
            pos = torch.min(ratio * ai, ratio.clamp(1 - args.clip, 1 + args.clip) * ai)
            neg = torch.max(ratio * ai, ratio.clamp(1 - args.clip, args.clip_neg) * ai)
            pl = -torch.where(ai >= 0, pos, neg).mean()
            vc = v_old[i] + (v.float() - v_old[i]).clamp(-args.clip_v, args.clip_v)
            vl = torch.max((v.float() - ret[i]) ** 2, (vc - ret[i]) ** 2).mean()
            wraw = F.binary_cross_entropy_with_logits(wl.float(), w[i],
                                                      reduction="none")
            wloss = (wraw * wm[i]).sum() / wm[i].sum().clamp(min=1)
            eraw = F.binary_cross_entropy_with_logits(eql.float(), eq[i],
                                                      reduction="none")
            eloss = (eraw * em[i]).sum() / em[i].sum().clamp(min=1)
            oraw = F.binary_cross_entropy_with_logits(opl.float(), oc[i],
                                                      reduction="none")
            oloss = (oraw.mean(-1) * em[i]).sum() / em[i].sum().clamp(min=1)
            ent = dist.entropy().mean()
            loss = (pl + args.vf * vl + args.aux * wloss - args.ent * ent
                    + args.eq * eloss + args.opp * oloss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats = (pl.item(), vl.item(), ent.item(), 0.0,
                     eloss.item(), oloss.item())

    i = torch.randperm(n, device=device)[:min(n, 4096)]
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
        lg = net(p[i], tk[i], mk[i], mm[i], mmk[i], sc[i])[0]
    lg = lg.float().masked_fill(msk[i] == 0, -1e9)
    lp = torch.distributions.Categorical(logits=lg).log_prob(a[i])
    d = lp - lp_old[i]
    kl = float(((d.exp() - 1.0) - d).mean())
    return stats[:3] + (kl,) + stats[4:]


def update_aux(net, opt, buf, device, args, amp):
    b = [x for x in buf if x["has_eq"] > 0]
    n = len(b)
    if n == 0:
        return 0.0, 0.0
    p, tk, mk, mm, mmk, sc = to_batch([x["obs"] for x in b], device)
    eq = torch.tensor([x["eq"] for x in b], device=device)
    oc = np.zeros((n, 52), dtype=np.float32)
    for k, x in enumerate(b):
        for c in x["opp"]:
            oc[k, c] = 1.0
    oc = torch.from_numpy(oc).to(device)
    el = ol = 0.0
    perm = torch.randperm(n, device=device)
    for s in range(0, n, args.minibatch):
        i = perm[s:s + args.minibatch]
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            out = net(p[i], tk[i], mk[i], mm[i], mmk[i], sc[i])
        eloss = F.binary_cross_entropy_with_logits(out[3].float(), eq[i])
        oloss = F.binary_cross_entropy_with_logits(out[4].float(), oc[i])
        loss = eloss + oloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        opt.step()
        el, ol = eloss.item(), oloss.item()
    return el, ol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--tables", type=int, default=512)
    ap.add_argument("--steps", type=int, default=16384)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--clip-neg", type=float, default=3.0)
    ap.add_argument("--clip-v", type=float, default=2.0)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--aux", type=float, default=0.3)
    ap.add_argument("--eq", type=float, default=0.5)
    ap.add_argument("--opp", type=float, default=0.5)
    ap.add_argument("--pretrain", type=int, default=0)
    ap.add_argument("--exploit", default="")
    ap.add_argument("--versus", nargs=2, default=None)
    ap.add_argument("--average", nargs="+", default=None)
    ap.add_argument("--roundrobin", nargs="+", default=None)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--versus-games", type=int, default=3000)
    ap.add_argument("--ent", type=float, default=0.02)
    ap.add_argument("--p-self", type=float, default=0.5)
    ap.add_argument("--p-frozen", type=float, default=0.5)
    ap.add_argument("--r-win", type=float, default=0.0)
    ap.add_argument("--r-bust", type=float, default=0.0)
    ap.add_argument("--pool-size", type=int, default=4)
    ap.add_argument("--pool-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    os.makedirs(args.out, exist_ok=True)

    if args.average:
        out_path, srcs = args.average[0], args.average[1:]
        sd = None
        for path in srcs:
            d = torch.load(path, map_location="cpu")["net"]
            if sd is None:
                sd = {k: v.clone().float() if v.dtype.is_floating_point
                      else v.clone() for k, v in d.items()}
            else:
                for k in sd:
                    if sd[k].dtype.is_floating_point:
                        sd[k] += d[k].float()
        for k in sd:
            if sd[k].dtype.is_floating_point:
                sd[k] /= len(srcs)
        torch.save({"net": sd, "it": -1}, out_path)
        print(f"moyenne de {len(srcs)} checkpoints ecrite dans {out_path}")
        return

    if args.roundrobin:
        nets = []
        for path in args.roundrobin:
            m = Net().to(device)
            m.load_state_dict(torch.load(path, map_location=device)["net"])
            m.eval()
            nets.append(m)
        names = [os.path.basename(p) for p in args.roundrobin]
        print("ligne = heros, colonne = paire adverse, equilibre 0.3333")
        for i in range(len(nets)):
            row = []
            for j in range(len(nets)):
                if i == j:
                    row.append(float("nan"))
                    continue
                wr, _, _ = versus(nets[i], nets[j], device, device == "cuda",
                                  rng, args.versus_games)
                row.append(wr)
            mean = float(np.nanmean(row))
            print(f"{names[i]:24s} " + " ".join(f"{x:.3f}" for x in row)
                  + f"  moyenne {mean:.3f}", flush=True)
        return

    if args.versus:
        a, b = Net().to(device), Net().to(device)
        a.load_state_dict(torch.load(args.versus[0], map_location=device)["net"])
        b.load_state_dict(torch.load(args.versus[1], map_location=device)["net"])
        a.eval()
        b.eval()
        wr, bb100, ci = versus(a, b, device, device == "cuda", rng,
                               args.versus_games)
        print(f"A {os.path.basename(args.versus[0])} vs 2x "
              f"{os.path.basename(args.versus[1])}")
        print(f"tournois gagnes par A {wr:.4f} (equilibre 0.3333)")
        print(f"jetons par 100 mains {bb100:+.2f} bb +/- {ci:.2f}")
        return

    net = Net().to(device)
    if args.exploit:
        opp_net = Net().to(device)
        opp_net.load_state_dict(torch.load(args.exploit,
                                           map_location=device)["net"])
        opp_net.eval()
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location=device)["net"])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ema = None
    if args.ema > 0:
        ema = Net().to(device)
        ema.load_state_dict(net.state_dict())
        ema.eval()
    frozen = [opp_net] if args.exploit else []
    pairs = [("station", "random"), ("aggro", "nit"), ("tag", "station")]
    tables = [new_table(rng, len(frozen), args.p_self, bool(args.exploit),
                        args.p_frozen) for _ in range(args.tables)]

    if args.pretrain:
        p_self = args.p_self
        args.p_self = 1.0
        pre = [new_table(rng, 0, 1.0) for _ in range(args.tables)]
        for it in range(args.pretrain):
            buf, _, _, _, _, _ = collect(net, [], pre, rng, device,
                                         args.steps, amp, args)
            el, ol = update_aux(net, opt, buf, device, args, amp)
            print(f"pre {it} n {len(buf)} eq {el:.4f} opp {ol:.4f}", flush=True)
        args.p_self = p_self
        torch.save({"net": net.state_dict(), "it": -1},
                   os.path.join(args.out, "pretrain.pt"))
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for it in range(args.iters):
        t0 = time.perf_counter()
        buf, seqs, t_env, t_fwd, wr, fself = collect(net, frozen, tables,
                                                     rng, device, args.steps,
                                                     amp, args)
        adv, ret, keep = gae(buf, seqs)
        t1 = time.perf_counter()
        st = update(net, opt, buf, adv, ret, keep, device, args, amp)
        t2 = time.perf_counter()
        if ema is not None:
            with torch.no_grad():
                for pe, pn in zip(ema.state_dict().values(),
                                  net.state_dict().values()):
                    if pe.dtype.is_floating_point:
                        pe.mul_(args.ema).add_(pn, alpha=1.0 - args.ema)
                    else:
                        pe.copy_(pn)
        freq = np.bincount([b["a"] for b in buf], minlength=N_ACT)
        freq = freq / max(freq.sum(), 1)
        print("freq " + " ".join(f"{x:.2f}" for x in freq), flush=True)
        print(f"it {it} n {int(keep.sum())} pool {len(frozen)} "
              f"wr {wr:.3f} self {fself:.2f} pl {st[0]:.4f} vl {st[1]:.4f} "
              f"ent {st[2]:.3f} kl {st[3]:.5f} "
              f"eq {st[4]:.4f} opp {st[5]:.4f} "
              f"env {t_env:.1f}s fwd {t_fwd:.1f}s upd {t2 - t1:.1f}s "
              f"gpu_frac {(t_fwd + t2 - t1) / (t2 - t0):.2f}", flush=True)

        if args.exploit:
            pass
        elif (it + 1) % args.pool_every == 0:
            snap = Net().to(device)
            snap.load_state_dict(net.state_dict())
            snap.eval()
            frozen.append(snap)
            if len(frozen) > args.pool_size:
                frozen.pop(0)
        if (it + 1) % args.eval_every == 0:
            net.eval()
            res = evaluate(net, device, amp, rng, args.eval_games, pairs)
            net.train()
            print("eval it %d " % it + " ".join(f"{k} {v:.3f}"
                                                for k, v in res.items()),
                  flush=True)
        if (it + 1) % args.save_every == 0:
            torch.save({"net": net.state_dict(), "it": it},
                       os.path.join(args.out, f"net_{it + 1}.pt"))
            if ema is not None:
                torch.save({"net": ema.state_dict(), "it": it},
                           os.path.join(args.out, f"ema_{it + 1}.pt"))


if __name__ == "__main__":
    main()

