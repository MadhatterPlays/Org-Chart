"""Infer reporting lines.

For each person we list the people who could plausibly be their manager, describe each pair
with ~25 features, and score them with a conditional-logit model (a softmax over that
person's candidates). Probabilities are then calibrated on held-aside training companies.

Placement at a confidence cut-off c:
  * draw an edge to the top candidate if its calibrated confidence >= c
  * otherwise take the smallest set of top candidates whose calibrated joint confidence >= c
    and roll the person up under the nearest leader all of them sit under
  * otherwise leave the person unlinked, with a reason
"""
import csv, math, os, pickle, random
from collections import defaultdict, Counter
import numpy as np
from parse import parse_record, norm_location, CITY_REGION

GRID = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99
PARENT = {
    "data": {"engineering", "product"}, "security": {"engineering", "it"}, "it": {"finance", "ops", "engineering"},
    "design": {"product", "engineering"}, "cs": {"sales", "ops"}, "legal": {"finance"}, "hr": {"ops", "finance"},
    "product": {"engineering"}, "marketing": {"sales"},
}
FEATURES = ["same_func", "parent_func", "q_ceo", "p_top", "p_top_x_q_ceo", "p_top_x_parent", "q_top", "gap0", "gap1", "gap2p", "level_diff",
            "team_match", "team_conflict", "q_no_team", "p_no_team", "q_top_of_team", "region_match", "region_conflict", "loc_match", "loc_conflict",
            "dept_match", "q_exec", "p_support_x_q_ceo", "p_support_x_q_exec", "ic_x_q_mgr", "ic_x_q_dir", "q_older", "q_newer", "p_vague",
            "cross_func_nonexec", "q_mgr_x_p_mgr", "span_log", "span_big", "equal_level"]
REASONS = {"no_title": "No title on the record", "vague": "Title too generic to place", "no_candidate": "No one above them in their function",
           "ambiguous": "Too many possible managers", "ceo_only": "Could only be placed under the CEO"}


def load(data_dir):
    comps = {r["company_id"]: r for r in csv.DictReader(open(os.path.join(data_dir, "companies.csv"), encoding="utf-8"))}
    people = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(data_dir, "profiles.csv"), encoding="utf-8")):
        people[r["company_id"]].append(r)
    truth = {r["person_id"]: r for r in csv.DictReader(open(os.path.join(data_dir, "truth.csv"), encoding="utf-8"))}
    return comps, people, truth


def start_months(s):
    if not s:
        return None
    y, m = s.split("-")
    return int(y) * 12 + int(m)


class CompanyModel:
    """Parsed people + candidate features for one company."""

    def __init__(self, cid, rows):
        self.cid = cid
        self.rows = rows
        self.n = len(rows)
        self.pid = [r["person_id"] for r in rows]
        self.idx = {p: i for i, p in enumerate(self.pid)}
        self.parsed = [parse_record(r) for r in rows]
        self.loc = [norm_location(r["location"]) for r in rows]
        self.start = [start_months(r["start_date"]) for r in rows]
        self.dept = [(r["department"] or "").strip().lower() for r in rows]
        self.find_ceo()
        self.function_tops()
        self.build_candidates()

    def find_ceo(self):
        lv0 = [i for i, p in enumerate(self.parsed) if p["level"] == 0]
        lv0.sort(key=lambda i: (0 if "ceo" in self.parsed[i]["norm"].split() or "chief executive" in self.parsed[i]["norm"] else 1, self.start[i] or 10 ** 9))
        self.ceo = lv0[0] if lv0 else None
        for i in lv0[1:]:
            self.parsed[i]["level"] = 1
            self.parsed[i]["notes"].append("Second top-level title in the company; treated as an executive below the CEO")

    def function_tops(self):
        best = {}
        for i, p in enumerate(self.parsed):
            if p["level"] is None or p["func"] is None or i == self.ceo:
                continue
            f = p["func"]
            if f not in best or p["level"] < best[f]:
                best[f] = p["level"]
        self.func_top_level = best
        team_best = {}
        for i, p in enumerate(self.parsed):
            if p["level"] is None:
                continue
            for t in p["teams"]:
                k = (p["func"], t)
                if k not in team_best or p["level"] < team_best[k]:
                    team_best[k] = p["level"]
        self.team_top_level = team_best
        # distinct manager levels per function
        lv = defaultdict(set)
        for i, p in enumerate(self.parsed):
            if p["level"] is not None and p["manager_capable"] and p["func"]:
                lv[p["func"]].add(p["level"])
        self.func_mgr_levels = {f: sorted(v) for f, v in lv.items()}
        # expected span of control: people whose nearest manager level (in function) is L, per manager at L
        below = Counter()
        at = Counter()
        for i, p in enumerate(self.parsed):
            if p["level"] is None or p["func"] is None:
                continue
            if p["manager_capable"]:
                at[(p["func"], p["level"])] += 1
            above = [l for l in self.func_mgr_levels.get(p["func"], []) if l < p["level"]]
            if above:
                below[(p["func"], max(above))] += 1
        self.span = {k: below[k] / at[k] for k in at}

    def is_top(self, i):
        p = self.parsed[i]
        return p["func"] is not None and p["level"] is not None and p["level"] <= self.func_top_level.get(p["func"], -1)

    def build_candidates(self):
        P = self.parsed
        mgrs = [j for j, q in enumerate(P) if q["level"] is not None and (q["manager_capable"] or j == self.ceo)]
        self.cands = [[] for _ in range(self.n)]
        self.feats = [None] * self.n
        for i, p in enumerate(P):
            if i == self.ceo or p["level"] is None:
                continue
            pf, pl = p["func"], p["level"]
            p_top_i = self.is_top(i)
            cl = []
            for j in mgrs:
                if j == i:
                    continue
                q = P[j]
                if q["level"] > pl or (q["level"] == pl and not (p_top_i and q["func"] in PARENT.get(pf, ()) and self.is_top(j))):
                    continue
                if j == self.ceo:
                    cl.append(j); continue
                if p["exec_support"]:
                    if q["level"] <= 1.5:
                        cl.append(j)
                    continue
                if pf is None:
                    cl.append(j); continue
                if q["func"] == pf or q["func"] in PARENT.get(pf, ()) or q["level"] <= 2:
                    cl.append(j)
            self.cands[i] = cl
            if cl:
                self.feats[i] = np.array([self.pair_features(i, j) for j in cl], dtype=np.float32)

    def pair_features(self, i, j):
        p, q = self.parsed[i], self.parsed[j]
        pf, qf = p["func"], q["func"]
        same = 1.0 if (pf is not None and pf == qf) else 0.0
        parent = 1.0 if (pf in PARENT and qf in PARENT[pf]) else 0.0
        q_ceo = 1.0 if j == self.ceo else 0.0
        p_top = 1.0 if self.is_top(i) else 0.0
        q_top = 1.0 if (j != self.ceo and self.is_top(j)) else 0.0
        gap0 = gap1 = gap2p = 0.0
        if same:
            above = [l for l in self.func_mgr_levels.get(pf, []) if l < p["level"]]
            above = sorted(above, reverse=True)
            if q["level"] in above:
                r = above.index(q["level"])
                gap0, gap1, gap2p = (1.0, 0, 0) if r == 0 else ((0, 1.0, 0) if r == 1 else (0, 0, 1.0))
        tp, tq = set(p["teams"]), set(q["teams"])
        team_match = 1.0 if tp & tq else 0.0
        team_conflict = 1.0 if (tp and tq and not tp & tq and same) else 0.0
        q_top_team = 0.0
        for t in tp & tq:
            if q["level"] <= self.team_top_level.get((qf, t), 99):
                q_top_team = 1.0
        rp = p["region"] or (CITY_REGION.get(self.loc[i]) if p["func"] in ("sales", "cs") and False else None)
        region_match = 1.0 if (p["region"] and q["region"] and p["region"] == q["region"]) else 0.0
        region_conflict = 1.0 if (p["region"] and q["region"] and p["region"] != q["region"]) else 0.0
        li, lj = self.loc[i], self.loc[j]
        loc_match = 1.0 if (li and lj and li == lj) else 0.0
        loc_conflict = 1.0 if (li and lj and li != lj) else 0.0
        dept_match = 1.0 if (self.dept[i] and self.dept[i] == self.dept[j]) else 0.0
        q_exec = 1.0 if q["level"] <= 2 else 0.0
        ic = 1.0 if p["level"] >= 5 else 0.0
        q_mgr = 1.0 if 3.5 <= q["level"] <= 4 else 0.0
        q_dir = 1.0 if 2.5 <= q["level"] < 3.5 else 0.0
        si, sj = self.start[i], self.start[j]
        older = 1.0 if (si and sj and sj <= si) else 0.0
        newer = 1.0 if (si and sj and sj > si) else 0.0
        support = 1.0 if p["exec_support"] else 0.0
        span = self.span.get((qf, q["level"]), 0.0) if same else 0.0
        return [same, parent, q_ceo, p_top, p_top * q_ceo, p_top * parent, q_top, gap0, gap1, gap2p, (p["level"] - q["level"]) / 6.0,
                team_match, team_conflict, 0.0 if tq else 1.0, 0.0 if tp else 1.0, q_top_team, region_match, region_conflict, loc_match, loc_conflict,
                dept_match, q_exec, support * q_ceo, support * q_exec, ic * q_mgr, ic * q_dir, older, newer, 1.0 if pf is None else 0.0,
                1.0 if (not same and not parent and not q_ceo and q["level"] > 1) else 0.0, (1.0 if p["level"] <= 4 else 0.0) * q_mgr,
                math.log1p(span) / 3.0, 1.0 if span > 12 else 0.0, 1.0 if q["level"] == p["level"] else 0.0]


# ---------------- model ----------------

def fit_clogit(groups, d, l2=0.5, iters=400, lr=0.1):
    """groups: list of (X [n x d], y_index). Adam on the softmax log-likelihood."""
    X = np.concatenate([g[0] for g in groups])
    sizes = np.array([len(g[0]) for g in groups])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    gid = np.repeat(np.arange(len(groups)), sizes)
    y_rows = starts + np.array([g[1] for g in groups])
    w = np.zeros(d)
    m = np.zeros(d); v = np.zeros(d)
    G = len(groups)
    for t in range(1, iters + 1):
        s = X @ w
        mx = np.maximum.reduceat(s, starts)
        e = np.exp(s - mx[gid])
        z = np.add.reduceat(e, starts)
        p = e / z[gid]
        grad = -(X[y_rows].sum(0) - (p[:, None] * X).sum(0)) / G + l2 * w / G
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    s = X @ w
    mx = np.maximum.reduceat(s, starts)
    nll = -(s[y_rows] - mx - np.log(np.add.reduceat(np.exp(s - mx[gid]), starts))).mean()
    return w, nll


def softmax(s):
    e = np.exp(s - s.max())
    return e / e.sum()


class Isotonic:
    """Pool-adjacent-violators isotonic regression, evaluated by step lookup."""

    def fit(self, x, y):
        o = np.argsort(x)
        x, y = np.asarray(x)[o], np.asarray(y, dtype=float)[o]
        blocks = []  # [sum_y, count, x_min, x_max]
        for xi, yi in zip(x, y):
            blocks.append([yi, 1, xi, xi])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                a = blocks.pop()
                b = blocks[-1]
                b[0] += a[0]; b[1] += a[1]; b[3] = a[3]
        self.xs = np.array([b[2] for b in blocks])
        self.ys = np.array([b[0] / b[1] for b in blocks])
        return self

    def __call__(self, x):
        k = np.searchsorted(self.xs, x, side="right") - 1
        k = np.clip(k, 0, len(self.ys) - 1)
        return self.ys[k]


# ---------------- placement ----------------

def place_company(cm, probs, g1, gk):
    """Return per-person list of placements, one per GRID cut-off: (kind, target, conf)
    kind: 'root' | 'edge' | 'rollup' | 'unlinked'. target: index or reason key."""
    P = cm.parsed
    order = sorted(range(cm.n), key=lambda i: (P[i]["level"] if P[i]["level"] is not None else 99))
    conf1 = [None] * cm.n
    confk = [None] * cm.n
    for i in range(cm.n):
        pr = probs[i]
        if pr is None:
            continue
        srt = np.sort(pr)[::-1]
        conf1[i] = float(g1(srt[0]))
        cum = np.cumsum(srt[:6])
        confk[i] = [float(x) for x in gk(cum)]  # index k-1 = joint confidence of top-k
    out = [[None] * len(GRID) for _ in range(cm.n)]
    for gi, c in enumerate(GRID):
        parent = {}
        for i in order:
            p = P[i]
            if i == cm.ceo:
                out[i][gi] = ("root", None, 1.0); continue
            if p["level"] is None:
                out[i][gi] = ("unlinked", "no_title", 0.0); continue
            pr = probs[i]
            if pr is None:
                out[i][gi] = ("unlinked", "vague" if p["vague"] else "no_candidate", 0.0); continue
            rank = np.argsort(-pr)
            cands = cm.cands[i]
            if conf1[i] >= c:
                tgt = cands[rank[0]]
                parent[i] = tgt
                out[i][gi] = ("edge", tgt, conf1[i]); continue
            placed = False
            for k in range(2, min(6, len(cands)) + 1):
                if confk[i][k - 1] >= c:
                    S = [cands[r] for r in rank[:k]]
                    lca = lowest_common_ancestor(S, parent, cm.ceo)
                    if lca is None:
                        break
                    if lca == cm.ceo:
                        out[i][gi] = ("unlinked", "ceo_only", confk[i][k - 1]); placed = True; break
                    parent[i] = lca
                    out[i][gi] = ("rollup", lca, confk[i][k - 1]); placed = True
                    break
            if not placed:
                out[i][gi] = ("unlinked", "vague" if p["vague"] else "ambiguous", conf1[i])
    return out, conf1, confk


def chain(i, parent, ceo, limit=20):
    out = [i]
    while i in parent and len(out) < limit:
        i = parent[i]
        out.append(i)
    return out


def lowest_common_ancestor(S, parent, ceo):
    chains = [chain(s, parent, ceo) for s in S]
    common = set(chains[0])
    for ch in chains[1:]:
        common &= set(ch)
    if not common:
        return None
    for x in chains[0]:
        if x in common:
            return x
    return None


# ---------------- driver ----------------

def run(data_dir, out_path, seed=7):
    comps, people, truth = load(data_dir)
    cids = sorted(comps)
    rng = random.Random(seed)
    shuffled = cids[:]
    rng.shuffle(shuffled)
    test = set(shuffled[:120])
    calib = set(shuffled[120:220])
    fit = set(shuffled[220:])
    split = {c: ("test" if c in test else ("train-calibrate" if c in calib else "train-fit")) for c in cids}
    print("parsing + candidates ...")
    models = {}
    for k, c in enumerate(cids):
        models[c] = CompanyModel(c, people[c])
        if k % 100 == 0:
            print(" ", k)
    # labels
    def true_idx(cm, i):
        mgr = truth[cm.pid[i]]["manager_id"]
        if not mgr:
            return None
        j = cm.idx[mgr]
        try:
            return cm.cands[i].index(j)
        except ValueError:
            return -1
    groups = []
    recall_in_cands = Counter()
    for c in fit:
        cm = models[c]
        for i in range(cm.n):
            if cm.feats[i] is None:
                continue
            y = true_idx(cm, i)
            recall_in_cands[y is not None and y >= 0] += 1
            if y is not None and y >= 0:
                groups.append((cm.feats[i], y))
    print("true manager among candidates:", recall_in_cands)
    w, nll = fit_clogit(groups, len(FEATURES))
    print("fit nll", round(nll, 4))
    for f, x in sorted(zip(FEATURES, w), key=lambda t: -abs(t[1])):
        print(f"  {f:22s} {x:+.2f}")
    probs = {}
    for c in cids:
        cm = models[c]
        probs[c] = [None if cm.feats[i] is None else softmax(cm.feats[i] @ w) for i in range(cm.n)]
    # calibration on train-calibrate companies
    x1, y1, xk, yk = [], [], [], []
    for c in calib:
        cm = models[c]
        for i in range(cm.n):
            pr = probs[c][i]
            if pr is None:
                continue
            y = true_idx(cm, i)
            rank = np.argsort(-pr)
            x1.append(pr[rank[0]]); y1.append(1 if y == rank[0] else 0)
            cum = 0.0
            for k in range(min(6, len(pr))):
                cum += pr[rank[k]]
                if k >= 1:
                    xk.append(cum); yk.append(1 if (y is not None and y >= 0 and y in rank[:k + 1]) else 0)
    g1 = Isotonic().fit(x1, y1)
    gk = Isotonic().fit(xk, yk)
    print("placing ...")
    placements, confs = {}, {}
    for k, c in enumerate(cids):
        out, conf1, confk = place_company(models[c], probs[c], g1, gk)
        placements[c] = out
        confs[c] = (conf1, confk)
        if k % 100 == 0:
            print(" ", k)
    with open(out_path, "wb") as fh:
        pickle.dump(dict(comps=comps, truth=truth, split=split, models=models, probs=probs, placements=placements, confs=confs,
                         weights=w.tolist(), features=FEATURES, g1=(g1.xs.tolist(), g1.ys.tolist()), gk=(gk.xs.tolist(), gk.ys.tolist())), fh)
    print("saved", out_path)


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    run(os.path.join(here, "..", "data"), os.path.join(here, "..", "data", "inference.pkl"))
