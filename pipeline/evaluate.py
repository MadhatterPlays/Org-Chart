"""Score inference against the hidden truth and write metrics.json."""
import json, os, pickle, random
from collections import Counter, defaultdict
from infer import GRID, CompanyModel, REASONS  # noqa: F401 (CompanyModel needed for unpickling)

EXEC_LEVELS = {"CEO", "C-level", "SVP", "VP"}
LEVEL_GROUP = {"CEO": "Executives", "C-level": "Executives", "SVP": "Executives", "VP": "Executives", "Director": "Directors",
               "Manager": "Managers", "IC-senior": "Individual contributors", "IC": "Individual contributors", "IC-junior": "Individual contributors",
               "Exec support": "Executive support"}
FUNC_NAMES = {"engineering": "engineering", "data": "data", "security": "security", "it": "IT", "product": "product", "design": "design",
              "sales": "sales", "cs": "customer success", "marketing": "marketing", "finance": "finance", "hr": "people / HR", "legal": "legal",
              "ops": "operations", "clinical": "clinical", "research": "research", "editorial": "editorial", "executive": "executive office"}


class Truth:
    def __init__(self, cm, truth):
        self.mgr = []
        self.rec = []
        for i in range(cm.n):
            t = truth[cm.pid[i]]
            self.rec.append(t)
            self.mgr.append(cm.idx[t["manager_id"]] if t["manager_id"] else None)

    def ancestors(self, i):
        out = []
        j = self.mgr[i]
        while j is not None:
            out.append(j)
            j = self.mgr[j]
        return out


def tree_at(placement, gi):
    """parent map (edges + rollups) and kinds at grid index gi"""
    parent, kind = {}, {}
    for i, pl in enumerate(placement):
        k, tgt, conf = pl[gi]
        kind[i] = (k, tgt, conf)
        if k in ("edge", "rollup"):
            parent[i] = tgt
    return parent, kind


def anc_inferred(i, parent):
    out = []
    while i in parent and len(out) < 30:
        i = parent[i]
        out.append(i)
    return out


def company_counts(cm, T, placement, gi):
    c = Counter()
    for i in range(cm.n):
        if i == cm.ceo:
            continue
        k, tgt, conf = placement[i][gi]
        c["people"] += 1
        if k == "edge":
            c["edges"] += 1
            c["edges_correct"] += int(tgt == T.mgr[i])
        elif k == "rollup":
            c["rollups"] += 1
            c["rollups_correct"] += int(tgt in T.ancestors(i))
        else:
            c["unlinked"] += 1
            c["unl_" + tgt] += 1
    return c


def summarize(c):
    e = c["edges"] or 1
    p = c["people"] or 1
    return dict(precision=c["edges_correct"] / e if c["edges"] else None, coverage=c["edges"] / p, recall=c["edges_correct"] / p,
                rollup_share=c["rollups"] / p, rollup_accuracy=(c["rollups_correct"] / c["rollups"]) if c["rollups"] else None,
                unlinked_share=c["unlinked"] / p, people=c["people"], edges=c["edges"], edges_correct=c["edges_correct"],
                rollups=c["rollups"], rollups_correct=c["rollups_correct"], unlinked=c["unlinked"])


# ---------- sales questions (mirrored in the web app) ----------

def who_runs(cm, parent, func=None, team=None):
    """Highest-level person matching a function (or team). Ties broken by inferred ancestry."""
    P = cm.parsed
    pool = [i for i in range(cm.n) if P[i]["level"] is not None and i != cm.ceo and
            ((func and P[i]["func"] == func) or (team and team in P[i]["teams"]))]
    if not pool:
        return None, "nobody"
    best = min(P[i]["level"] for i in pool)
    top = [i for i in pool if P[i]["level"] == best]
    if len(top) == 1:
        return top[0], "unique"
    # someone who is an inferred ancestor of all the others
    for i in top:
        if all(i == j or i in anc_inferred(j, parent) for j in top):
            return i, "ancestor"
    return None, "tie"


def truth_runs(cm, T, func=None, team=None):
    pool = [i for i in range(cm.n) if i != cm.ceo and ((func and T.rec[i]["function"] == func) or (team and T.rec[i]["team"].lower() == team))]
    tops = [i for i in pool if T.mgr[i] is None or T.mgr[i] not in pool]
    return tops[0] if len(tops) == 1 else None


def economic_buyer(cm, parent, i):
    for j in anc_inferred(i, parent):
        lv = cm.parsed[j]["level"]
        if lv is not None and lv <= 2:
            return j
    return None


def truth_buyer(T, i):
    for j in T.ancestors(i):
        if T.rec[j]["level"] in EXEC_LEVELS:
            return j
    return None


def run(pkl, out_json):
    D = pickle.load(open(pkl, "rb"))
    comps, truth, split, models, placements = D["comps"], D["truth"], D["split"], D["models"], D["placements"]
    truths = {c: Truth(models[c], truth) for c in models}
    train = [c for c in models if split[c] != "test"]
    test = [c for c in models if split[c] == "test"]

    # ---- choose default cut-off on training companies ----
    def agg(cids, gi):
        c = Counter()
        for cid in cids:
            c.update(company_counts(models[cid], truths[cid], placements[cid], gi))
        return summarize(c)
    train_curve = [agg(train, gi) for gi in range(len(GRID))]
    default_gi = next((gi for gi, s in enumerate(train_curve) if s["precision"] and s["precision"] >= 0.95), len(GRID) - 1)
    default_c = GRID[default_gi]
    print("default cut-off", default_c, train_curve[default_gi])

    test_curve = [agg(test, gi) for gi in range(len(GRID))]
    headline = test_curve[default_gi]
    print("test at default", headline)

    # ---- calibration (test, top candidate) ----
    bins = [[0, 0] for _ in range(10)]
    for cid in test:
        cm, T = models[cid], truths[cid]
        conf1 = D["confs"][cid][0]
        for i in range(cm.n):
            if conf1[i] is None:
                continue
            pr = D["probs"][cid][i]
            top = cm.cands[i][int(pr.argmax())]
            b = min(9, int(conf1[i] * 10))
            bins[b][0] += 1
            bins[b][1] += int(top == T.mgr[i])
    calibration = [dict(lo=b / 10, hi=(b + 1) / 10, n=n, accuracy=(k / n if n else None)) for b, (n, k) in enumerate(bins)]

    # ---- breakdowns + error types at default (test) ----
    groups = defaultdict(Counter)
    errors = Counter()
    errors_with_mess = Counter()
    error_examples = defaultdict(list)
    unl = Counter()
    for cid in test:
        cm, T = models[cid], truths[cid]
        comp = comps[cid]
        mess_band = "Low mess" if float(comp["messiness"]) < 0.87 else ("Medium mess" if float(comp["messiness"]) < 1.23 else "High mess")
        for i in range(cm.n):
            if i == cm.ceo:
                continue
            k, tgt, conf = placements[cid][i][default_gi]
            tags = [x for x in T.rec[i]["mess"].split("|") if x]
            keys = [("Industry", comp["industry"]), ("Company size", {"small": "100–140", "mid": "141–260", "large": "261–450"}[comp["size_band"]]),
                    ("Seniority (true)", LEVEL_GROUP[T.rec[i]["level"]]), ("Function (true)", FUNC_NAMES.get(T.rec[i]["function"], T.rec[i]["function"])),
                    ("Company messiness", mess_band)]
            keys += [("Mess on the record", t.replace("_", " ")) for t in tags] or [("Mess on the record", "clean record")]
            for key in keys:
                g = groups[key]
                g["people"] += 1
                if k == "edge":
                    g["edges"] += 1
                    g["edges_correct"] += int(tgt == T.mgr[i])
                elif k == "rollup":
                    g["rollups"] += 1
                    g["rollups_correct"] += int(tgt in T.ancestors(i))
                else:
                    g["unlinked"] += 1
            if k == "unlinked":
                unl[tgt] += 1
            if k == "edge" and tgt != T.mgr[i]:
                tm = T.mgr[i]
                if tm is not None and T.mgr[tm] == tgt:
                    et = "Skipped a level (picked their manager's manager)"
                elif tm is not None and T.mgr[tgt] is not None and T.mgr[tgt] == T.mgr[tm]:
                    et = "Picked a peer of their real manager"
                elif tm is not None and tm in T.ancestors(tgt):
                    et = "Picked someone below their real manager"
                elif tm is not None and T.rec[tgt]["function"] == T.rec[tm]["function"]:
                    et = "Right function, wrong branch"
                else:
                    et = "Wrong function"
                errors[et] += 1
                mtags = set(tags) | set(x for x in T.rec[tm]["mess"].split("|") if x) if tm is not None else set(tags)
                if mtags & {"stale_title", "vague_title", "missing_title", "typo", "non_english"}:
                    errors_with_mess[et] += 1
                if len(error_examples[et]) < 6:
                    error_examples[et].append(dict(company=cid, person=cm.rows[i]["name"], title=cm.rows[i]["title"],
                                                   picked=cm.rows[tgt]["title"], real=(cm.rows[tm]["title"] if tm is not None else ""), confidence=round(conf, 3)))
    breakdowns = defaultdict(list)
    for (dim, val), g in groups.items():
        s = summarize(g)
        breakdowns[dim].append(dict(group=val, **{k: s[k] for k in ("people", "precision", "coverage", "rollup_accuracy", "unlinked_share")}))
    for dim in breakdowns:
        breakdowns[dim].sort(key=lambda r: -r["people"])

    # ---- the three sales questions (test, default cut-off) ----
    rng = random.Random(11)
    q1 = Counter(); q1t = Counter(); q2 = Counter(); q3 = Counter()
    q_examples = defaultdict(list)
    for cid in test:
        cm, T = models[cid], truths[cid]
        parent, kind = tree_at(placements[cid], default_gi)
        funcs = sorted({T.rec[i]["function"] for i in range(cm.n)} - {"executive"})
        for f in funcs:
            truth_head = truth_runs(cm, T, func=f)
            if truth_head is None:
                continue
            ans, how = who_runs(cm, parent, func=f)
            q1["asked"] += 1
            if ans is not None:
                q1["answered"] += 1
                q1["correct"] += int(ans == truth_head)
            # Q2: who does the head of f report to
            q2["asked"] += 1
            if ans is not None and kind[ans][0] == "edge":
                q2["answered"] += 1
                q2["correct"] += int(ans == truth_head and kind[ans][1] == T.mgr[truth_head])
            elif ans is not None and kind[ans][0] == "rollup":
                q2["rolled_up"] += 1
            if len(q_examples["q1"]) < 8 and ans is not None:
                q_examples["q1"].append(dict(company=comps[cid]["name"], question=f"who runs the {FUNC_NAMES.get(f, f)} team?",
                                             answer=cm.rows[ans]["title"], correct=ans == truth_head))
        teams = sorted({T.rec[i]["team"].lower() for i in range(cm.n) if T.rec[i]["team"]})
        for tm in teams:
            th = truth_runs(cm, T, team=tm)
            if th is None:
                continue
            ans, how = who_runs(cm, parent, team=tm)
            q1t["asked"] += 1
            if ans is not None:
                q1t["answered"] += 1
                q1t["correct"] += int(ans == th)
        pool = [i for i in range(cm.n) if T.rec[i]["level"] not in EXEC_LEVELS and i != cm.ceo]
        for i in rng.sample(pool, min(30, len(pool))):
            tb = truth_buyer(T, i)
            b = economic_buyer(cm, parent, i)
            q3["asked"] += 1
            if b is not None:
                q3["answered"] += 1
                q3["correct"] += int(b == tb)

    def qs(c):
        return dict(asked=c["asked"], answered=c["answered"], answered_share=c["answered"] / max(1, c["asked"]),
                    accuracy=(c["correct"] / c["answered"]) if c["answered"] else None, rolled_up=c.get("rolled_up", 0))

    # ---- per company scores at default ----
    per_company = {}
    for cid in models:
        s = summarize(company_counts(models[cid], truths[cid], placements[cid], default_gi))
        per_company[cid] = dict(precision=s["precision"], coverage=s["coverage"], rollup_accuracy=s["rollup_accuracy"])

    metrics = dict(
        default_cutoff=default_c, default_index=default_gi, grid=GRID,
        n_train=len(train), n_test=len(test),
        train_people=sum(models[c].n for c in train), test_people=sum(models[c].n for c in test),
        headline=headline, train_at_default=train_curve[default_gi],
        curve=[dict(cutoff=GRID[gi], precision=test_curve[gi]["precision"], coverage=test_curve[gi]["coverage"],
                    rollup_share=test_curve[gi]["rollup_share"], rollup_accuracy=test_curve[gi]["rollup_accuracy"],
                    train_precision=train_curve[gi]["precision"], train_coverage=train_curve[gi]["coverage"]) for gi in range(len(GRID))],
        calibration=calibration, breakdowns=breakdowns,
        errors=[dict(type=k, count=v, with_title_mess=errors_with_mess[k], examples=error_examples[k]) for k, v in errors.most_common()],
        unlinked={REASONS[k]: v for k, v in unl.most_common()},
        questions=dict(who_runs_function=qs(q1), who_runs_team=qs(q1t), reports_to=qs(q2), economic_buyer=qs(q3), examples=q_examples),
        per_company=per_company,
        weights=dict(zip(D["features"], D["weights"])),
    )
    json.dump(metrics, open(out_json, "w"), indent=1, default=float)
    print(json.dumps(metrics["questions"], indent=1, default=float)[:1500])
    print("errors", metrics["errors"][:5] and [(e["type"], e["count"]) for e in metrics["errors"]])
    print("unlinked", metrics["unlinked"])
    return metrics


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    run(os.path.join(here, "..", "data", "inference.pkl"), os.path.join(here, "..", "data", "metrics.json"))
