"""Pack per-company results into compact gzip+base64 blobs and inline them into web/index.html."""
import base64, csv, gzip, json, os, pickle
from collections import Counter
from infer import GRID, CompanyModel, REASONS, PARENT  # noqa: F401
from evaluate import Truth

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
REASON_KEYS = list(REASONS)
KIND = {"root": 0, "edge": 1, "rollup": 2, "unlinked": 3}
MESS_DESC = {
    "abbreviation": "Abbreviated words (Sr., Mgr, VP, SWE, AE, CTO...)", "missing_department": "Department left blank",
    "missing_start_date": "Start date left blank", "missing_location": "Location left blank", "location_alias": "Same city written differently (NYC, Bangalore, Gurgaon...)",
    "case": "All caps, all lower case or Title Case", "remote": "Location says Remote", "fluff": "Decoration (| Hiring!, Ex-Amazon, Interim, emoji...)",
    "typo": "Typo in the title (dropped or swapped letter)", "level_numeral": "Level as a numeral (Engineer II / III)", "name_format": "Name in caps or initialled",
    "missing_title": "Title left blank", "vague_title": "Title with no function (Consultant, Team Member, Associate...)",
    "duplicate_name": "Two people in the company share a name", "stale_title": "Manager still showing their old individual-contributor title",
    "non_english": "Title in Spanish or Portuguese",
}


def codes_for(cm, i, j):
    f = dict(zip(cm_features, cm.pair_features(i, j)))
    s = ""
    if j == cm.ceo: s += "C"
    if f["same_func"]: s += "F"
    if f["parent_func"]: s += "P"
    if f["gap0"]: s += "G"
    if f["team_match"]: s += "T"
    if f["team_conflict"]: s += "X"
    if f["region_match"]: s += "R"
    if f["region_conflict"]: s += "r"
    if f["loc_match"]: s += "L"
    if f["loc_conflict"]: s += "l"
    if f["span_big"]: s += "S"
    if f["equal_level"]: s += "E"
    if f["p_top"]: s += "H"
    if f["cross_func_nonexec"]: s += "Z"
    return s


def r3(x):
    return None if x is None else round(float(x), 3)


def company_blob(cid, comp, cm, T, probs, placement, conf1, split):
    rows = [[r["name"], r["title"], r["department"], r["location"], r["start_date"]] for r in cm.rows]
    parsed = []
    for p in cm.parsed:
        flags = (1 if p["exec_support"] else 0) | (2 if p["vague"] else 0) | (4 if p["contractor"] else 0)
        parsed.append([p["norm"], p["level"], p["level_label"], p["func"], "|".join(p["teams"]), p["region"], "\n".join(p["notes"]),
                       p["func_source"], flags])
    cands = []
    for i in range(cm.n):
        pr = probs[i]
        if pr is None:
            cands.append([]); continue
        order = sorted(range(len(pr)), key=lambda k: -pr[k])[:6]
        cands.append([[cm.cands[i][k], r3(pr[k]), codes_for(cm, i, cm.cands[i][k])] for k in order])
    pl = []
    for i in range(cm.n):
        seq, last = [], None
        for gi, (k, tgt, conf) in enumerate(placement[i]):
            tg = REASON_KEYS.index(tgt) if k == "unlinked" else (tgt if tgt is not None else -1)
            cur = (KIND[k], tg, r3(conf))
            if cur != last:
                seq.append([gi, *cur])
                last = cur
        pl.append(seq)
    truth = [[T.mgr[i] if T.mgr[i] is not None else -1, T.rec[i]["function"], T.rec[i]["team"], T.rec[i]["level"], T.rec[i]["clean_title"], T.rec[i]["mess"]]
             for i in range(cm.n)]
    blob = dict(id=cid, rows=rows, parsed=parsed, cands=cands, pl=pl, conf1=[r3(x) for x in conf1], truth=truth, ceo=cm.ceo)
    raw = json.dumps(blob, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return base64.b64encode(gzip.compress(raw, 9)).decode("ascii"), len(raw)


def main():
    global cm_features
    D = pickle.load(open(os.path.join(ROOT, "data", "inference.pkl"), "rb"))
    metrics = json.load(open(os.path.join(ROOT, "data", "metrics.json")))
    cm_features = D["features"]
    comps, split, models = D["comps"], D["split"], D["models"]
    blobs, index = {}, []
    raw_total = 0
    for cid in sorted(models):
        cm = models[cid]
        T = Truth(cm, D["truth"])
        b, n = company_blob(cid, comps[cid], cm, T, D["probs"][cid], D["placements"][cid], D["confs"][cid][0], split[cid])
        blobs[cid] = b
        raw_total += n
        c = comps[cid]
        s = metrics["per_company"][cid]
        index.append(dict(id=cid, name=c["name"], industry=c["industry"], band=c["size_band"], n=cm.n, split=split[cid],
                          offices=c["offices"].split("|"), messiness=float(c["messiness"]), precision=s["precision"], coverage=s["coverage"]))
    # data summary
    truth_rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "truth.csv"), encoding="utf-8")))
    mess = Counter()
    for r in truth_rows:
        for t in r["mess"].split("|"):
            if t:
                mess[t] += 1
    people_with_mess = sum(1 for r in truth_rows if r["mess"])
    summary = dict(companies=len(index), people=len(truth_rows), people_with_mess=people_with_mess,
                   industries=Counter(c["industry"] for c in index), bands=Counter(c["band"] for c in index),
                   sizes=dict(min=min(c["n"] for c in index), max=max(c["n"] for c in index), mean=round(sum(c["n"] for c in index) / len(index))),
                   mess=[dict(key=k, label=MESS_DESC.get(k, k), count=v, share=v / len(truth_rows)) for k, v in mess.most_common()],
                   levels=Counter(r["level"] for r in truth_rows), functions=Counter(r["function"] for r in truth_rows))
    meta = dict(index=index, metrics=metrics, summary=summary, reasons=[REASONS[k] for k in REASON_KEYS], grid=GRID)
    meta_b64 = base64.b64encode(gzip.compress(json.dumps(meta, separators=(",", ":"), default=float).encode(), 9)).decode()
    tpl = open(os.path.join(ROOT, "web", "template.html"), encoding="utf-8").read()
    html = tpl.replace("__META__", meta_b64).replace("__COMPANIES__", json.dumps(blobs, separators=(",", ":")))
    out = os.path.join(ROOT, "web", "index.html")
    open(out, "w", encoding="utf-8").write(html)
    print(f"raw company json {raw_total / 1e6:.1f} MB; index.html {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
