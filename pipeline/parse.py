"""Read one raw profile into structured fields, and keep notes on how it was read."""
import re
import difflib

WORD_EXPAND = {
    "sr": "senior", "snr": "senior", "jr": "junior", "mgr": "manager", "dir": "director", "engr": "engineer",
    "swe": "software engineer", "ae": "account executive", "acct": "account", "sdr": "sales development representative",
    "bdr": "business development representative", "csm": "customer success manager", "hr": "human resources",
    "ops": "operations", "mktg": "marketing", "assoc": "associate", "pm": "product manager", "ml": "machine learning",
    "rep": "representative", "exec": "executive", "vp": "vice president", "svp": "senior vice president", "evp": "executive vice president",
    "sre": "site reliability engineer", "ea": "executive assistant", "bi": "business intelligence", "hrbp": "human resources business partner",
    "qa": "qa", "ux": "ux", "it": "it", "gm": "general manager", "cs": "customer success",
    # Spanish / Portuguese
    "ingeniero": "engineer", "engenheiro": "engineer", "gerente": "manager", "ventas": "sales", "ejecutivo": "executive",
    "cuentas": "account", "analista": "analyst", "datos": "data", "dados": "data", "reclutador": "recruiter", "contador": "accountant",
    "especialista": "specialist", "soporte": "support", "ingeniería": "engineering", "ingenieria": "engineering", "diseñador": "designer",
    "producto": "product", "financeiro": "financial",
}
STOP = {"de", "of", "the", "and", "&", "to", "a", "-", ","}

CSUITE = {
    "ceo": ("executive", None), "cto": ("engineering", None), "cfo": ("finance", None), "coo": ("ops", None), "cro": ("sales", None),
    "ciso": ("security", None), "cio": ("it", None), "cdo": ("data", None), "chro": ("hr", None), "clo": ("legal", None),
    "cmo": ("marketing", ["marketing", "clinical"]), "cpo": ("product", ["product", "hr"]), "cco": ("cs", ["cs", "editorial", "legal"]),
    "cso": ("research", ["research", "security", "sales"]),
}
CHIEF_WORD = {"technology": "engineering", "financial": "finance", "operating": "ops", "revenue": "sales", "marketing": "marketing",
              "information security": "security", "security": "security", "information": "it", "data": "data", "product": "product",
              "people": "hr", "human resources": "hr", "legal": "legal", "customer": "cs", "scientific": "research", "medical": "clinical",
              "content": "editorial", "executive": "executive", "sales": "sales"}

# phrase -> (function, weight). Longest phrases are matched first.
FUNC_PHRASES = {
    "software engineer": ("engineering", 3), "frontend engineer": ("engineering", 3), "mobile engineer": ("engineering", 3), "ios engineer": ("engineering", 3),
    "android engineer": ("engineering", 3), "devops": ("engineering", 3), "site reliability": ("engineering", 3), "qa engineer": ("engineering", 3),
    "test automation": ("engineering", 3), "engineering manager": ("engineering", 3), "engineering": ("engineering", 2), "developer": ("engineering", 2),
    "software developer": ("engineering", 3), "software": ("engineering", 2), "platform": ("engineering", 1), "backend": ("engineering", 2), "frontend": ("engineering", 2),
    "infrastructure": ("engineering", 2), "qa": ("engineering", 2), "data engineering": ("data", 4), "business intelligence developer": ("data", 4), "it support specialist": ("it", 4), "clinical research associate": ("clinical", 4), "clinical research": ("clinical", 4), "mobile": ("engineering", 1), "payments": ("engineering", 1),
    "data engineer": ("data", 3), "data analyst": ("data", 3), "data scientist": ("data", 3), "analytics engineer": ("data", 3), "analytics": ("data", 2),
    "machine learning": ("data", 3), "business intelligence": ("data", 3), "data science": ("data", 3), "data": ("data", 2), "bi developer": ("data", 3),
    "security": ("security", 3), "infosec": ("security", 3), "soc analyst": ("security", 3), "grc": ("security", 3), "iam engineer": ("security", 3),
    "identity": ("security", 2), "security operations": ("security", 3),
    "it support": ("it", 3), "help desk": ("it", 3), "systems administrator": ("it", 3), "network engineer": ("it", 3), "it manager": ("it", 3),
    "it": ("it", 2), "information technology": ("it", 3), "corporate systems": ("it", 3),
    "product manager": ("product", 3), "product analyst": ("product", 3), "group product manager": ("product", 3), "product lead": ("product", 3),
    "product": ("product", 2), "core product": ("product", 3), "platform product": ("product", 3), "growth": ("product", 1),
    "designer": ("design", 3), "ux": ("design", 3), "ux researcher": ("design", 3), "design": ("design", 2), "product design": ("design", 3), "brand design": ("design", 3),
    "account executive": ("sales", 3), "sales": ("sales", 3), "sales development": ("sales", 3), "business development": ("sales", 3),
    "solutions engineer": ("sales", 3), "sales engineer": ("sales", 3), "enterprise": ("sales", 1), "mid-market": ("sales", 2), "smb": ("sales", 2),
    "revenue": ("sales", 2), "solutions engineering": ("sales", 3), "regional sales manager": ("sales", 3),
    "customer success": ("cs", 3), "support specialist": ("cs", 3), "technical support": ("cs", 3), "implementation": ("cs", 3), "onboarding": ("cs", 3),
    "customer experience": ("cs", 3), "support": ("cs", 2), "customer": ("cs", 2),
    "marketing": ("marketing", 3), "demand generation": ("marketing", 3), "content writer": ("marketing", 3), "content marketing": ("marketing", 3),
    "brand": ("marketing", 2), "communications": ("marketing", 2), "events": ("marketing", 2), "product marketing": ("marketing", 4), "content": ("marketing", 1),
    "accountant": ("finance", 3), "accounting": ("finance", 3), "financial": ("finance", 3), "finance": ("finance", 3), "fp&a": ("finance", 3),
    "procurement": ("finance", 3), "payroll": ("finance", 3), "controller": ("finance", 3),
    "recruiter": ("hr", 3), "sourcer": ("hr", 3), "talent": ("hr", 3), "people": ("hr", 2), "human resources": ("hr", 3), "l&d": ("hr", 3),
    "learning & development": ("hr", 3), "compensation": ("hr", 3), "total rewards": ("hr", 3), "people operations": ("hr", 3), "hr generalist": ("hr", 3),
    "counsel": ("legal", 3), "paralegal": ("legal", 3), "legal": ("legal", 3), "compliance": ("legal", 2), "general counsel": ("legal", 4),
    "operations": ("ops", 2), "supply chain": ("ops", 3), "warehouse": ("ops", 3), "fulfillment": ("ops", 3), "facilities": ("ops", 3), "buyer": ("ops", 2),
    "inventory": ("ops", 3), "production technician": ("ops", 3), "machine operator": ("ops", 3), "quality inspector": ("ops", 3), "quality engineer": ("ops", 3),
    "quality": ("ops", 2), "manufacturing": ("ops", 3), "logistics": ("ops", 3), "business operations": ("ops", 3),
    "nurse": ("clinical", 3), "clinical": ("clinical", 3), "care coordinator": ("clinical", 3), "nursing": ("clinical", 3), "medical": ("clinical", 2), "care management": ("clinical", 3),
    "research scientist": ("research", 3), "research associate": ("research", 3), "scientist": ("research", 2), "lab technician": ("research", 3),
    "discovery": ("research", 3), "translational": ("research", 3), "research": ("research", 2), "lab operations": ("research", 3), "lab": ("research", 2),
    "reporter": ("editorial", 3), "editor": ("editorial", 3), "writer": ("editorial", 1), "video producer": ("editorial", 3), "editorial": ("editorial", 3),
    "managing editor": ("editorial", 4), "newsroom": ("editorial", 3), "news": ("editorial", 2), "features": ("editorial", 2),
    "chief of staff": ("executive", 5), "executive assistant": ("executive", 5), "executive business partner": ("executive", 5),
}

DEPT_MAP = {
    "engineering": ["engineering"], "r&d": ["engineering", "research"], "tech": ["engineering"], "product & engineering": ["engineering", "product"],
    "data": ["data"], "analytics": ["data"], "data & analytics": ["data"], "security": ["security"], "infosec": ["security"],
    "information security": ["security"], "it": ["it"], "information technology": ["it"], "corporate it": ["it"], "product": ["product", "design"],
    "product management": ["product"], "design": ["design"], "ux": ["design"], "creative": ["design"], "sales": ["sales"], "revenue": ["sales"],
    "go-to-market": ["sales", "marketing"], "business development": ["sales"], "customer success": ["cs"], "customer support": ["cs"], "support": ["cs"],
    "customer experience": ["cs"], "marketing": ["marketing"], "growth marketing": ["marketing"], "brand & communications": ["marketing"],
    "finance": ["finance"], "finance & accounting": ["finance"], "accounting": ["finance"], "people": ["hr"], "hr": ["hr"], "human resources": ["hr"],
    "people & culture": ["hr"], "talent": ["hr"], "legal": ["legal"], "legal & compliance": ["legal"], "compliance": ["legal", "security"],
    "operations": ["ops"], "ops": ["ops"], "business operations": ["ops"], "supply chain": ["ops"], "clinical": ["clinical"], "clinical operations": ["clinical"],
    "medical": ["clinical"], "research": ["research"], "discovery": ["research"], "editorial": ["editorial"], "content": ["editorial", "marketing"],
    "newsroom": ["editorial"], "executive": ["executive"], "leadership": ["executive"], "office of the ceo": ["executive"],
}

TEAM_PHRASES = ["platform", "backend", "frontend", "mobile", "infrastructure", "qa", "site reliability", "payments", "data engineering", "analytics",
    "data science", "machine learning", "business intelligence", "security operations", "application security", "grc", "identity & access", "it support",
    "corporate systems", "core product", "growth", "platform product", "product design", "ux research", "brand design", "enterprise", "mid-market", "smb",
    "sales development", "sales operations", "solutions engineering", "customer success", "support", "onboarding", "demand generation", "content",
    "product marketing", "brand", "events", "accounting", "fp&a", "procurement", "payroll", "talent acquisition", "people operations",
    "learning & development", "total rewards", "commercial legal", "compliance", "business operations", "supply chain", "facilities", "warehouse",
    "fulfillment", "quality", "manufacturing", "nursing", "clinical operations", "care management", "discovery", "translational", "lab operations",
    "news", "features", "video"]
TEAM_ALIAS = {"lab ops": "lab operations", "mid market": "mid-market", "sales ops": "sales operations", "business ops": "business operations",
              "people ops": "people operations", "sales dev": "sales development", "l&d": "learning & development", "ml": "machine learning",
              "identity and access": "identity & access", "appsec": "application security"}
# role words that imply a team
ROLE_TEAM = [("data engineer", "data engineering"), ("data analyst", "analytics"), ("analytics engineer", "analytics"), ("data scientist", "data science"),
    ("machine learning engineer", "machine learning"), ("bi analyst", "business intelligence"), ("bi developer", "business intelligence"),
    ("business intelligence", "business intelligence"), ("qa engineer", "qa"), ("test automation", "qa"), ("site reliability", "site reliability"),
    ("devops", "infrastructure"), ("mobile engineer", "mobile"), ("ios engineer", "mobile"), ("android engineer", "mobile"), ("frontend engineer", "frontend"),
    ("security analyst", "security operations"), ("soc analyst", "security operations"), ("application security", "application security"),
    ("grc analyst", "grc"), ("iam engineer", "identity & access"), ("it support", "it support"), ("help desk", "it support"),
    ("systems administrator", "corporate systems"), ("network engineer", "corporate systems"), ("product designer", "product design"),
    ("ux designer", "product design"), ("ux researcher", "ux research"), ("graphic designer", "brand design"), ("brand designer", "brand design"),
    ("enterprise account executive", "enterprise"), ("sales development", "sales development"), ("business development representative", "sales development"),
    ("sdr manager", "sales development"), ("sales operations", "sales operations"), ("solutions engineer", "solutions engineering"), ("sales engineer", "solutions engineering"),
    ("customer success manager", "customer success"), ("support specialist", "support"), ("technical support", "support"), ("implementation specialist", "onboarding"),
    ("onboarding specialist", "onboarding"), ("demand generation", "demand generation"), ("content writer", "content"), ("content marketing", "content"),
    ("product marketing", "product marketing"), ("brand specialist", "brand"), ("communications specialist", "brand"), ("events coordinator", "events"),
    ("accountant", "accounting"), ("financial analyst", "fp&a"), ("procurement", "procurement"), ("payroll", "payroll"), ("recruiter", "talent acquisition"),
    ("sourcer", "talent acquisition"), ("hr generalist", "people operations"), ("people operations", "people operations"), ("l&d", "learning & development"),
    ("compensation", "total rewards"), ("corporate counsel", "commercial legal"), ("paralegal", "commercial legal"), ("compliance specialist", "compliance"),
    ("operations analyst", "business operations"), ("supply chain", "supply chain"), ("buyer", "supply chain"), ("facilities", "facilities"),
    ("warehouse", "warehouse"), ("inventory", "warehouse"), ("fulfillment", "fulfillment"), ("quality", "quality"), ("production technician", "manufacturing"),
    ("machine operator", "manufacturing"), ("nurse", "nursing"), ("clinical coordinator", "clinical operations"), ("clinical research", "clinical operations"),
    ("care coordinator", "care management"), ("research scientist", "discovery"), ("lab technician", "lab operations"), ("reporter", "news"),
    ("video producer", "video")]
REGIONS = {"emea": "EMEA", "apac": "APAC", "na": "NA", "north america": "NA", "latam": "LATAM", "americas": "NA"}

IC_MANAGER_PHRASES = ["customer success manager", "product manager", "account manager", "product marketing manager", "content marketing manager",
                      "marketing manager", "program manager", "project manager", "community manager", "office manager"]
EXEC_SUPPORT = ["executive assistant", "chief of staff", "executive business partner"]
FLUFF = [r"\|\s*hiring!?", r"\|\s*ex-\w+", r"\|\s*building the future", r"@\s*hq", r"✨", r"🚀"]

VOCAB = set()
for d in (FUNC_PHRASES, DEPT_MAP):
    for k in d:
        VOCAB.update(k.split())
for k in TEAM_PHRASES + IC_MANAGER_PHRASES + EXEC_SUPPORT:
    VOCAB.update(k.split())
VOCAB.update("senior junior associate principal staff lead head director manager vice president chief officer executive founder co-founder president "
             "interim acting founding team specialist analyst engineer coordinator technician scientist representative regional general group contract "
             "contractor remote".split())
VOCAB = {w for w in VOCAB if len(w) >= 4 and w.isalpha()}

LEVEL_LABEL = [(0, "CEO"), (1, "C-level"), (1.5, "SVP"), (2, "VP"), (2.7, "Senior director"), (3, "Director / head"), (3.8, "Senior manager"),
               (4, "Manager / lead"), (5, "Principal / staff IC"), (5.5, "Senior IC"), (6, "IC"), (6.5, "Junior IC")]


def level_label(lv):
    best = LEVEL_LABEL[0][1]
    for v, lab in LEVEL_LABEL:
        if lv >= v:
            best = lab
    return best


def _clean(raw, notes):
    t = raw.strip()
    low = t.lower()
    for pat in FLUFF:
        if re.search(pat, low):
            low = re.sub(pat, " ", low)
            notes.append("Dropped decoration from the title")
    contractor = False
    if "(contract)" in low or "contractor" in low:
        contractor = True
        low = low.replace("(contract)", " ").replace("contractor", " ")
        notes.append("Marked as a contractor")
    if re.search(r"\s-\s*remote\b", low):
        low = re.sub(r"\s-\s*remote\b", " ", low)
    for pre in ("interim ", "acting ", "founding "):
        if low.startswith(pre):
            low = low[len(pre):]
            notes.append(f"Ignored the word '{pre.strip()}'")
    if raw.isupper() and len(raw) > 4 and " " in raw:
        notes.append("Title was in all caps")
    elif raw.islower() and len(raw) > 4:
        notes.append("Title was all lower case")
    low = low.replace("v.p.", "vp").replace("head -", "head of").replace("head,", "head of")
    return low, contractor


def _tokens(low, notes):
    # split keeping & and - inside words
    low = re.sub(r"[|/()]", " , ", low)
    low = low.replace(",", " , ")
    toks = [x.strip(".") for x in low.split()]
    out = []
    expanded, fixed = [], []
    for i, w in enumerate(toks):
        if not w:
            continue
        if w == "eng":
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            prv = toks[i - 1] if i > 0 else ""
            rep = "engineering" if nxt in ("manager", "mgr", "director", "dir") or prv in ("of", "director", "dir", "vp", "head", "vice", "president", ",") else "engineer"
            out.extend(rep.split()); expanded.append(f"eng→{rep}"); continue
        if w == "dev":
            prv = toks[i - 1] if i > 0 else ""
            rep = "developer" if prv in ("software", "bi") else "development"
            out.append(rep); expanded.append(f"dev→{rep}"); continue
        if w == "tech":
            prv = toks[i - 1] if i > 0 else ""
            rep = "technician" if prv in ("lab", "production", "help") else "tech"
            if rep != w:
                expanded.append("tech→technician")
            out.append(rep); continue
        if w in ("ii", "iii", "iv", "i") and i == len(toks) - 1:
            out.append({"i": "<junior>", "ii": "<mid>", "iii": "<senior>", "iv": "<senior>"}[w]); expanded.append(f"level '{w.upper()}'"); continue
        if w in WORD_EXPAND and WORD_EXPAND[w] != w:
            out.extend(WORD_EXPAND[w].split()); expanded.append(f"{w}→{WORD_EXPAND[w]}"); continue
        if w.isalpha() and len(w) >= 5 and w not in VOCAB and w not in CSUITE:
            m = difflib.get_close_matches(w, VOCAB, n=1, cutoff=0.75)
            if m:
                fixed.append(f"{w}→{m[0]}")
                out.append(m[0]); continue
        out.append(w)
    if expanded:
        notes.append("Expanded " + ", ".join(expanded[:4]))
    if fixed:
        notes.append("Fixed likely typo " + ", ".join(fixed))
    return out


def parse_record(rec):
    """rec: dict with title, department, location, start_date. Returns parsed dict."""
    notes = []
    raw = rec.get("title") or ""
    dept_raw = (rec.get("department") or "").strip().lower()
    dept_funcs = DEPT_MAP.get(dept_raw, [])
    out = dict(norm="", level=None, level_label="Unknown", func=None, func_source=None, teams=[], region=None,
               manager_capable=False, exec_support=False, contractor=False, notes=notes, vague=False, ambiguous_func=None)
    if not raw.strip():
        notes.append("No title on the record")
        out["func"] = dept_funcs[0] if len(dept_funcs) == 1 else None
        if out["func"]:
            out["func_source"] = "department"
        return out
    low, contractor = _clean(raw, notes)
    out["contractor"] = contractor
    toks = _tokens(low, notes)
    norm = " ".join(t for t in toks if t != ",")
    norm = re.sub(r"\s+", " ", norm).strip()
    padded = f" {norm} "
    out["norm"] = norm

    # ---- level ----
    lvl = None
    cs_hit = [t for t in toks if t in CSUITE]
    if "ceo" in toks or "chief executive" in norm or ("president" in toks and "vice" not in toks and ("ceo" in toks or "chief" in toks)):
        lvl = 0
    elif "chief of staff" in norm:
        lvl = 5
    elif cs_hit or re.search(r"\bchief\b.*\bofficer\b", norm) or "general counsel" in norm:
        lvl = 1
    elif "president" in toks and "vice" not in toks:
        lvl = 1
    elif "executive vice president" in norm or "senior vice president" in norm:
        lvl = 1.5
    elif "vice president" in norm:
        lvl = 2
    elif "senior director" in norm:
        lvl = 2.7
    elif "head of" in norm or norm.startswith("head ") or "director" in toks:
        lvl = 3
    elif ("senior manager" in norm or ("senior" in toks and "manager" in toks)) and not any(p in norm for p in IC_MANAGER_PHRASES):
        lvl = 3.8
    elif "managing editor" in norm or "general manager" in norm:
        lvl = 4
    elif "manager" in toks and not any(p in norm for p in IC_MANAGER_PHRASES):
        lvl = 4
    elif "group product manager" in norm:
        lvl = 4
    elif re.search(r"\blead\b\s*(,|$)", " ".join(toks)) or "team lead" in norm or "product lead" in norm:
        lvl = 4
    if "group product manager" in norm:
        lvl = 4
    if any(p in norm for p in EXEC_SUPPORT):
        out["exec_support"] = True
        lvl = 5
        notes.append("Executive support role: usually sits with the CEO or an executive")
    if lvl is None:
        if "principal" in toks or "staff" in toks[:1]:
            lvl = 5
        elif toks and toks[0] == "lead":
            lvl = 5.2
        elif "senior" in toks or "<senior>" in toks:
            lvl = 5.5
        elif "junior" in toks or "associate" in toks[:1] or "<junior>" in toks or "intern" in toks:
            lvl = 6.5
        else:
            lvl = 6
        if any(p in norm for p in IC_MANAGER_PHRASES):
            notes.append("'Manager' here is an individual-contributor title, not a people manager")
    if lvl == 0 and "founder" in norm:
        pass
    if "co-founder" in toks and lvl is None:
        lvl = 1
    out["level"] = lvl
    out["level_label"] = level_label(lvl)
    out["manager_capable"] = lvl <= 4

    # ---- function ----
    scores = {}
    ambiguous = None
    if lvl <= 1:
        for t in cs_hit:
            f, alts = CSUITE[t]
            if alts:
                pick = next((a for a in alts if a in dept_funcs), None)
                if pick:
                    scores[pick] = scores.get(pick, 0) + 6
                    notes.append(f"'{t.upper()}' is ambiguous; department '{rec.get('department')}' points to {pick}")
                else:
                    scores[f] = scores.get(f, 0) + 4
                    ambiguous = alts
                    notes.append(f"'{t.upper()}' is ambiguous ({' / '.join(alts)}); assumed {f}")
            else:
                scores[f] = scores.get(f, 0) + 6
        m = re.search(r"chief (.+?) officer", norm)
        if m:
            words = m.group(1)
            for k in sorted(CHIEF_WORD, key=len, reverse=True):
                if k in words:
                    scores[CHIEF_WORD[k]] = scores.get(CHIEF_WORD[k], 0) + 6
                    break
        if "general counsel" in norm:
            scores["legal"] = scores.get("legal", 0) + 6
    if not scores:
        used = padded
        for phrase in sorted(FUNC_PHRASES, key=len, reverse=True):
            p = f" {phrase} "
            if p in used:
                f, w = FUNC_PHRASES[phrase]
                scores[f] = scores.get(f, 0) + w
                used = used.replace(p, "  ", 1)
        if " it " in padded and "it" in toks:
            scores["it"] = scores.get("it", 0) + 1
    if "ceo" in toks or lvl == 0:
        scores = {"executive": 10}
    for f in dept_funcs:
        if f in scores:
            scores[f] += 1.5
    if scores:
        best = max(scores.values())
        top = [f for f, s in scores.items() if s == best]
        if len(top) > 1:
            pick = next((f for f in top if f in dept_funcs), top[0])
            notes.append(f"Title fits {' or '.join(top)}; chose {pick}" + (" from the department" if pick in dept_funcs else ""))
        else:
            pick = top[0]
        out["func"] = pick
        out["func_source"] = "title+department" if pick in dept_funcs else "title"
    elif len(dept_funcs) >= 1:
        out["func"] = dept_funcs[0]
        out["func_source"] = "department"
        notes.append(f"Title says nothing about function; used department '{rec.get('department')}'")
    else:
        out["vague"] = True
        notes.append("Title is too generic to tell the function" + ("" if dept_raw else ", and there is no department"))
    out["ambiguous_func"] = ambiguous

    # ---- teams ----
    teams = set()
    s = padded
    for a, b in TEAM_ALIAS.items():
        s = s.replace(f" {a} ", f" {b} ")
    for phrase in sorted(TEAM_PHRASES, key=len, reverse=True):
        if f" {phrase} " in s:
            teams.add(phrase)
            s = s.replace(f" {phrase} ", "  ")
    for role, tm in ROLE_TEAM:
        if role in norm:
            teams.add(tm)
    # 'product' alone, 'support', 'content' are noisy when they are the function word of a leader
    if out["level"] is not None and out["level"] <= 3 and teams and len(teams) == 1:
        pass
    out["teams"] = sorted(teams)
    for k, v in REGIONS.items():
        if re.search(rf"(^|\s){re.escape(k)}(\s|$)", norm):
            out["region"] = v
    return out


LOC_ALIAS = {"nyc": "New York", "new york, ny": "New York", "new york": "New York", "sf": "San Francisco", "san francisco bay area": "San Francisco",
             "sf bay area": "San Francisco", "bangalore": "Bengaluru", "bengaluru, ka": "Bengaluru", "london, uk": "London", "greater london": "London",
             "gurgaon": "Gurugram", "delhi ncr": "Gurugram", "sao paulo": "São Paulo", "cdmx": "Mexico City", "ciudad de méxico": "Mexico City"}
CITY_REGION = {"New York": "NA", "San Francisco": "NA", "Austin": "NA", "Chicago": "NA", "Toronto": "NA", "Boston": "NA", "Seattle": "NA",
               "London": "EMEA", "Berlin": "EMEA", "Dublin": "EMEA", "Amsterdam": "EMEA", "Paris": "EMEA", "Bengaluru": "APAC", "Singapore": "APAC",
               "Sydney": "APAC", "Gurugram": "APAC", "Tokyo": "APAC", "São Paulo": "LATAM", "Mexico City": "LATAM", "Buenos Aires": "LATAM"}


def norm_location(loc):
    l = (loc or "").strip()
    if not l or l.lower() == "remote":
        return None
    return LOC_ALIAS.get(l.lower(), l)
