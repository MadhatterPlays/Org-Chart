"""Generate 520 synthetic companies with a hidden true org chart and messy,
model-visible employee profiles.

Outputs (in data/):
  profiles.csv   what the model sees: company_id, person_id, name, title, department, location, start_date
  truth.csv      hidden: person_id, manager_id, function, team, level, clean_title, mess
  companies.csv  company_id, name, industry, size_band, employees, offices, messiness
"""
import csv, json, math, random, os
from collections import Counter

SEED = 20261027
N_COMPANIES = 520

INDUSTRIES = {
    # function weights per industry
    "SaaS":          dict(engineering=30, data=7, security=4, it=2, product=7, design=4, sales=16, cs=9, marketing=8, finance=4, hr=4, legal=2, ops=3),
    "Fintech":       dict(engineering=26, data=8, security=6, it=3, product=6, design=3, sales=12, cs=9, marketing=6, finance=8, hr=4, legal=5, ops=4),
    "Healthcare":    dict(engineering=10, data=5, security=3, it=5, product=3, design=1, sales=8, cs=8, marketing=5, finance=6, hr=6, legal=4, ops=10, clinical=26),
    "Retail & E-commerce": dict(engineering=14, data=7, security=2, it=4, product=5, design=4, sales=6, cs=12, marketing=12, finance=6, hr=5, legal=2, ops=21),
    "Manufacturing": dict(engineering=10, data=3, security=2, it=5, product=3, design=2, sales=10, cs=5, marketing=4, finance=6, hr=6, legal=2, ops=42),
    "Logistics":     dict(engineering=12, data=6, security=2, it=5, product=3, design=1, sales=9, cs=10, marketing=3, finance=6, hr=6, legal=2, ops=35),
    "Media":         dict(engineering=14, data=6, security=2, it=3, product=5, design=8, sales=14, cs=4, marketing=16, finance=5, hr=5, legal=4, ops=4, editorial=10),
    "Cybersecurity": dict(engineering=26, data=4, security=16, it=2, product=6, design=2, sales=16, cs=8, marketing=7, finance=4, hr=4, legal=2, ops=3),
    "Biotech":       dict(engineering=6, data=8, security=2, it=4, product=2, design=1, sales=5, cs=2, marketing=4, finance=6, hr=5, legal=6, ops=9, research=40),
    "Telecom":       dict(engineering=20, data=6, security=5, it=8, product=5, design=2, sales=14, cs=14, marketing=7, finance=5, hr=5, legal=3, ops=6),
    "Insurance":     dict(engineering=12, data=9, security=4, it=6, product=4, design=2, sales=16, cs=14, marketing=6, finance=10, hr=5, legal=6, ops=6),
    "EdTech":        dict(engineering=22, data=6, security=2, it=2, product=8, design=6, sales=12, cs=10, marketing=10, finance=4, hr=5, legal=2, ops=4, editorial=7),
}

# Function catalogue. teams: team -> list of (IC role, weight)
F = {
 "engineering": dict(name="Engineering", c="Chief Technology Officer", depts=["Engineering", "R&D", "Tech", "Product & Engineering"],
    mgr="Engineering Manager", director="Director of Engineering",
    seniority=[("Junior", 12), ("", 40), ("Senior", 30), ("Staff", 12), ("Principal", 6)],
    teams={"Platform": [("Software Engineer", 1)], "Backend": [("Software Engineer", 1)], "Frontend": [("Frontend Engineer", 1), ("Software Engineer", 1)],
           "Mobile": [("Mobile Engineer", 1), ("iOS Engineer", 1), ("Android Engineer", 1)], "Infrastructure": [("DevOps Engineer", 1), ("Software Engineer", 1)],
           "QA": [("QA Engineer", 2), ("Test Automation Engineer", 1)], "Site Reliability": [("Site Reliability Engineer", 1)], "Payments": [("Software Engineer", 1)]}),
 "data": dict(name="Data", c="Chief Data Officer", depts=["Data", "Analytics", "Data & Analytics", "Engineering"],
    mgr="Data Manager", director="Director of Data",
    seniority=[("Junior", 12), ("", 45), ("Senior", 30), ("Lead", 8), ("Principal", 5)],
    teams={"Data Engineering": [("Data Engineer", 1)], "Analytics": [("Data Analyst", 2), ("Analytics Engineer", 1)], "Data Science": [("Data Scientist", 1)],
           "Machine Learning": [("Machine Learning Engineer", 1)], "Business Intelligence": [("BI Analyst", 1), ("BI Developer", 1)]}),
 "security": dict(name="Security", c="Chief Information Security Officer", depts=["Security", "InfoSec", "Information Security", "Engineering"],
    mgr="Security Manager", director="Director of Security",
    seniority=[("Junior", 10), ("", 45), ("Senior", 35), ("Principal", 10)],
    teams={"Security Operations": [("Security Analyst", 1), ("SOC Analyst", 1)], "Application Security": [("Application Security Engineer", 1), ("Security Engineer", 1)],
           "GRC": [("GRC Analyst", 1), ("Compliance Analyst", 1)], "Identity & Access": [("IAM Engineer", 1)]}),
 "it": dict(name="IT", c="Chief Information Officer", depts=["IT", "Information Technology", "Corporate IT"],
    mgr="IT Manager", director="Director of IT",
    seniority=[("Junior", 15), ("", 55), ("Senior", 30)],
    teams={"IT Support": [("IT Support Specialist", 2), ("Help Desk Technician", 1)], "Corporate Systems": [("Systems Administrator", 1), ("Network Engineer", 1)]}),
 "product": dict(name="Product", c="Chief Product Officer", depts=["Product", "Product Management", "Product & Engineering"],
    mgr="Group Product Manager", director="Director of Product",
    seniority=[("Associate", 15), ("", 50), ("Senior", 30), ("Principal", 5)],
    teams={"Core Product": [("Product Manager", 1)], "Growth": [("Product Manager", 1), ("Product Analyst", 1)], "Platform Product": [("Product Manager", 1)]}),
 "design": dict(name="Design", c=None, depts=["Design", "UX", "Product", "Creative"],
    mgr="Design Manager", director="Director of Design",
    seniority=[("Junior", 15), ("", 50), ("Senior", 30), ("Lead", 5)],
    teams={"Product Design": [("Product Designer", 2), ("UX Designer", 1)], "UX Research": [("UX Researcher", 1)], "Brand Design": [("Graphic Designer", 1), ("Brand Designer", 1)]}),
 "sales": dict(name="Sales", c="Chief Revenue Officer", depts=["Sales", "Revenue", "Go-To-Market", "Business Development"],
    mgr="Sales Manager", director="Director of Sales",
    seniority=[("Associate", 10), ("", 60), ("Senior", 30)],
    teams={"Enterprise": [("Enterprise Account Executive", 2), ("Account Executive", 1)], "Mid-Market": [("Account Executive", 1)], "SMB": [("Account Executive", 1)],
           "Sales Development": [("Sales Development Representative", 2), ("Business Development Representative", 1)],
           "Sales Operations": [("Sales Operations Analyst", 1)], "Solutions Engineering": [("Solutions Engineer", 1), ("Sales Engineer", 1)]}),
 "cs": dict(name="Customer Success", c="Chief Customer Officer", depts=["Customer Success", "Customer Support", "Support", "Customer Experience"],
    mgr="Customer Success Team Lead", director="Director of Customer Success",
    seniority=[("Associate", 10), ("", 65), ("Senior", 25)],
    teams={"Customer Success": [("Customer Success Manager", 1)], "Support": [("Support Specialist", 2), ("Technical Support Engineer", 1)], "Onboarding": [("Implementation Specialist", 1), ("Onboarding Specialist", 1)]}),
 "marketing": dict(name="Marketing", c="Chief Marketing Officer", depts=["Marketing", "Growth Marketing", "Brand & Communications"],
    mgr="Marketing Team Lead", director="Director of Marketing",
    seniority=[("Associate", 15), ("", 60), ("Senior", 25)],
    teams={"Demand Generation": [("Demand Generation Specialist", 1), ("Marketing Manager", 1)], "Content": [("Content Writer", 1), ("Content Marketing Manager", 1)],
           "Product Marketing": [("Product Marketing Manager", 1)], "Brand": [("Brand Specialist", 1), ("Communications Specialist", 1)], "Events": [("Events Coordinator", 1)]}),
 "finance": dict(name="Finance", c="Chief Financial Officer", depts=["Finance", "Finance & Accounting", "Accounting"],
    mgr="Finance Manager", director="Director of Finance",
    seniority=[("Junior", 10), ("", 60), ("Senior", 30)],
    teams={"Accounting": [("Accountant", 2), ("Staff Accountant", 1)], "FP&A": [("Financial Analyst", 1)], "Procurement": [("Procurement Specialist", 1)], "Payroll": [("Payroll Specialist", 1)]}),
 "hr": dict(name="People", c="Chief People Officer", depts=["People", "HR", "Human Resources", "People & Culture", "Talent"],
    mgr="HR Manager", director="Director of People",
    seniority=[("Associate", 15), ("", 60), ("Senior", 25)],
    teams={"Talent Acquisition": [("Recruiter", 2), ("Technical Recruiter", 1), ("Sourcer", 1)], "People Operations": [("HR Generalist", 1), ("People Operations Specialist", 1)],
           "Learning & Development": [("L&D Specialist", 1)], "Total Rewards": [("Compensation Analyst", 1)]}),
 "legal": dict(name="Legal", c="Chief Legal Officer", depts=["Legal", "Legal & Compliance", "Compliance"],
    mgr="Legal Manager", director="Director of Legal",
    seniority=[("", 70), ("Senior", 30)],
    teams={"Commercial Legal": [("Corporate Counsel", 1), ("Paralegal", 1)], "Compliance": [("Compliance Specialist", 1)]}),
 "ops": dict(name="Operations", c="Chief Operating Officer", depts=["Operations", "Ops", "Business Operations", "Supply Chain"],
    mgr="Operations Manager", director="Director of Operations",
    seniority=[("Junior", 10), ("", 70), ("Senior", 20)],
    teams={"Business Operations": [("Operations Analyst", 1)], "Supply Chain": [("Supply Chain Analyst", 1), ("Buyer", 1)], "Facilities": [("Facilities Coordinator", 1)],
           "Warehouse": [("Warehouse Associate", 3), ("Inventory Specialist", 1)], "Fulfillment": [("Fulfillment Associate", 2)], "Quality": [("Quality Inspector", 1), ("Quality Engineer", 1)],
           "Manufacturing": [("Production Technician", 3), ("Machine Operator", 2)]}),
 "clinical": dict(name="Clinical", c="Chief Medical Officer", depts=["Clinical", "Clinical Operations", "Medical"],
    mgr="Clinical Manager", director="Director of Clinical Operations",
    seniority=[("", 75), ("Senior", 25)],
    teams={"Nursing": [("Registered Nurse", 3), ("Nurse Practitioner", 1)], "Clinical Operations": [("Clinical Coordinator", 1), ("Clinical Research Associate", 1)], "Care Management": [("Care Coordinator", 1)]}),
 "research": dict(name="Research", c="Chief Scientific Officer", depts=["R&D", "Research", "Discovery"],
    mgr="Research Manager", director="Director of Research",
    seniority=[("", 55), ("Senior", 30), ("Principal", 15)],
    teams={"Discovery": [("Research Scientist", 2), ("Research Associate", 2)], "Translational": [("Scientist", 1), ("Research Associate", 1)], "Lab Operations": [("Lab Technician", 2)]}),
 "editorial": dict(name="Editorial", c="Chief Content Officer", depts=["Editorial", "Content", "Newsroom"],
    mgr="Managing Editor", director="Editorial Director",
    seniority=[("Junior", 15), ("", 60), ("Senior", 25)],
    teams={"News": [("Reporter", 2), ("Editor", 1)], "Features": [("Writer", 1), ("Editor", 1)], "Video": [("Video Producer", 1)]}),
}

# Which leader a non-CEO-reporting function head reports to (truth)
REPORTING_OPTIONS = {
    "data":     [("engineering", 50), ("product", 20), ("ceo", 30)],
    "security": [("engineering", 45), ("it", 20), ("ceo", 35)],
    "it":       [("finance", 35), ("ops", 30), ("engineering", 20), ("ceo", 15)],
    "design":   [("product", 70), ("engineering", 15), ("ceo", 15)],
    "cs":       [("sales", 40), ("ops", 25), ("ceo", 35)],
    "legal":    [("ceo", 60), ("finance", 40)],
    "hr":       [("ceo", 70), ("ops", 20), ("finance", 10)],
    "product":  [("ceo", 85), ("engineering", 15)],
    "marketing":[("ceo", 75), ("sales", 25)],
}

CITIES = [("New York", "NA"), ("San Francisco", "NA"), ("Austin", "NA"), ("Chicago", "NA"), ("Toronto", "NA"), ("Boston", "NA"), ("Seattle", "NA"),
          ("London", "EMEA"), ("Berlin", "EMEA"), ("Dublin", "EMEA"), ("Amsterdam", "EMEA"), ("Paris", "EMEA"),
          ("Bengaluru", "APAC"), ("Singapore", "APAC"), ("Sydney", "APAC"), ("Gurugram", "APAC"), ("Tokyo", "APAC"),
          ("São Paulo", "LATAM"), ("Mexico City", "LATAM"), ("Buenos Aires", "LATAM")]
CITY_ALIASES = {"New York": ["NYC", "New York, NY", "new york"], "San Francisco": ["SF", "San Francisco Bay Area", "SF Bay Area"],
                "Bengaluru": ["Bangalore", "Bengaluru, KA"], "London": ["London, UK", "Greater London"], "Gurugram": ["Gurgaon", "Delhi NCR"],
                "São Paulo": ["Sao Paulo"], "Mexico City": ["CDMX", "Ciudad de México"]}

FIRST = ("James Mary Robert Patricia John Jennifer Michael Linda David Elizabeth William Barbara Richard Susan Joseph Jessica Thomas Sarah Charles Karen "
         "Daniel Nancy Matthew Lisa Anthony Betty Mark Sandra Steven Ashley Andrew Kimberly Joshua Emily Kevin Donna Brian Michelle Priya Aarav Ananya Rohan "
         "Isha Vikram Neha Arjun Kavya Rahul Sneha Aditya Meera Karthik Divya Wei Mei Hiroshi Yuki Jin Soo-ah Min-jun Li Chen Hana Sofia Mateo Valentina "
         "Santiago Camila Diego Lucia Javier Elena Carlos Ana Lukas Emma Jonas Mia Leon Hannah Felix Clara Noah Olivia Liam Ava Ethan Chloe Omar Fatima Amir "
         "Layla Yusuf Zara Kwame Amara Chidi Nia Tunde Zainab Lars Ingrid Sven Freya Pierre Camille Luc Margaux Giulia Marco Sara Luca Ahmed Noor Ravi Pooja").split()
LAST = ("Smith Johnson Williams Brown Jones Garcia Miller Davis Rodriguez Martinez Hernandez Lopez Gonzalez Wilson Anderson Thomas Taylor Moore Jackson Martin "
        "Lee Perez Thompson White Harris Sanchez Clark Ramirez Lewis Robinson Walker Young Allen King Wright Scott Torres Nguyen Hill Flores Sharma Patel Gupta "
        "Iyer Reddy Nair Mehta Kapoor Singh Verma Rao Bose Chatterjee Menon Wang Zhang Liu Chen Yang Huang Tanaka Suzuki Sato Kim Park Choi Silva Santos Oliveira "
        "Costa Fernandez Romero Alvarez Muller Schmidt Schneider Fischer Weber Becker Dubois Moreau Laurent Rossi Russo Ferrari Bianchi Okafor Mensah Adeyemi "
        "Hassan Khan Ali Rahman Haddad Johansson Larsen Nielsen Andersson Novak Kowalski Ivanova Petrov Murphy Kelly O'Brien Byrne Cohen Levi Friedman Dewan").split()

NAME_A = "North Blue Iron Bright Clear Silver Summit Harbor Cedar Quantum Atlas Nova Vertex Lumen Polar Copper Granite Orbit Pioneer Meridian Cobalt Beacon Ember Halcyon Juniper Keystone Lattice Monarch Nimbus Onyx Prism Redwood Sable Tidal Union Vantage Willow Zenith Aster Boreal Crest Delta Echo Fable Grove Helix Indigo Jade Kestrel Lotus".split()
NAME_B = {"SaaS": ["Cloud", "Stack", "Labs", "Systems", "Software", "Works"], "Fintech": ["Pay", "Capital", "Ledger", "Finance", "Money", "Bank"],
          "Healthcare": ["Health", "Care", "Clinics", "Medical", "Wellness"], "Retail & E-commerce": ["Goods", "Market", "Supply", "Store", "Commerce"],
          "Manufacturing": ["Industries", "Manufacturing", "Fabrication", "Components", "Materials"], "Logistics": ["Freight", "Logistics", "Transit", "Shipping", "Cargo"],
          "Media": ["Media", "Studios", "Press", "Broadcasting", "Publishing"], "Cybersecurity": ["Security", "Shield", "Defense", "Guard", "Cyber"],
          "Biotech": ["Bio", "Therapeutics", "Genomics", "Biosciences", "Pharma"], "Telecom": ["Telecom", "Networks", "Wireless", "Connect", "Fiber"],
          "Insurance": ["Insurance", "Assurance", "Mutual", "Underwriters", "Risk"], "EdTech": ["Learning", "Academy", "Education", "Classroom", "Tutor"]}

# ---------- mess ----------
ABBREV = [("Vice President", ["VP"]), ("VP of ", ["VP ", "VP, ", "V.P. "]), ("Senior Vice President", ["SVP", "Sr. VP"]),
          ("Senior ", ["Sr. ", "Sr ", "Snr "]), ("Junior ", ["Jr. ", "Jr "]), ("Manager", ["Mgr", "Mgr."]), ("Director", ["Dir.", "Dir"]),
          ("Software Engineer", ["SWE", "Software Eng", "Software Dev"]), ("Engineering", ["Eng", "Eng."]), ("Engineer", ["Eng", "Engr"]),
          ("Account Executive", ["AE", "Acct Exec", "Account Exec"]), ("Sales Development Representative", ["SDR", "Sales Dev Rep"]),
          ("Business Development Representative", ["BDR"]), ("Customer Success Manager", ["CSM"]), ("Customer Success", ["CS"]),
          ("Human Resources", ["HR"]), ("Operations", ["Ops"]), ("Marketing", ["Mktg"]), ("Associate ", ["Assoc. "]), ("Product Manager", ["PM"]),
          ("Machine Learning", ["ML"]), ("Quality Assurance", ["QA"]), ("Development", ["Dev"]), ("Information Technology", ["IT"]),
          ("Chief Technology Officer", ["CTO"]), ("Chief Financial Officer", ["CFO"]), ("Chief Executive Officer", ["CEO"]), ("Chief Operating Officer", ["COO"]),
          ("Chief Marketing Officer", ["CMO"]), ("Chief Revenue Officer", ["CRO"]), ("Chief Information Security Officer", ["CISO"]), ("Chief Information Officer", ["CIO"]),
          ("Chief Data Officer", ["CDO"]), ("Chief Product Officer", ["CPO"]), ("Chief People Officer", ["CPO", "CHRO"]), ("Chief Legal Officer", ["CLO", "General Counsel"]),
          ("Chief Customer Officer", ["CCO"]), ("Chief Scientific Officer", ["CSO"]), ("Chief Medical Officer", ["CMO"]), ("Chief Content Officer", ["CCO"]),
          ("Site Reliability Engineer", ["SRE"]), ("Head of ", ["Head - ", "Head, "]), ("Representative", ["Rep"]), ("Technician", ["Tech"])]
TRANSLATE = {"Software Engineer": "Ingeniero de Software", "Senior Software Engineer": "Ingeniero de Software Senior", "Account Executive": "Ejecutivo de Cuentas",
             "Sales Manager": "Gerente de Ventas", "Data Analyst": "Analista de Datos", "Recruiter": "Reclutador", "Accountant": "Contador",
             "Support Specialist": "Especialista de Soporte", "Engineering Manager": "Gerente de Ingeniería", "Product Designer": "Diseñador de Producto",
             "Data Engineer": "Engenheiro de Dados", "Financial Analyst": "Analista Financeiro", "Marketing Manager": "Gerente de Marketing"}
VAGUE = ["Team Member", "Consultant", "Specialist", "Associate", "Staff", "Employee", "Contributor", "Analyst"]
FLUFF_SUFFIX = [" | Hiring!", " | Ex-Amazon", " (Contract)", " ✨", " | Building the future", " @ HQ", " - Remote"]
FLUFF_PREFIX = ["Interim ", "Acting ", "Founding "]


def wchoice(rng, pairs):
    tot = sum(w for _, w in pairs)
    r = rng.random() * tot
    for v, w in pairs:
        r -= w
        if r <= 0:
            return v
    return pairs[-1][0]


class Company:
    def __init__(self, rng, cid):
        self.rng = rng
        self.cid = cid
        self.industry = rng.choice(list(INDUSTRIES))
        band = wchoice(rng, [("small", 45), ("mid", 40), ("large", 15)])
        self.band = band
        self.target = {"small": rng.randint(100, 140), "mid": rng.randint(141, 260), "large": rng.randint(261, 450)}[band]
        n_off = {"small": wchoice(rng, [(1, 55), (2, 35), (3, 10)]), "mid": wchoice(rng, [(1, 30), (2, 40), (3, 25), (4, 5)]),
                 "large": wchoice(rng, [(2, 25), (3, 40), (4, 35)])}[band]
        self.offices = rng.sample(CITIES, n_off)
        self.messiness = round(rng.uniform(0.5, 1.6), 2)
        self.name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B[self.industry])}"
        self.nodes = []
        self.used_names = Counter()

    # --- nodes ---
    def person_name(self):
        rng = self.rng
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"

    def add(self, title, func, team, level, manager, loc=None, region=None):
        node = dict(id=len(self.nodes), title=title, func=func, team=team, level=level, manager=manager,
                    loc=loc or self.office_for(manager), region=region, name=self.person_name())
        self.nodes.append(node)
        return node

    def office_for(self, manager):
        rng = self.rng
        if manager is None or rng.random() < 0.2:
            return rng.choice(self.offices)[0]
        return self.nodes[manager]["loc"]

    def region_of(self, city):
        for c, r in CITIES:
            if c == city:
                return r
        return None

    # --- structure ---
    def build(self):
        rng = self.rng
        w = INDUSTRIES[self.industry]
        funcs = [f for f, wt in w.items() if wt >= 4 or rng.random() < 0.6]
        if self.band == "small":
            funcs = [f for f in funcs if w[f] >= 3 or rng.random() < 0.4]
        ceo_title = wchoice(rng, [("Chief Executive Officer", 50), ("Co-Founder & CEO", 20), ("Founder and CEO", 10), ("CEO & President", 10), ("President and Chief Executive Officer", 10)])
        ceo = self.add(ceo_title, "executive", None, "CEO", None)
        overhead = 1
        if rng.random() < {"small": 0.2, "mid": 0.45, "large": 0.75}[self.band]:
            self.add("Chief of Staff", "executive", None, "Exec support", ceo["id"], loc=ceo["loc"]); overhead += 1
        if rng.random() < {"small": 0.3, "mid": 0.55, "large": 0.85}[self.band]:
            self.add(rng.choice(["Executive Assistant to the CEO", "Executive Assistant", "EA to CEO", "Executive Business Partner"]), "executive", None, "Exec support", ceo["id"], loc=ceo["loc"]); overhead += 1
        remaining = self.target - overhead
        tot = sum(w[f] for f in funcs)
        heads = {f: max(3, int(round(remaining * w[f] / tot))) for f in funcs}
        diff = remaining - sum(heads.values())
        big = max(heads, key=heads.get)
        heads[big] += diff
        # decide who each function head reports to
        reports_to = {}
        for f in funcs:
            opts = [(o, wt) for o, wt in REPORTING_OPTIONS.get(f, [("ceo", 100)]) if o == "ceo" or o in funcs]
            reports_to[f] = wchoice(rng, opts) if opts else "ceo"
        # avoid chains deeper than 1 (a function reporting to a function that itself reports to non-CEO)
        for f in funcs:
            p = reports_to[f]
            if p != "ceo" and reports_to.get(p, "ceo") != "ceo":
                reports_to[f] = "ceo"
        order = sorted(funcs, key=lambda f: 0 if reports_to[f] == "ceo" else 1)
        head_node = {}
        for f in order:
            parent = ceo["id"] if reports_to[f] == "ceo" else head_node[reports_to[f]]["id"]
            node = self.make_head(f, parent, reports_to[f] == "ceo", heads[f])
            head_node[f] = node
            self.build_function(node, f, heads[f] - 1)
        self.funcs = funcs
        self.reports_to = reports_to

    def make_head(self, f, parent, to_ceo, hc):
        rng, spec = self.rng, F[f]
        nm = spec["name"]
        if to_ceo:
            if self.band == "large":
                kind = wchoice(rng, [("c", 60), ("svp", 20), ("vp", 20)])
            elif self.band == "mid":
                kind = wchoice(rng, [("c", 35), ("vp", 55), ("head", 10)])
            else:
                kind = wchoice(rng, [("c", 15), ("vp", 45), ("head", 40)])
            if spec["c"] is None and kind == "c":
                kind = "vp"
        else:
            kind = wchoice(rng, [("vp", 35), ("head", 45), ("director", 20)]) if hc > 15 else wchoice(rng, [("head", 55), ("director", 30), ("manager", 15)])
        if kind == "c":
            title, level = spec["c"], "C-level"
            if f == "engineering" and rng.random() < 0.2:
                title = "Co-founder & CTO"
            if f == "legal" and rng.random() < 0.5:
                title = "General Counsel"
            if f == "hr" and rng.random() < 0.4:
                title = "Chief Human Resources Officer"
        elif kind == "svp":
            title, level = rng.choice([f"SVP, {nm}", f"Senior Vice President of {nm}", f"SVP of {nm}"]), "SVP"
        elif kind == "vp":
            title, level = rng.choice([f"VP of {nm}", f"Vice President, {nm}", f"VP, {nm}", f"Vice President of {nm}"]), "VP"
        elif kind == "head":
            title, level = f"Head of {nm}", "Director"
        elif kind == "director":
            title, level = spec["director"], "Director"
        else:
            title, level = f"{nm} Manager" if f not in ("engineering",) else "Engineering Manager", "Manager"
        return self.add(title, f, None, level, parent)

    def build_function(self, head, f, H):
        rng, spec = self.rng, F[f]
        if H <= 0:
            return
        teams = list(spec["teams"])
        if f == "ops":
            pref = {"Manufacturing": ["Manufacturing", "Quality", "Supply Chain", "Warehouse"], "Retail & E-commerce": ["Fulfillment", "Warehouse", "Supply Chain", "Business Operations"],
                    "Logistics": ["Warehouse", "Fulfillment", "Supply Chain", "Business Operations"]}.get(self.industry, ["Business Operations", "Facilities", "Supply Chain"])
            teams = pref
        else:
            rng.shuffle(teams)
        if H <= 8:
            for _ in range(H):
                self.add_ic(f, rng.choice(teams), head["id"], with_team=False)
            return
        k = max(1, min(len(teams), int(round(H / 13))))
        if H > 18:
            k = max(k, 2)
        teams = teams[:k]
        parents = [head]
        group_of = {t: head for t in teams}
        if H > 90 and k >= 4:
            # insert two senior leaders under the function head
            half = k // 2
            groups = [teams[:half], teams[half:]]
            parents = []
            for g in groups:
                gname = " & ".join(g[:2])
                t = rng.choice([f"VP, {gname}", f"Senior Director, {gname}", f"Senior Director of {spec['name']}"])
                lvl = "VP" if t.startswith("VP") else "Director"
                node = self.add(t, f, g[0], lvl, head["id"])
                parents.append(node)
                for tm in g:
                    group_of[tm] = node
            H -= 2
        # split H across teams
        shares = [rng.uniform(0.6, 1.4) for _ in teams]
        s = sum(shares)
        sizes = [max(3, int(round(H * x / s))) for x in shares]
        sizes[0] += H - sum(sizes)
        for tm, sz in zip(teams, sizes):
            if sz > 0:
                self.build_team(group_of[tm], f, tm, sz)

    def build_team(self, parent, f, team, t):
        rng, spec = self.rng, F[f]
        if t <= 9:
            lead = self.add(self.mgr_title(f, team), f, team, "Manager", parent["id"])
            for _ in range(t - 1):
                self.add_ic(f, team, lead["id"])
            return
        if parent["level"] in ("Director", "Manager"):
            # leader of the function is only a director: team leads are senior managers
            dtitle = rng.choice([f"Senior Manager, {team}", f"Senior {F[f]['mgr']}, {team}", f"{team} Senior Manager"])
            dlevel = "Manager"
        else:
            dtitle = rng.choice([f"Director, {team}", f"Head of {team}", f"Director of {team}", f"Senior Manager, {team}" if t < 16 else f"Head of {team}"])
            dlevel = "Manager" if dtitle.startswith("Senior Manager") else "Director"
        lead = self.add(dtitle, f, team, dlevel, parent["id"])
        rest = t - 1
        n_direct_ic = rng.choice([0, 0, 1, 2]) if rest > 12 else 0
        for _ in range(n_direct_ic):
            self.add_ic(f, team, lead["id"], senior_only=True)
        rest -= n_direct_ic
        m = max(2, int(round(rest / 7.5)))
        rest -= m
        multi_region = len({self.region_of(c) for c, _ in self.offices}) > 1
        mgrs = []
        for i in range(m):
            loc = self.offices[i % len(self.offices)][0] if rng.random() < 0.7 else rng.choice(self.offices)[0]
            region = self.region_of(loc) if (f in ("sales", "cs") and multi_region) else None
            mgr = self.add(self.mgr_title(f, team, region), f, team, "Manager", lead["id"], loc=loc, region=region)
            mgrs.append(mgr)
        per = [rest // m + (1 if i < rest % m else 0) for i in range(m)]
        for mgr, n in zip(mgrs, per):
            for _ in range(n):
                self.add_ic(f, team, mgr["id"], region=mgr["region"])

    def mgr_title(self, f, team, region=None):
        rng, spec = self.rng, F[f]
        base = spec["mgr"]
        opts = [(f"{base}, {team}", 30), (f"{team} Manager", 20), (f"Manager, {team}", 20), (base, 15), (f"Team Lead, {team}", 8), (f"{team} Lead", 7)]
        if f == "sales":
            opts = [(f"Sales Manager, {team}", 30), (f"{team} Sales Manager", 25), ("Sales Manager", 20), (f"Manager, {team}", 15), (f"Regional Sales Manager", 10)]
            if team == "Sales Development":
                opts = [("SDR Manager", 40), ("Sales Development Manager", 40), ("Manager, Sales Development", 20)]
        if f == "marketing":
            opts = [(f"{team} Lead", 30), (f"Manager, {team}", 25), (f"Senior Marketing Manager", 15), (f"{team} Manager", 15), (f"{team} Team Lead", 15)]
        if f == "product":
            opts = [("Group Product Manager", 50), (f"Group Product Manager, {team}", 30), (f"Product Lead, {team}", 20)]
        t = wchoice(rng, opts)
        if region and rng.random() < 0.85:
            t = f"{t}, {region}" if rng.random() < 0.6 else f"{t} - {region}"
        return t

    def add_ic(self, f, team, mgr_id, with_team=True, senior_only=False, region=None):
        rng, spec = self.rng, F[f]
        role = wchoice(rng, spec["teams"][team])
        sen = wchoice(rng, spec["seniority"])
        if senior_only:
            sen = rng.choice(["Staff", "Principal", "Senior"]) if f == "engineering" else "Senior"
        title = f"{sen} {role}".strip()
        lvl = {"Junior": "IC-junior", "Associate": "IC-junior", "": "IC", "Senior": "IC-senior", "Staff": "IC-senior", "Principal": "IC-senior", "Lead": "IC-senior"}[sen]
        clean = title
        if with_team and team.split()[0] not in role and rng.random() < 0.55:
            clean = rng.choice([f"{title}, {team}", f"{title} - {team}", f"{title} ({team})", f"{team} {title}" if team in ("Enterprise", "Mid-Market", "SMB", "Platform", "Growth") else f"{title}, {team}"])
        if region and rng.random() < 0.5:
            clean = f"{clean}, {region}" if rng.random() < 0.5 else f"{clean} - {region}"
        node = self.add(clean, f, team, lvl, mgr_id, region=region)
        if rng.random() < 0.8:
            node["loc"] = self.nodes[mgr_id]["loc"]
        return node

    # --- profiles with mess ---
    def render(self):
        rng, m = self.rng, self.messiness
        dept_alias = {f: rng.choice(F[f]["depts"]) for f in F}
        profiles, truth = [], []
        mess_counter = Counter()
        # stale titles: a few managers still show their old IC title
        for n in self.nodes:
            mess = []
            t = n["title"]
            if n["level"] in ("Manager",) and rng.random() < 0.02 * m and n["func"] in F:
                role = wchoice(rng, F[n["func"]]["teams"].get(n["team"], [("Specialist", 1)]) if n["team"] in F[n["func"]]["teams"] else [("Specialist", 1)])
                t = f"Senior {role}"
                mess.append("stale_title")
            if rng.random() < 0.012 * m and n["level"] not in ("CEO",):
                t = rng.choice(VAGUE)
                mess.append("vague_title")
            if rng.random() < 0.01 and t in TRANSLATE and any(r in ("LATAM",) for _, r in self.offices):
                t = TRANSLATE[t]
                mess.append("non_english")
            # level numerals instead of words
            if "Engineer" in t and not any(x in t for x in ("Head", "Director", "Manager", "VP", "Vice", "Lead", "Chief")) and rng.random() < 0.12:
                if t.startswith("Senior "):
                    t = t.replace("Senior ", "", 1) + " III"; mess.append("level_numeral")
                elif t.startswith("Junior "):
                    t = t.replace("Junior ", "", 1) + " I"; mess.append("level_numeral")
                elif not t.split()[0] in ("Staff", "Principal") and "Manager" not in t and "Director" not in t:
                    t = t + " II"; mess.append("level_numeral")
            # abbreviations
            abbr = False
            for full, shorts in ABBREV:
                if full in t and rng.random() < (0.55 if full.startswith("Chief") else 0.3 * m):
                    t = t.replace(full, rng.choice(shorts), 1)
                    abbr = True
            if abbr:
                mess.append("abbreviation")
            # typo
            if rng.random() < 0.03 * m:
                words = t.split()
                idx = [i for i, wd in enumerate(words) if len(wd) >= 6 and wd.isalpha()]
                if idx:
                    i = rng.choice(idx)
                    wd = words[i]
                    j = rng.randrange(1, len(wd) - 1)
                    words[i] = wd[:j] + wd[j + 1] + wd[j] + wd[j + 2:] if rng.random() < 0.5 else wd[:j] + wd[j + 1:]
                    t = " ".join(words)
                    mess.append("typo")
            # fluff
            if rng.random() < 0.025 * m:
                t = t + rng.choice(FLUFF_SUFFIX); mess.append("fluff")
            if rng.random() < 0.01 and n["level"] not in ("CEO",):
                t = rng.choice(FLUFF_PREFIX) + t; mess.append("fluff")
            # case
            r = rng.random()
            if r < 0.05 * m:
                t = t.lower(); mess.append("case")
            elif r < 0.07 * m:
                t = t.upper(); mess.append("case")
            elif r < 0.09 * m:
                t = t.title(); mess.append("case")
            # missing title (never for CEO)
            if n["level"] != "CEO" and rng.random() < 0.015 * m:
                t = ""; mess.append("missing_title")
            # department
            if n["func"] == "executive":
                dept = rng.choice(["Executive", "Leadership", "Office of the CEO", ""])
            else:
                dept = dept_alias[n["func"]] if rng.random() < 0.85 else rng.choice(F[n["func"]]["depts"])
            if rng.random() < 0.25 * m:
                dept = ""; mess.append("missing_department")
            # location
            loc = n["loc"]
            if rng.random() < 0.08:
                loc = "Remote"; mess.append("remote")
            elif loc in CITY_ALIASES and rng.random() < 0.35:
                loc = rng.choice(CITY_ALIASES[loc]); mess.append("location_alias")
            if rng.random() < 0.15 * m:
                loc = ""; mess.append("missing_location")
            # start date: leaders tend to have been there longer
            base_years = {"CEO": 8, "C-level": 5, "SVP": 5, "VP": 4, "Director": 3.5, "Manager": 3, "IC-senior": 2.5, "IC": 2, "IC-junior": 1, "Exec support": 3}[n["level"]]
            yrs = max(0.05, rng.gauss(base_years, base_years * 0.5))
            months = int(yrs * 12) + 1
            total = 2026 * 12 + 9 - months  # months before Oct 2026
            start = f"{total // 12}-{total % 12 + 1:02d}"
            if rng.random() < 0.2 * m:
                start = ""; mess.append("missing_start_date")
            name = n["name"]
            self.used_names[name] += 1
            if self.used_names[name] > 1:
                mess.append("duplicate_name")
            if rng.random() < 0.02:
                name = name.upper() if rng.random() < 0.5 else name.split()[0][0] + ". " + name.split()[1]
                mess.append("name_format")
            for x in mess:
                mess_counter[x] += 1
            pid = f"{self.cid}-{n['id']:04d}"
            profiles.append(dict(company_id=self.cid, person_id=pid, name=name, title=t, department=dept, location=loc, start_date=start))
            truth.append(dict(person_id=pid, manager_id=(f"{self.cid}-{n['manager']:04d}" if n["manager"] is not None else ""), function=n["func"],
                              team=n["team"] or "", level=n["level"], clean_title=n["title"], mess="|".join(mess)))
        # shuffle row order like a real export
        idx = list(range(len(profiles)))
        rng.shuffle(idx)
        return [profiles[i] for i in idx], [truth[i] for i in idx], mess_counter


def main(out_dir):
    rng = random.Random(SEED)
    os.makedirs(out_dir, exist_ok=True)
    allp, allt, comps = [], [], []
    total_mess = Counter()
    for i in range(N_COMPANIES):
        cid = f"C{i + 1:03d}"
        c = Company(random.Random(rng.random()), cid)
        c.build()
        p, t, mc = c.render()
        total_mess.update(mc)
        allp += p
        allt += t
        comps.append(dict(company_id=cid, name=c.name, industry=c.industry, size_band=c.band, employees=len(p),
                          offices="|".join(o for o, _ in c.offices), messiness=c.messiness,
                          structure=json.dumps(c.reports_to)))
    # de-duplicate company names
    seen = Counter()
    for c in comps:
        seen[c["name"]] += 1
        if seen[c["name"]] > 1:
            c["name"] = f"{c['name']} {['Group', 'Inc', 'Co', 'Holdings', 'Partners', 'Global', 'International', 'Ventures'][seen[c['name']] % 8]}"
    for name, rows in (("profiles.csv", allp), ("truth.csv", allt), ("companies.csv", comps)):
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(comps)} companies, {len(allp)} people; min size {min(c['employees'] for c in comps)}")
    print(dict(total_mess.most_common()))


if __name__ == "__main__":
    main(os.path.join(os.path.dirname(__file__), "..", "data"))
