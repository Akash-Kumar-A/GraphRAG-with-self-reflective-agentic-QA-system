"""Step 1 - clean the public PSG corpus and generate SYNTHETIC confidential records.
Public corpus  -> trains the 'public LLM' (never private).
Synthetic recs -> the sensitive corpus D (one record = one student = one DP individual)."""
import re, json, random, glob, os
random.seed(7)
RAW, OUT = "data_raw", "prepared"
EMAIL = re.compile(r"\S+@\S+")

# ---------- public corpus ----------
pub, progs, faculty = [], [], []
for f in sorted(glob.glob(f"{RAW}/*.txt")):
    t = open(f, encoding="utf-8", errors="ignore").read().replace("\r\n", "\n").strip()
    if t.startswith("Akash kumar"): continue                      # team note, not PSG content
    t = EMAIL.sub("", t)                                          # emails never enter the corpus
    title = t.split("\n")[0].strip()
    if re.match(r"^(Dr|Mr|Mrs|Ms|Prof)\b", title):
        cat = "faculty"
        m = re.match(r"^((?:Dr|Mr|Mrs|Ms|Prof)\.?\s+[A-Za-z\. ]+?)\s+(?:is|was)\b", title)
        if m: faculty.append(re.sub(r"\.", "", m.group(1)).strip())
    elif re.match(r"^(B\.?E|B\.?Tech|BSc|M\.?E)\b", title):
        cat = "programme"
        progs.append(re.sub(r",?\s*since.*$", "", title).replace(".", "").replace(",", "").strip())
    else: cat = "institution"
    pub.append({"title": title[:80], "category": cat, "text": t})
progs = sorted(set(progs)); faculty = sorted(set(faculty))

# ---------- synthetic confidential records ----------
TOPICS = [
 ("medical leave", "leave request", ["viral fever","dengue fever","typhoid","surgery recovery","accident injury","chickenpox"]),
 ("fee waiver", "waiver request", ["family income loss","single parent","medical emergency of parent","natural calamity","job loss of guardian","education loan delay"]),
 ("scholarship", "scholarship request", ["merit based","first graduate","sports quota","community support","research grant","need based"]),
 ("counselling", "counselling request", ["exam anxiety","homesickness","career confusion","peer conflict","sleep issues","low motivation"]),
 ("hostel room change", "change request", ["noise disturbance","roommate conflict","maintenance issue","medical need","ground floor requirement","wifi problem"]),
 ("attendance condonation", "condonation request", ["hospitalisation","sports event","paper presentation","family bereavement","internship overlap","transport strike"]),
 ("disciplinary", "review request", ["late arrival","dress code violation","mobile phone misuse","lab misconduct","unauthorised absence","ragging complaint"]),
 ("internship noc", "noc request", ["summer internship","research internship","startup internship","industry project","remote internship","foreign internship"]),
 ("exam revaluation", "revaluation request", ["totalling error","unanswered section","illegible script","mismatched marks","answer key dispute","missing question"]),
 ("bonafide certificate", "certificate request", ["passport application","bank loan","visa application","scholarship portal","hostel admission","competitive exam"]),
 ("lab equipment loan", "loan request", ["capstone project","hackathon","research prototype","paper demo","club activity","startup pitch"]),
 ("transport pass", "pass request", ["new route","route change","duplicate pass","fee concession","late bus","day scholar"]),
]
SIZES = [22, 26, 32, 38, 44, 50, 56, 62, 68, 75, 82, 90]       # 20-29 ... 80+ relevant docs per topic
MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"]
OUTCOMES = ["approved","rejected","deferred","escalated","withdrawn"]
FIRST = "Arun Divya Karthik Meena Rahul Priya Vignesh Sneha Harish Kavya Naveen Anitha Surya Lakshmi Deepak Janani Manoj Pooja Ganesh Nisha".split()
LAST = "Kumar Raj Krishnan Subramanian Iyer Nair Murugan Selvam Prabhu Devi Sundaram Natarajan Ramesh Balaji".split()
FAC = [re.sub(r"[^A-Za-z ]", "", f) for f in faculty]
def skew(items, lo=0.5, hi=0.8):
    top = random.uniform(lo, hi); items = items[:]; random.shuffle(items)
    w = [top] + [(1 - top) * (0.55 ** i) for i in range(len(items) - 1)]
    s = sum(w); return items, [x / s for x in w]
def draw(items, w): return random.choices(items, w)[0]
def fmt(r): return (f"Student {r['name']} ({r['roll']}) of {r['prog']} submitted a {r['topic']} request in the month of {r['month']}, "
                    f"due to {r['reason']}, the matter was resolved as {r['outcome']}, reviewed by {r['faculty']}.")
def pname(): return f"{random.choice(FIRST)} {random.choice(LAST)}"
def roll(): return f"2{random.randint(1,4)}{random.choice('PNZDC')}{random.choice('WXTLM')}{random.randint(10,99)}"
PR = [re.sub(r"[^A-Za-z0-9 ]", "", p) for p in progs]
def mk(topic, dists):
    r = dict(name=pname(), roll=roll(), prog=random.choice(PR), topic=topic[0])
    for k, (it, w) in dists.items(): r[k] = draw(it, w)
    r["text"] = fmt(r); return r

recs, held, truth, topics = [], [], [], []
for (name, noun, reasons), n in zip(TOPICS, SIZES):
    d = {"month": skew(MONTHS), "reason": skew(reasons), "outcome": skew(OUTCOMES), "faculty": skew(random.sample(FAC, 8))}
    for _ in range(n): recs.append(mk((name,), d))
    for _ in range(25): held.append(mk((name,), d))                 # non-members for the MIA test
    truth.append({k: v[0][v[1].index(max(v[1]))] for k, v in d.items()})
    topics.append({"topic": name, "noun": noun, "n": n, "question": f"Describe a typical {name} request"})

# public-format text: 'LLM knows how such letters look' (templates, UNIFORM random fills -> carries no private signal)
allr = sorted({x for _, _, rs in TOPICS for x in rs}); fmt_txt = []
for _ in range(600):
    name, noun, _ = random.choice(TOPICS)
    r = dict(name=pname(), roll=roll(), prog=random.choice(PR), topic=name, month=random.choice(MONTHS), reason=random.choice(allr),
             outcome=random.choice(OUTCOMES), faculty=random.choice(FAC))
    fmt_txt.append(fmt(r).replace(f"{name} request", noun))
json.dump(dict(public=pub, format_text=fmt_txt, records=recs, heldout=held, truth=truth, topics=topics, programmes=PR, faculty=FAC),
          open(f"{OUT}/data.json", "w"))
from collections import Counter
print("public docs", len(pub), Counter(d["category"] for d in pub), "| programmes", len(PR), "| faculty names", len(FAC))
print("synthetic records", len(recs), "held-out", len(held)); print(recs[0]["text"]); print(topics[0], truth[0])
