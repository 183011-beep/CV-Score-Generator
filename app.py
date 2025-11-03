# app.py
import streamlit as st
import re
from docx import Document
from io import BytesIO
import pandas as pd
import concurrent.futures
from rapidfuzz import fuzz
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import PyPDF2
import plotly.express as px

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Scalable ATS Matcher", page_icon="📊", layout="wide")
st.title("📊 Scalable AI-Powered ATS Matcher")
st.markdown(
    """
Upload multiple Job Descriptions (JDs) and multiple resumes. 
The app matches each resume to each JD (100+ resumes supported), shows a score matrix, leaderboards,
gap analysis and lets you download CSV reports.
"""
)

# -------------------- UTILITIES --------------------
STOP_WORDS = {
    "and","or","the","a","an","to","for","of","in","on","with","we","are","is","be","as","by","you","our","your",
    "will","can","should","candidate","experience","years","year","requirement","skills","knowledge","ability"
}

# File text extraction - cached
@st.cache_data(show_spinner=False)
def extract_text_from_docx(file_bytes):
    # file_bytes: UploadedFile (has .getvalue() or .read())
    try:
        # Document accepts a file-like object
        doc = Document(file_bytes)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes):
    try:
        reader = PyPDF2.PdfReader(file_bytes)
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def extract_text_from_file(uploaded_file):
    # Accepts streamlit UploadedFile; returns text string
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".docx"):
            return extract_text_from_docx(uploaded_file)
        elif name.endswith(".pdf"):
            return extract_text_from_pdf(uploaded_file)
        else:
            # txt or fallback
            raw = uploaded_file.read()
            if isinstance(raw, bytes):
                return raw.decode("utf-8", errors="ignore")
            return str(raw)
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

def normalize_text(s):
    s = s.lower()
    s = re.sub(r'[\W_]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def top_tokens(text, n=20):
    text = normalize_text(text)
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 2]
    counts = pd.Series(tokens).value_counts()
    return list(counts.head(n).index)

# fuzzy match helper
def fuzzy_contains(kw, text, threshold=75):
    """Return True if keyword kw fuzzy matches any contiguous phrase in text with threshold."""
    kw = normalize_text(kw)
    text = normalize_text(text)
    if kw in text:
        return True
    # check against words and n-grams up to len(kw words)
    words = text.split()
    kw_words = kw.split()
    max_ngram = min(6, len(words))
    # check partial ratio against each word and small windows
    for w in words:
        if fuzz.partial_ratio(kw, w) >= threshold:
            return True
    # check small windows (2-4)
    for size in range(2, min(6, len(kw_words)+2)):
        for i in range(len(words)-size+1):
            window = " ".join(words[i:i+size])
            if fuzz.partial_ratio(kw, window) >= threshold:
                return True
    return False

# -------------------- SCORING ENGINE --------------------
# global category pools (extendable)
GLOBAL_SKILLS = {"python","sql","excel","machine learning","data analysis","communication","nlp","statistics","r","ml"}
GLOBAL_TOOLS = {"tableau","power bi","powerbi","google analytics","hubspot","pandas","numpy","matplotlib","aws","crm"}
GLOBAL_EDU = {"bachelor","master","mba","b.tech","m.tech","phd","degree","bsc","msc"}

def extract_jd_keywords(jd_text):
    jd_norm = normalize_text(jd_text)
    # pick category-specific keywords that appear in JD
    skills = {k for k in GLOBAL_SKILLS if k in jd_norm}
    tools = {k for k in GLOBAL_TOOLS if k in jd_norm}
    edu = {k for k in GLOBAL_EDU if k in jd_norm}
    # also general top tokens
    general = set(top_tokens(jd_text, n=30))
    return {
        "skills": sorted(list(skills)),
        "tools": sorted(list(tools)),
        "education": sorted(list(edu)),
        "general": sorted(list(general))
    }

def score_resume_against_jd(resume_text, jd_text, jd_keywords, fuzzy_threshold=78):
    """
    returns: dict with per-category percent (0-100) and weighted overall score
    We use:
      Skills: 40%, Tools: 25%, Experience: 20%, Education: 15%
    Experience is heuristically derived from presence of 'year' tokens or 'intern' etc.
    """
    weights = {"skills": 0.40, "tools": 0.25, "experience": 0.20, "education": 0.15}
    categories = {"skills": jd_keywords.get("skills", []),
                  "tools": jd_keywords.get("tools", []),
                  "education": jd_keywords.get("education", [])}
    res_norm = normalize_text(resume_text)
    jd_norm = normalize_text(jd_text)
    breakdown = {}
    matched = {}
    missing = {}
    # Skills/Tools/Edu matching using fuzzy_contains
    for cat, kws in categories.items():
        if not kws:
            # If JD didn't specify explicit items in that category, use general matching later
            breakdown[cat] = None
            matched[cat] = []
            missing[cat] = []
            continue
        m = []
        mm = []
        for kw in kws:
            if fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold):
                m.append(kw)
            else:
                mm.append(kw)
        pct = (len(m) / len(kws)) * 100 if kws else 0
        breakdown[cat] = round(pct, 2)
        matched[cat] = m
        missing[cat] = mm

    # Experience: heuristic
    # If resume contains "X years" or "years of", capture numeric, otherwise match presence of internship/project/manager tokens
    exp_pct = 0
    # look for explicit years mention in resume
    yrs = re.findall(r'(\d{1,2})\s*\+?\s*(?:years|yrs?)', resume_text.lower())
    if yrs:
        # pick the largest number (experience)
        yrs_num = max(int(x) for x in yrs)
        # check JD desired years (if present)
        jd_yrs = re.findall(r'(\d{1,2})\s*(?:-|to|–)\s*(\d{1,2})\s*(?:years|yrs?)', jd_text.lower())
        if jd_yrs:
            lo, hi = int(jd_yrs[0][0]), int(jd_yrs[0][1])
            if lo <= yrs_num <= hi:
                exp_pct = 100
            else:
                # penalty proportional to distance
                dist = min(abs(yrs_num - lo), abs(yrs_num - hi))
                exp_pct = max(30, 100 - 20 * dist)
        else:
            # JD didn't specify range; give good credit if years >=1
            exp_pct = min(100, 50 + yrs_num * 10)
    else:
        # fallback: check for keywords indicating project/intern/lead/managed
        exp_indicators = ["intern", "project", "team lead", "lead", "managed", "manager", "worked"]
        found = sum(1 for tok in exp_indicators if tok in res_norm)
        exp_pct = min(100, found * 40)  # 0,40,80, etc.

    breakdown["experience"] = round(exp_pct, 2)

    # If a category had no explicit keywords from JD, we fall back to general tokens:
    general_jd = jd_keywords.get("general", [])
    if not categories["skills"]:
        # measure presence of important general tokens (top 10)
        gmatch = sum(1 for kw in general_jd[:10] if fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold))
        breakdown["skills"] = round((gmatch / max(1, len(general_jd[:10]))) * 100, 2)
        matched["skills"] = [kw for kw in general_jd[:10] if fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold)]
        missing["skills"] = [kw for kw in general_jd[:10] if not fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold)]
    if not categories["tools"]:
        gtools = [t for t in GLOBAL_TOOLS if t in jd_norm]  # try pick tools too
        if gtools:
            gmatch = sum(1 for kw in gtools if fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold))
            breakdown["tools"] = round((gmatch / len(gtools)) * 100, 2)
            matched["tools"] = [kw for kw in gtools if fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold)]
            missing["tools"] = [kw for kw in gtools if not fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold)]
        else:
            # use general tokens as proxy
            gmatch = sum(1 for kw in general_jd[10:20] if fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold))
            breakdown["tools"] = round((gmatch / max(1, len(general_jd[10:20]))) * 100, 2)
            matched["tools"] = [kw for kw in general_jd[10:20] if fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold)]
            missing["tools"] = [kw for kw in general_jd[10:20] if not fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold)]
    if not categories["education"]:
        # give credit if resume mentions bachelor/master/mba etc
        ematch = sum(1 for kw in GLOBAL_EDU if fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold))
        breakdown["education"] = round((ematch / len(GLOBAL_EDU)) * 100, 2)
        matched["education"] = [kw for kw in GLOBAL_EDU if fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold)]
        missing["education"] = [kw for kw in GLOBAL_EDU if not fuzzy_contains(kw, res_norm, threshold=fuzzy_threshold)]

    # Now compute weighted overall
    overall = 0.0
    for cat, w in weights.items():
        pct = breakdown.get(cat, 0) or 0
        overall += pct * w

    overall = round(overall, 2)

    # Decision thresholds (kept slightly generous)
    if overall >= 80:
        decision = "✅ Strong Fit"
        color = "green"
    elif overall >= 60:
        decision = "⚖️ Medium Fit"
        color = "orange"
    else:
        decision = "❌ Weak Fit"
        color = "red"

    return {
        "overall": overall,
        "breakdown": breakdown,
        "matched": matched,
        "missing": missing,
        "decision": decision,
        "color": color
    }

# -------------------- PDF REPORT GENERATOR --------------------
def generate_pdf_buffer(candidate_name, jd_name, score_dict):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(60, 750, f"ATS Report — Candidate: {candidate_name}")
    c.setFont("Helvetica", 11)
    c.drawString(60, 730, f"Matched Job Description: {jd_name}")
    c.drawString(60, 710, f"Overall Score: {score_dict['overall']}  Decision: {score_dict['decision']}")
    y = 680
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, y, "Category Breakdown:")
    c.setFont("Helvetica", 11)
    for cat, val in score_dict["breakdown"].items():
        y -= 18
        c.drawString(80, y, f"{cat.title()}: {val}%")
    y -= 25
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, y, "Matched Keywords:")
    c.setFont("Helvetica", 11)
    for cat, words in score_dict["matched"].items():
        y -= 16
        if words:
            c.drawString(80, y, f"{cat.title()}: {', '.join(words)}")
    y -= 18
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, y, "Missing Keywords (Gap Analysis):")
    c.setFont("Helvetica", 11)
    for cat, words in score_dict["missing"].items():
        y -= 16
        if words:
            c.drawString(80, y, f"{cat.title()}: {', '.join(words)}")
    c.save()
    buffer.seek(0)
    return buffer

# -------------------- UI: Inputs --------------------
st.markdown("## Inputs")
col1, col2 = st.columns(2)

with col1:
    st.write("### Job Descriptions")
    uploaded_jds = st.file_uploader("Upload multiple JDs (DOCX / PDF / TXT) — or paste below (use '---' to separate multiple JDs)", type=["docx","pdf","txt"], accept_multiple_files=True, key="jds")
    jd_text_blob = st.text_area("Or paste multiple JDs separated by '---' (optional)", height=120, key="paste_jds")

with col2:
    st.write("### Resumes")
    uploaded_resumes = st.file_uploader("Upload multiple resumes (DOCX / PDF / TXT). Can upload 100+ files.", type=["docx","pdf","txt"], accept_multiple_files=True, key="resumes")
    st.write("Tip: For faster runs, upload DOCX; PDF parsing can be slower.")

st.write("---")

# -------------------- PREPARE JD LIST --------------------
def read_uploaded_jd(file):
    txt = extract_text_from_file(file)
    return {"name": file.name, "text": txt}

@st.cache_data(show_spinner=False)
def prepare_jd_list(uploaded_jds, pasted_blob):
    jd_list = []
    # from files
    if uploaded_jds:
        for f in uploaded_jds:
            try:
                txt = extract_text_from_file(f)
                name = f.name
            except Exception:
                txt = ""
                name = f.name
            jd_list.append({"name": name, "text": txt})
    # from pasted blob
    if pasted_blob:
        parts = [p.strip() for p in pasted_blob.split('---') if p.strip()]
        for i, p in enumerate(parts, 1):
            jd_list.append({"name": f"Pasted_JD_{i}", "text": p})
    return jd_list

jd_list = prepare_jd_list(uploaded_jds, jd_text_blob)

if not jd_list:
    st.info("Upload or paste at least one Job Description (JD) to proceed.")
    st.stop()

if not uploaded_resumes:
    st.info("Upload one or more resumes to evaluate.")
    st.stop()

# -------------------- PARSE RESUMES (concurrent) --------------------
st.write("## Running matching — this may take a short while for many files. Progress will be shown.")
max_workers = min(16, max(4, (len(uploaded_resumes)//5)+1))

@st.cache_data(show_spinner=False)
def parse_all_resumes(uploaded_resumes):
    parsed = []
    # use ThreadPoolExecutor for IO-bound parsing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(extract_text_from_file, f): f.name for f in uploaded_resumes}
        for fut in concurrent.futures.as_completed(futures):
            fname = futures[fut]
            try:
                txt = fut.result()
            except Exception:
                txt = ""
            parsed.append({"name": fname, "text": txt})
    return parsed

with st.spinner("Parsing resumes..."):
    parsed_resumes = parse_all_resumes(uploaded_resumes)

st.success(f"Parsed {len(parsed_resumes)} resumes, {len(jd_list)} JD(s).")

# -------------------- BUILD JD KEYWORDS --------------------
jd_keywords_map = {}
for jd in jd_list:
    jd_keywords_map[jd["name"]] = extract_jd_keywords(jd["text"])

# -------------------- SCORING MATRIX (resume x jd) --------------------
# We'll compute scores in parallel per resume to speed up.
def score_for_resume_against_all_jds(resume):
    resname = resume["name"]
    restext = resume["text"]
    out = {"Candidate": resname}
    per_jd_details = {}
    for jd in jd_list:
        jdname = jd["name"]
        jdtext = jd["text"]
        jdkeys = jd_keywords_map[jdname]
        score_dict = score_resume_against_jd(restext, jdtext, jdkeys)
        out[f"{jdname} Overall"] = score_dict["overall"]
        # store subfields if needed later
        per_jd_details[jdname] = score_dict
    return out, per_jd_details

# parallel scoring
results_short = []
details_map = {}  # candidate -> {jdname:score_dict}
progress_bar = st.progress(0)
total = len(parsed_resumes)
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {ex.submit(score_for_resume_against_all_jds, r): r["name"] for r in parsed_resumes}
    completed = 0
    for fut in concurrent.futures.as_completed(futures):
        try:
            out, per_details = fut.result()
        except Exception:
            out = {"Candidate": futures[fut]}
            per_details = {}
        results_short.append(out)
        details_map[out["Candidate"]] = per_details
        completed += 1
        progress_bar.progress(min(1.0, completed / total))
progress_bar.empty()

# build DataFrame matrix
df = pd.DataFrame(results_short)
# ensure JD columns ordering
jd_columns = [f"{jd['name']} Overall" for jd in jd_list]
cols = ["Candidate"] + jd_columns
df = df.reindex(columns=cols)

# compute best match per resume
def best_match_row(row):
    scores = row[jd_columns].to_dict()
    best_jd = max(scores, key=scores.get)
    return best_jd.replace(" Overall", ""), scores[best_jd]

df["Best Match (JD)"] = df.apply(best_match_row, axis=1).apply(lambda x: x[0])
df["Best Match Score"] = df.apply(best_match_row, axis=1).apply(lambda x: x[1])

# Decision mapping for best match
def map_decision(score):
    if score >= 80: return "✅ Strong Fit"
    if score >= 60: return "⚖️ Medium Fit"
    return "❌ Weak Fit"

df["Decision (Best)"] = df["Best Match Score"].apply(map_decision)

# -------------------- DISPLAY MATRIX --------------------
st.markdown("## Matching Matrix")
st.write("Rows = Candidates, Columns = JDs (Overall score). Best match per candidate shown in 'Best Match (JD)'.")
# style the overall columns with background gradient
styled = df.style.background_gradient(subset=jd_columns, cmap="RdYlGn", vmin=0, vmax=100)
st.dataframe(styled, use_container_width=True)

# Download full matrix CSV
csv_all = df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Full Match Matrix (CSV)", data=csv_all, file_name="match_matrix.csv", mime="text/csv")

# -------------------- PER-JD LEADERBOARDS --------------------
st.markdown("## Per-JD Leaderboards")
for jd in jd_list:
    jdname = jd["name"]
    colname = f"{jdname} Overall"
    st.markdown(f"### 📋 Top matches for **{jdname}**")
    leaderboard = df[["Candidate", colname]].sort_values(by=colname, ascending=False).head(20)
    # add decision color column
    leaderboard["Decision"] = leaderboard[colname].apply(map_decision)
    st.table(leaderboard.reset_index(drop=True))
    # Download per JD
    st.download_button(f"📥 Download Top Matches for {jdname} (CSV)", data=leaderboard.to_csv(index=False).encode("utf-8"),
                       file_name=f"top_matches_{jdname}.csv", mime="text/csv")

# -------------------- SELECTED JD / CANDIDATE VISUALS --------------------
st.markdown("## Visual Insights")
sel_mode = st.selectbox("View by", ["Best Matches", "Select JD", "Select Candidate"])
if sel_mode == "Select JD":
    jd_choice = st.selectbox("Choose JD", [jd["name"] for jd in jd_list])
    colname = f"{jd_choice} Overall"
    vis_df = df[["Candidate", colname]].sort_values(by=colname, ascending=False).head(50)
    st.bar_chart(vis_df.set_index("Candidate"))
    # radar for top 3 candidates
    top3 = vis_df.head(3)["Candidate"].tolist()
    if top3:
        st.markdown("### Top 3 Detailed Breakdown (Radar)")
        radar_rows = []
        for cand in top3:
            details = details_map[cand][jd_choice]
            row = {"Candidate": cand}
            row.update(details["breakdown"])
            radar_rows.append(row)
        radar_df = pd.DataFrame(radar_rows).set_index("Candidate")
        # normalize columns for radar chart by melting
        radar_melt = radar_df.reset_index().melt(id_vars="Candidate", var_name="Category", value_name="Score")
        fig = px.line_polar(radar_melt, r="Score", theta="Category", color="Candidate", line_close=True)
        st.plotly_chart(fig, use_container_width=True)

elif sel_mode == "Select Candidate":
    cand_choice = st.selectbox("Choose Candidate", df["Candidate"].tolist())
    best_jd = df.loc[df["Candidate"] == cand_choice, "Best Match (JD)"].values[0]
    st.markdown(f"### Candidate: **{cand_choice}** — Best matched JD: **{best_jd}**")
    # show breakdown for best_jd
    details = details_map[cand_choice][best_jd]
    st.write("Overall Score:", details["overall"], details["decision"])
    st.subheader("Category Breakdown")
    for cat, val in details["breakdown"].items():
        st.write(f"**{cat.title()}**: {val}%")
        st.progress(val / 100)
    st.subheader("Gap Analysis")
    for cat, miss in details["missing"].items():
        st.write(f"**{cat.title()} Missing:** {', '.join(miss) if miss else 'None'}")
    # buttons: download this candidate's PDF for best_jd
    pdf_buf = generate_pdf_buffer(cand_choice, best_jd, details)
    st.download_button("📥 Download Candidate Report (PDF)", data=pdf_buf, file_name=f"{cand_choice}_{best_jd}_report.pdf", mime="application/pdf")

else:
    st.write("Showing best matches per candidate.")
    fig = px.bar(df.sort_values("Best Match Score", ascending=False).head(50), x="Candidate", y="Best Match Score", color="Decision (Best)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("Notes: Matching uses fuzzy similarity (rapidfuzz). If JD doesn't list specific skills/tools, the system falls back to JD top tokens to estimate relevance. Use the CSV downloads for offline filtering/HR workflows.")
