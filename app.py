import streamlit as st
import re
from docx import Document
from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from rapidfuzz import fuzz  # fuzzy matching
import plotly.express as px

# ------------------- CONFIG -------------------
st.set_page_config(page_title="ATS Score Generator", page_icon="📊", layout="wide")
st.title("📊 AI-Powered ATS Score Generator")
st.markdown("Choose a mode below to evaluate candidate fit with a weighted scoring system and fuzzy matching.")

# ------------------- FUNCTIONS -------------------

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def keyword_match(text, keywords, threshold=80):
    """Fuzzy keyword matching (more realistic scores)."""
    matched = []
    missing = []
    text = text.lower()

    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in text:  # exact match
            matched.append(kw)
        else:
            words = re.findall(r'\w+', text)
            if any(fuzz.partial_ratio(kw_lower, w) >= threshold for w in words):
                matched.append(kw)
            else:
                missing.append(kw)
    return matched, missing

def score_resume(resume_text, jd_text):
    categories = {
        "Skills": ["python", "sql", "excel", "machine learning", "communication", "seo", "campaigns", "data analysis"],
        "Tools": ["tableau", "powerbi", "google analytics", "pandas", "numpy", "matplotlib", "hubspot", "crm"],
        "Experience": ["years", "internship", "project", "team lead", "manager", "analyst"],
        "Education": ["bachelor", "master", "mba", "b.tech", "degree"]
    }

    weights = {"Skills": 0.4, "Tools": 0.25, "Experience": 0.2, "Education": 0.15}
    breakdown = {}
    matched_all = {}
    missing_all = {}
    total_score = 0

    for cat, kws in categories.items():
        matched, missing = keyword_match(resume_text + jd_text, kws)
        matched_all[cat] = matched
        missing_all[cat] = missing
        cat_score = (len(matched) / len(kws)) * 100 if kws else 0
        breakdown[cat] = round(cat_score, 2)
        total_score += cat_score * weights[cat]

    total_score = round(total_score, 2)  # keep realistic scores

    # Decision
    if total_score >= 75:
        decision, color = "✅ Strong Fit - Shortlist", "green"
    elif 50 <= total_score < 75:
        decision, color = "⚖️ Medium Fit - Consider/Training", "orange"
    else:
        decision, color = "❌ Weak Fit - Reject", "red"

    return total_score, breakdown, matched_all, missing_all, decision, color

def generate_pdf(name, score, breakdown, matched, missing, decision):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 750, f"ATS Report for {name}")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Final Score: {score}/100")
    c.drawString(100, 710, f"Decision: {decision}")

    y = 680
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, "Category Scores:")
    c.setFont("Helvetica", 11)
    for cat, val in breakdown.items():
        y -= 20
        c.drawString(120, y, f"{cat}: {val}%")

    y -= 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, "Matched Keywords:")
    y -= 20
    c.setFont("Helvetica", 11)
    for cat, words in matched.items():
        if words:
            c.drawString(120, y, f"{cat}: {', '.join(words)}")
            y -= 20

    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, "Missing Keywords (Gap Analysis):")
    y -= 20
    c.setFont("Helvetica", 11)
    for cat, words in missing.items():
        if words:
            c.drawString(120, y, f"{cat}: {', '.join(words)}")
            y -= 20

    c.save()
    buffer.seek(0)
    return buffer

# ------------------- NEW FEATURE: COMPARATIVE DASHBOARD -------------------
def display_comparative_dashboard(results):
    """
    results: list of dicts → each dict has candidate scores + breakdown
    """
    df = pd.DataFrame(results)

    # 📊 Comparative Data Table
    st.subheader("📊 Comparative Resume Evaluation Table")
    st.dataframe(df.style.background_gradient(cmap="RdYlGn", subset=["Overall"]))

    # 🏆 Leaderboard
    leaderboard = df.sort_values(by="Overall", ascending=False).reset_index(drop=True)
    st.subheader("🏆 Leaderboard")
    st.table(leaderboard[["Candidate", "Overall", "Decision"]])

    # 📈 Bar chart
    st.subheader("📈 Category-wise Comparison")
    df_melted = df.melt(id_vars=["Candidate"], value_vars=["Skills", "Tools", "Experience", "Education"],
                        var_name="Category", value_name="Score")
    fig = px.bar(df_melted, x="Candidate", y="Score", color="Category", barmode="group", text="Score")
    st.plotly_chart(fig, use_container_width=True)

    # 🕸 Radar chart
    st.subheader("🕸 Candidate Competency Radar")
    fig_radar = px.line_polar(df_melted, r="Score", theta="Category", color="Candidate", line_close=True)
    st.plotly_chart(fig_radar, use_container_width=True)

    # 💾 Download option
    st.download_button(
        label="📥 Download Comparative Results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="comparative_results.csv",
        mime="text/csv"
    )

# ------------------- MODE SELECTION -------------------
mode = st.radio("Select Mode", ["Single Resume Mode", "Multi-Resume Ranking"])

# ------------------- SINGLE RESUME MODE -------------------
if mode == "Single Resume Mode":
    st.subheader("📄 Single Resume Evaluation")
    jd_text = st.text_area("Paste Job Description (JD)", height=200, key="jd_single")
    uploaded_file = st.file_uploader("Upload one resume (DOCX only)", type=["docx"], accept_multiple_files=False, key="resume_single")

    if st.button("🔍 Evaluate Single Resume") and jd_text and uploaded_file:
        resume_text = extract_text_from_docx(uploaded_file)
        score, breakdown, matched, missing, decision, color = score_resume(resume_text, jd_text)

        # Dashboard
        st.markdown(f"**Final Score: {score}/100**")
        st.markdown(f"<span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)

        for cat, value in breakdown.items():
            st.progress(value / 100)
            st.write(f"**{cat}: {value}%**")

        st.markdown("**✅ Matched Keywords:** " + (", ".join([", ".join(v) for v in matched.values() if v]) if any(matched.values()) else "None"))
        st.markdown("**❌ Missing Keywords (Gap Analysis):** " + (", ".join([", ".join(v) for v in missing.values() if v]) if any(missing.values()) else "None"))

        # Download reports
        pdf_buffer = generate_pdf(uploaded_file.name, score, breakdown, matched, missing, decision)
        st.download_button("📥 Download ATS Report (PDF)", data=pdf_buffer,
                           file_name=f"{uploaded_file.name}_ATS_Report.pdf", mime="application/pdf")

        df = pd.DataFrame({"Category": list(breakdown.keys()), "Score (%)": list(breakdown.values())})
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Breakdown (CSV)", data=csv,
                           file_name=f"{uploaded_file.name}_ATS_Breakdown.csv", mime="text/csv")

# ------------------- MULTI-RESUME MODE -------------------
elif mode == "Multi-Resume Ranking":
    st.subheader("📂 Multi-Resume Ranking")
    jd_text = st.text_area("Paste Job Description (JD)", height=200, key="jd_multi")
    uploaded_files = st.file_uploader("Upload 2–6 resumes (DOCX only)", type=["docx"], accept_multiple_files=True, key="resume_multi")

    if st.button("🚀 Run Multi-Resume Evaluation") and jd_text and uploaded_files:
        results = []
        results_for_dashboard = []

        for file in uploaded_files:
            resume_text = extract_text_from_docx(file)
            score, breakdown, matched, missing, decision, color = score_resume(resume_text, jd_text)
            results.append((file.name, score, breakdown, matched, missing, decision, color))

            results_for_dashboard.append({
                "Candidate": file.name,
                "Skills": breakdown["Skills"],
                "Tools": breakdown["Tools"],
                "Experience": breakdown["Experience"],
                "Education": breakdown["Education"],
                "Overall": score,
                "Decision": decision
            })

        # Leaderboard + Individual reports
        st.subheader("🏆 Candidate Leaderboard")
        leaderboard = pd.DataFrame(
            [(name, score, decision) for name, score, _, _, _, decision, _ in results],
            columns=["Candidate", "Score", "Decision"]
        ).sort_values(by="Score", ascending=False).reset_index(drop=True)
        st.dataframe(leaderboard, use_container_width=True)

        # Comparative Dashboard
        display_comparative_dashboard(results_for_dashboard)

        # Individual reports
        st.subheader("📑 Detailed Candidate Reports")
        for name, score, breakdown, matched, missing, decision, color in results:
            st.markdown(f"### {name}")
            st.markdown(f"**Final Score: {score}/100**")
            st.markdown(f"<span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)

            for cat, value in breakdown.items():
                st.progress(value / 100)
                st.write(f"**{cat}: {value}%**")

            st.markdown("**✅ Matched Keywords:** " + (", ".join([", ".join(v) for v in matched.values() if v]) if any(matched.values()) else "None"))
            st.markdown("**❌ Missing Keywords (Gap Analysis):** " + (", ".join([", ".join(v) for v in missing.values() if v]) if any(missing.values()) else "None"))

            # Downloads
            pdf_buffer = generate_pdf(name, score, breakdown, matched, missing, decision)
            st.download_button("📥 Download ATS Report (PDF)", data=pdf_buffer,
                               file_name=f"{name}_ATS_Report.pdf", mime="application/pdf")

            df = pd.DataFrame({"Category": list(breakdown.keys()), "Score (%)": list(breakdown.values())})
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Breakdown (CSV)", data=csv,
                               file_name=f"{name}_ATS_Breakdown.csv", mime="text/csv")

            st.markdown("---")
