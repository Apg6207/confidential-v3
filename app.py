"""
DocSentinel — Enterprise Document Confidentiality Classifier
Run: streamlit run app.py
Requires: pip install streamlit anthropic pytesseract Pillow pypdf
Set your API key: export ANTHROPIC_API_KEY=sk-...
"""

import streamlit as st
import anthropic
import json
import base64
import datetime
import io
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocSentinel",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Lora&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #080808;
    color: #d0d0d0;
}
.stApp { background-color: #080808; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0c0c0c;
    border-right: 1px solid #141414;
}
section[data-testid="stSidebar"] * { color: #888 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0e0e0e;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    padding: 14px;
}
[data-testid="metric-container"] label { color: #444 !important; font-size: 10px; letter-spacing: 2px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ccc !important; font-size: 22px; }

/* Expanders */
details { background: #0e0e0e !important; border: 1px solid #1e1e1e !important; border-radius: 10px !important; margin-bottom: 8px !important; }
details[open] { border-color: #2a2a2a !important; }
summary { color: #bbb !important; font-size: 13px !important; padding: 10px 14px !important; }

/* Buttons */
.stButton > button {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    color: #888 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 6px !important;
    font-size: 12px !important;
}
.stButton > button:hover { border-color: #444 !important; color: #ccc !important; }

/* Approve / escalate buttons */
.approve-btn > button { border-color: #1a4a2a !important; color: #2a9a5a !important; }
.escalate-btn > button { border-color: #6a1010 !important; color: #cc4444 !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0e0e0e;
    border: 1px dashed #222;
    border-radius: 10px;
    padding: 10px;
}

/* Progress bar */
.stProgress > div > div { background-color: #2a7a4a !important; }

/* Tabs */
[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid #111 !important; gap: 0 !important; }
[data-baseweb="tab"] { background: transparent !important; color: #333 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 10px !important; letter-spacing: 1.5px !important; border-bottom: 1px solid transparent !important; }
[aria-selected="true"][data-baseweb="tab"] { color: #ccc !important; border-bottom: 1px solid #888 !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #141414; border-radius: 8px; }

/* Divider */
hr { border-color: #141414 !important; }

/* Info / warning / error boxes */
.stAlert { border-radius: 8px !important; font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TAXONOMY = {
    "PUBLIC":       {"icon": "🌐", "label": "Public",       "color": "#2a7a4a", "badge": "🟢 Public"},
    "INTERNAL":     {"icon": "🏢", "label": "Internal",     "color": "#7a7a2a", "badge": "🟡 Internal"},
    "CONFIDENTIAL": {"icon": "🔒", "label": "Confidential", "color": "#c07020", "badge": "🟠 Confidential"},
    "RESTRICTED":   {"icon": "🔴", "label": "Restricted",   "color": "#cc2020", "badge": "🔴 Restricted"},
}

TAXONOMY_TRIGGERS = {
    "PUBLIC":       ["Marketing copy", "Press releases", "Public pricing", "Product docs"],
    "INTERNAL":     ["Policy docs", "Onboarding materials", "Internal memos", "Non-sensitive ops"],
    "CONFIDENTIAL": ["Contracts", "Internal strategy", "Budgets", "Vendor agreements", "NDA"],
    "RESTRICTED":   ["PII / SSN", "Financial statements", "Salaries", "Medical data", "Credentials", "HR records", "Legal obligations", "M&A data"],
}

SYSTEM_PROMPT = """You are an enterprise document classification engine using this 4-level taxonomy:
- PUBLIC: Approved for external audiences. No sensitive content.
- INTERNAL: Employees only. Low sensitivity. No PII or financial secrets.
- CONFIDENTIAL: Sensitive, need-to-know. Contains internal strategy, agreements, or identifiable data.
- RESTRICTED: Highest sensitivity. PII, financial data, legal obligations, credentials, HR records, regulated data.

SENSITIVITY BIAS: Always err toward higher classification. Missing a sensitive document is far worse than over-flagging. When uncertain between two levels, choose the higher one.

Set needsReview=true if: confidence < 0.82 OR classification is RESTRICTED.

If the document appears scanned or handwritten, set ocr=true.

Identify up to 4 sensitive text snippets and format them with [MASKED] replacing the actual sensitive value.

Respond ONLY with valid JSON, no markdown, no backticks:
{"classification":"PUBLIC"|"INTERNAL"|"CONFIDENTIAL"|"RESTRICTED","confidence":0.0-1.0,"reason":"One sentence.","signals":["tag1","tag2"],"needsReview":true|false,"maskedSnippets":["text [MASKED]"],"ocr":true|false}"""

SAMPLE_DOCS = [
    {"id":"s1","name":"Q4-2024-FinancialReport.pdf","source":"SharePoint","classification":"RESTRICTED","confidence":0.97,"reason":"Contains quarterly revenue figures, margin breakdowns, and forward-looking guidance not yet disclosed publicly.","signals":["Revenue data","EPS guidance","M&A pipeline"],"status":"auto-approved","timestamp":"2025-04-21 09:14:32","masked_snippets":["Net revenue of $[MASKED]M","EPS guidance of $[MASKED]"],"ocr":False},
    {"id":"s2","name":"Employee_Handbook_2025.pdf","source":"Google Drive","classification":"INTERNAL","confidence":0.91,"reason":"Standard employee policy document. Contains internal procedures but no PII or sensitive financial data.","signals":["Policy document","Internal procedures"],"status":"auto-approved","timestamp":"2025-04-21 09:15:01","masked_snippets":[],"ocr":False},
    {"id":"s3","name":"ScannedNDA_TechCorp.png","source":"Upload","classification":"RESTRICTED","confidence":0.71,"reason":"Scanned NDA detected via OCR. Legal binding language and non-disclosure clauses found. Confidence below threshold.","signals":["NDA language","Third-party entity","Legal obligation","OCR source"],"status":"pending-review","timestamp":"2025-04-21 09:15:44","masked_snippets":["Party: [MASKED] Inc.","Effective date: [MASKED]"],"ocr":True},
    {"id":"s4","name":"MarketingOnePageF24.docx","source":"Slack","classification":"PUBLIC","confidence":0.95,"reason":"External-facing marketing collateral with no sensitive content. Cleared for public distribution.","signals":["Marketing copy","Product features","Public pricing"],"status":"auto-approved","timestamp":"2025-04-21 09:16:10","masked_snippets":[],"ocr":False},
    {"id":"s5","name":"HRReview_JSmith_2024.xlsx","source":"SharePoint","classification":"RESTRICTED","confidence":0.88,"reason":"Annual performance review containing salary data, PII, and manager commentary. Requires restricted access.","signals":["PII detected","Salary data","HR record"],"status":"pending-review","timestamp":"2025-04-21 09:17:03","masked_snippets":["Salary: $[MASKED]","SSN: [MASKED]","DoB: [MASKED]"],"ocr":False},
]

INIT_AUDIT = [
    {"time":"09:17:03","event":"Flagged for review","doc":"HRReview_JSmith_2024.xlsx","actor":"System","level":"RESTRICTED"},
    {"time":"09:16:10","event":"Auto-approved","doc":"MarketingOnePageF24.docx","actor":"System","level":"PUBLIC"},
    {"time":"09:15:44","event":"OCR completed + flagged","doc":"ScannedNDA_TechCorp.png","actor":"System","level":"RESTRICTED"},
    {"time":"09:15:01","event":"Auto-approved","doc":"Employee_Handbook_2025.pdf","actor":"System","level":"INTERNAL"},
    {"time":"09:14:32","event":"Auto-approved","doc":"Q4-2024-FinancialReport.pdf","actor":"System","level":"RESTRICTED"},
    {"time":"09:14:00","event":"Sync started","doc":"SharePoint: /Finance","actor":"Scheduler","level":None},
]

# ── Session state ──────────────────────────────────────────────────────────────
if "docs" not in st.session_state:
    st.session_state.docs = SAMPLE_DOCS.copy()
if "audit" not in st.session_state:
    st.session_state.audit = INIT_AUDIT.copy()
if "mask_state" not in st.session_state:
    st.session_state.mask_state = {}   # doc_id -> bool (True = masked)

def add_audit(event, doc_name, actor, level):
    st.session_state.audit.insert(0, {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "event": event,
        "doc": doc_name,
        "actor": actor,
        "level": level,
    })

# ── Classify via Claude API ────────────────────────────────────────────────────
def classify_file(file_bytes: bytes, filename: str, mime: str) -> dict:
    client = anthropic.Anthropic()

    is_image = mime.startswith("image/")
    is_pdf   = mime == "application/pdf"
    b64      = base64.standard_b64encode(file_bytes).decode()

    if is_image:
        user_content = [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            {"type": "text",  "text": f"Filename: {filename}. If scanned document, perform OCR first. Classify and return only JSON."},
        ]
    elif is_pdf:
        user_content = [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}},
            {"type": "text", "text": f"Filename: {filename}. Classify and return only JSON."},
        ]
    else:
        text = file_bytes.decode("utf-8", errors="replace")[:6000]
        user_content = f"Filename: {filename}\nContent:\n{text}\nReturn only JSON."

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Helpers ────────────────────────────────────────────────────────────────────
def badge_html(level):
    t = TAXONOMY.get(level, TAXONOMY["INTERNAL"])
    colors = {"PUBLIC":"#2a7a4a","INTERNAL":"#7a7a2a","CONFIDENTIAL":"#c07020","RESTRICTED":"#cc2020"}
    bgs    = {"PUBLIC":"#071a0d","INTERNAL":"#1a1a07","CONFIDENTIAL":"#1a0f07","RESTRICTED":"#1a0707"}
    borders= {"PUBLIC":"#1a4a2a","INTERNAL":"#4a4a1a","CONFIDENTIAL":"#6a3a10","RESTRICTED":"#6a1010"}
    c, bg, b = colors[level], bgs[level], borders[level]
    return f'<span style="background:{bg};border:1px solid {b};color:{c};padding:3px 9px;border-radius:4px;font-size:11px;font-weight:600;font-family:IBM Plex Mono,monospace;">{t["badge"]}</span>'

def status_dot(status):
    c = {"pending-review":"#cc7000","escalated":"#cc2020","auto-approved":"#2a7a4a"}.get(status,"#555")
    return f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{c};box-shadow:0 0 5px {c}66;margin-right:6px;"></span>'

def conf_bar_html(value, warn=False):
    color = "#cc2020" if warn else ("#2a7a4a" if value > 0.85 else "#c07020")
    pct   = round(value * 100)
    return f"""
    <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
      <div style="flex:1;height:3px;background:#1e1e1e;border-radius:2px;overflow:hidden;">
        <div style="width:{pct}%;height:100%;background:{color};"></div>
      </div>
      <span style="font-size:10px;color:#555;min-width:28px;">{pct}%</span>
    </div>"""

def signal_pills(signals, level):
    border_colors = {"PUBLIC":"#1a4a2a","INTERNAL":"#4a4a1a","CONFIDENTIAL":"#6a3a10","RESTRICTED":"#6a1010"}
    text_colors   = {"PUBLIC":"#2a7a4a","INTERNAL":"#7a7a2a","CONFIDENTIAL":"#c07020","RESTRICTED":"#cc2020"}
    bc = border_colors.get(level,"#2a2a2a")
    tc = text_colors.get(level,"#888")
    pills = " ".join(f'<span style="font-size:10px;background:{bc}33;color:{tc};padding:2px 7px;border-radius:3px;margin:2px;">{s}</span>' for s in signals)
    return f'<div style="margin-top:8px;">{pills}</div>'

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔐 DOCSENTINEL")
    st.caption("Enterprise · Document Intelligence")
    st.divider()

    api_key = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
    if api_key:
        import os; os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.markdown("**SENSITIVITY BIAS**")
    st.success("● HIGH — false negatives prioritized")
    st.caption("Conf. threshold: 82%\nAny RESTRICTED → human review")

    st.divider()
    st.markdown("**INTEGRATIONS**")
    integrations = [
        ("📁 SharePoint", True), ("💚 Google Drive", True),
        ("📦 Box", False),       ("✉️ Gmail", False), ("💬 Slack", True),
    ]
    for name, conn in integrations:
        col1, col2 = st.columns([3,1])
        col1.caption(name)
        col2.caption("🟢" if conn else "⚫")

    st.divider()
    st.markdown("**TRAINING DATA**")
    st.caption("Labeled docs: 2,841\nPolicy rules: 47\nLast retrain: Apr 14")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<h1 style="font-family:IBM Plex Mono,monospace;font-size:22px;letter-spacing:2px;color:#e0e0e0;margin-bottom:4px;">DOCSENTINEL</h1>', unsafe_allow_html=True)
st.caption("Enterprise Document Confidentiality Classifier")
st.divider()

# ── Stats bar ─────────────────────────────────────────────────────────────────
docs = st.session_state.docs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Scanned",   len(docs))
c2.metric("🔴 Restricted",   sum(1 for d in docs if d["classification"]=="RESTRICTED"))
c3.metric("🔒 Confidential", sum(1 for d in docs if d["classification"]=="CONFIDENTIAL"))
c4.metric("⚠ Pending Review",sum(1 for d in docs if d["status"]=="pending-review"))

pending_count = sum(1 for d in docs if d["status"]=="pending-review")
if pending_count:
    st.warning(f"⚠ {pending_count} document{'s' if pending_count>1 else ''} awaiting human review — confidence below 82% or classification is RESTRICTED.")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_queue, tab_upload, tab_taxonomy, tab_audit = st.tabs([
    f"📋 DOCUMENT QUEUE ({pending_count} pending)" if pending_count else "📋 DOCUMENT QUEUE",
    "⬆ UPLOAD & SCAN",
    "📚 POLICY TAXONOMY",
    "📜 AUDIT LOG",
])

# ────────────────────────────────────────────────────────────────────────────────
# TAB 1 — DOCUMENT QUEUE
# ────────────────────────────────────────────────────────────────────────────────
with tab_queue:
    sorted_docs = sorted(docs, key=lambda d: 0 if d["status"]=="pending-review" else (1 if d["status"]=="escalated" else 2))

    for doc in sorted_docs:
        review    = doc["status"] == "pending-review"
        escalated = doc["status"] == "escalated"
        t         = TAXONOMY[doc["classification"]]
        warn      = review and doc["confidence"] < 0.82

        # Expander label
        ocr_tag    = " [OCR]" if doc.get("ocr") else ""
        status_tag = " ⚠ REVIEW" if review else (" ↑ ESCALATED" if escalated else "")
        label      = f"{t['icon']} {doc['name']}{ocr_tag}  ·  {doc['classification']}{status_tag}"

        with st.expander(label, expanded=review):
            # Top row: badge + confidence
            h1, h2 = st.columns([3,1])
            with h1:
                st.markdown(
                    f"{status_dot(doc['status'])}"
                    f"<span style='font-size:12px;color:#888;'>via {doc['source']} · {doc['timestamp']}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(conf_bar_html(doc["confidence"], warn=warn), unsafe_allow_html=True)
            with h2:
                st.markdown(badge_html(doc["classification"]), unsafe_allow_html=True)

            st.markdown("---")

            # Body: reason + snippets
            left, right = st.columns(2)
            with left:
                st.markdown('<div style="font-size:9px;color:#444;letter-spacing:2px;">CLASSIFICATION REASON</div>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:12px;color:#777;line-height:1.7;font-family:Lora,serif;">{doc["reason"]}</p>', unsafe_allow_html=True)
                st.markdown(signal_pills(doc["signals"], doc["classification"]), unsafe_allow_html=True)

            with right:
                st.markdown('<div style="font-size:9px;color:#444;letter-spacing:2px;">SENSITIVE SNIPPETS</div>', unsafe_allow_html=True)
                snippets = doc.get("masked_snippets", [])
                if not snippets:
                    st.caption("No sensitive snippets detected.")
                else:
                    doc_id = doc["id"]
                    # Default to masked
                    if doc_id not in st.session_state.mask_state:
                        st.session_state.mask_state[doc_id] = True
                    is_masked = st.session_state.mask_state[doc_id]

                    toggle_label = "👁 Reveal" if is_masked else "🙈 Mask"
                    if st.button(toggle_label, key=f"mask_{doc_id}"):
                        st.session_state.mask_state[doc_id] = not is_masked
                        st.rerun()

                    for s in snippets:
                        display = s if is_masked else s.replace("[MASKED]", "████████")
                        color   = "#cc2020" if is_masked else "#666"
                        st.markdown(
                            f'<div style="font-size:11px;font-family:IBM Plex Mono,monospace;background:#111;padding:6px 10px;border-radius:4px;color:{color};border:1px solid #1a1a1a;margin-bottom:4px;">{display}</div>',
                            unsafe_allow_html=True
                        )

            # Review actions
            if review:
                st.markdown("---")
                st.caption("Confidence below threshold — human sign-off required.")
                ba, bb, _ = st.columns([1, 1, 4])
                with ba:
                    with st.container():
                        if st.button("✓ Approve", key=f"approve_{doc['id']}"):
                            for d in st.session_state.docs:
                                if d["id"] == doc["id"]:
                                    d["status"] = "auto-approved"
                            add_audit("Human approved", doc["name"], "You", doc["classification"])
                            st.rerun()
                with bb:
                    if st.button("↑ Escalate", key=f"escalate_{doc['id']}"):
                        for d in st.session_state.docs:
                            if d["id"] == doc["id"]:
                                d["status"] = "escalated"
                        add_audit("Escalated to Legal", doc["name"], "You", doc["classification"])
                        st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# TAB 2 — UPLOAD & SCAN
# ────────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("### Upload Documents to Classify")
    st.caption("Supports PDF, images (JPG/PNG — OCR applied), TXT, CSV, JSON, DOCX, XLSX, MD")

    uploaded = st.file_uploader(
        "Drop files here or browse",
        accept_multiple_files=True,
        type=["pdf","png","jpg","jpeg","txt","csv","json","md","docx","xlsx","xml"],
    )

    if uploaded:
        if st.button("🔍 Classify All", type="primary"):
            if not st.session_state.get("ANTHROPIC_API_KEY") and not __import__("os").environ.get("ANTHROPIC_API_KEY"):
                st.error("Enter your Anthropic API key in the sidebar first.")
            else:
                progress = st.progress(0, text="Starting classification…")
                for i, f in enumerate(uploaded):
                    progress.progress((i) / len(uploaded), text=f"Classifying {f.name}…")
                    try:
                        result = classify_file(f.read(), f.name, f.type or "text/plain")
                        new_doc = {
                            "id": f"u{i}{datetime.datetime.now().timestamp()}",
                            "name": f.name,
                            "source": "Upload",
                            "classification": result.get("classification","INTERNAL"),
                            "confidence": result.get("confidence", 0.7),
                            "reason": result.get("reason",""),
                            "signals": result.get("signals",[]),
                            "status": "pending-review" if result.get("needsReview") else "auto-approved",
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "masked_snippets": result.get("maskedSnippets",[]),
                            "ocr": result.get("ocr", False),
                        }
                        st.session_state.docs.insert(0, new_doc)
                        event = "Flagged for review" if new_doc["status"]=="pending-review" else "Auto-approved"
                        add_audit(event, f.name, "System", new_doc["classification"])
                        st.success(f"✓ {f.name} → {new_doc['classification']} ({round(new_doc['confidence']*100)}% confidence)")
                    except Exception as e:
                        st.error(f"✗ {f.name}: {e}")
                progress.progress(1.0, text="Done!")
                st.rerun()

# ────────────────────────────────────────────────────────────────────────────────
# TAB 3 — POLICY TAXONOMY
# ────────────────────────────────────────────────────────────────────────────────
with tab_taxonomy:
    st.markdown("### Organizational Policy Taxonomy")
    st.caption("4-level classification framework. Sensitivity bias: HIGH — when uncertain, the higher level is always selected.")

    for level, meta in TAXONOMY.items():
        count = sum(1 for d in docs if d["classification"]==level)
        triggers = TAXONOMY_TRIGGERS[level]
        with st.expander(f"{meta['icon']} {meta['label']} — {count} documents", expanded=level=="RESTRICTED"):
            st.markdown(f"**{meta['badge']}**")
            st.caption({
                "PUBLIC": "Approved for external distribution. No sensitive content.",
                "INTERNAL": "Employees only. Low sensitivity. No PII or financial secrets.",
                "CONFIDENTIAL": "Sensitive, need-to-know basis. Internal strategy or identifiable data.",
                "RESTRICTED": "Highest sensitivity. PII, financial data, legal obligations, HR records.",
            }[level])
            st.markdown("**Triggers:**")
            cols = st.columns(4)
            for j, tag in enumerate(triggers):
                cols[j % 4].markdown(
                    f'<span style="font-size:10px;background:#1a1a1a;color:#666;padding:3px 8px;border-radius:3px;">{tag}</span>',
                    unsafe_allow_html=True
                )

    st.divider()
    st.markdown("### Sensitivity Bias Configuration")
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence Threshold", "82%", help="Docs below this go to human review")
    col2.metric("Tie-breaking Rule", "Higher level", help="When uncertain between two levels, pick the stricter one")
    col3.metric("RESTRICTED auto-approve", "Never", help="All RESTRICTED docs go to human review regardless of confidence")

# ────────────────────────────────────────────────────────────────────────────────
# TAB 4 — AUDIT LOG
# ────────────────────────────────────────────────────────────────────────────────
with tab_audit:
    st.markdown("### Immutable Audit Trail")
    st.caption("Every classification event is logged with timestamp, actor, and level. In production: append-only, cryptographically signed.")

    import pandas as pd
    audit_rows = []
    for row in st.session_state.audit:
        level_badge = TAXONOMY[row["level"]]["badge"] if row["level"] and row["level"] in TAXONOMY else "—"
        audit_rows.append({
            "Time":     row["time"],
            "Event":    row["event"],
            "Document": row["doc"],
            "Actor":    row["actor"],
            "Level":    level_badge,
        })

    df = pd.DataFrame(audit_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False)
    st.download_button("⬇ Export Audit Log (CSV)", csv, "audit_log.csv", "text/csv")
