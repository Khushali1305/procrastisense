import streamlit as st
import json, os, sqlite3
from pathlib import Path
from groq import Groq

from model_inference import ModelInference
from cohort_matching import CohortMatcher

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ProcrastiSense", page_icon="🧠", layout="wide",
                    initial_sidebar_state="expanded")

# ── Load v2 models + cohort data (cached so this only runs once per session) ──
@st.cache_resource
def load_inference():
    return ModelInference(models_dir="stage3_models_v2", demo_json_path="demo_profiles_v2.json")

@st.cache_resource
def load_cohort_matcher():
    return CohortMatcher(data_dir=".")

inference = load_inference()
cohort = load_cohort_matcher()

# ── Groq client ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    return Groq(api_key=api_key) if api_key else None

GROQ_MODEL = "llama-3.3-70b-versatile"

# ── SQLite check-in store ─────────────────────────────────────────────────────
DB_PATH = Path("procrastisense_checkins.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now')),
            mood TEXT, energy INTEGER, sleep_hours REAL, tasks_on_time TEXT,
            task_name TEXT, proc_type TEXT,
            nudge TEXT, micro_action TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_checkin(mood, energy, sleep, tasks_on_time, task, proc_type, nudge, action):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO checkins (mood,energy,sleep_hours,tasks_on_time,task_name,proc_type,nudge,micro_action) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (mood, energy, sleep, tasks_on_time, task, proc_type, nudge, action)
    )
    conn.commit()
    conn.close()

init_db()

# ── Nudge generator (live check-in — cohort-grounded, not model-scored) ──────
SYSTEM_PROMPT = """You are ProcrastiSense, a warm and supportive academic coach.
Rules:
- 2-3 sentences for the nudge, warm and specific. Never generic.
- Never shame, never lecture. Acknowledge the mood first.
- Micro-action must be completable in under 10 minutes.
- Be specific to the procrastination TYPE given.
- You may reference the real cohort statistic naturally if it fits, but do not
  invent additional statistics.
- Output ONLY valid JSON: {"nudge": "...", "micro_action": "..."}"""

TYPE_GUIDANCE = {
    "deadline_panic":
        "This student submits at the last minute and feels anxious near deadlines. "
        "Use calm urgency. Break it into tiny steps. Do NOT add pressure.",
    "distraction_escape":
        "This student avoids tasks by scrolling or switching apps. "
        "Redirect gently. Suggest environment design — phone in another room, one tab open.",
    "perfectionism_paralysis":
        "This student overthinks starting. Fears doing it wrong. "
        "Give permission to start badly. Done > perfect. Lower the stakes."
}

def generate_nudge_live(proc_type, mood, task_name, cohort_narrative=""):
    client = get_groq_client()
    if client is None:
        st.error("DEBUG: get_groq_client() returned None — GROQ_API_KEY secret is missing or empty.")
        return {
            "nudge": f"Hey, it looks like {task_name} has been waiting a while. That's okay — let's take one tiny step together.",
            "micro_action": "Open the document and write just one sentence. Nothing more."
        }
    prompt = f"""Student profile:
- Procrastination type (self-reported): {proc_type}
- {TYPE_GUIDANCE.get(proc_type, "")}
- Current mood: {mood}
- Task they are avoiding: {task_name}
- Real cohort context: {cohort_narrative}
Generate the nudge + micro_action JSON now."""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL, max_tokens=250, temperature=0.7,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}]
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        st.error(f"DEBUG: Groq call failed with: {type(e).__name__}: {e}")
        return {
            "nudge": "We see you are carrying a lot right now. One small step is all it takes.",
            "micro_action": "Open the task document and write just one sentence."
        }

# ── Demo profile flavor text (illustrative only — all NUMBERS below are real) ─
DEMO_FLAVOR = {
    "deadline_panic": {
        "name": "Priya", "emoji": "🔴", "type_label": "Deadline-panic",
        "task": "Data Structures assignment due in 2 days", "mood": "stressed",
        "nudge": "Priya, I can see the deadline is close and that feels overwhelming — that makes total sense. You have done difficult assignments before and you can break this one down too.",
        "micro_action": "Set a 15-minute timer and write just the function signature and one comment for each method. Nothing else.",
        "description": "Submits everything at the last minute. High anxiety near deadlines.",
        "color": "#E24B4A",
    },
    "distraction_escape": {
        "name": "Dev", "emoji": "🔵", "type_label": "Distraction-escape",
        "task": "Machine Learning project report", "mood": "distracted",
        "nudge": "Dev, it looks like your focus has been pulled in a lot of directions lately — that is a really normal response to a daunting task. Let's just create one small protected window.",
        "micro_action": "Put your phone in another room, close every tab except the report document, and write one paragraph in the next 20 minutes.",
        "description": "High screen time and app-switching. Avoids tasks by scrolling.",
        "color": "#378ADD",
    },
    "perfectionism_paralysis": {
        "name": "Parth", "emoji": "🟡", "type_label": "Perfectionism-paralysis",
        "task": "Research paper introduction", "mood": "overwhelmed",
        "nudge": "Parth, the introduction is sitting there waiting and you know exactly what it needs — the gap is just getting started. A rough first draft you can improve is infinitely more useful than a perfect plan that stays in your head.",
        "micro_action": "Write the worst possible opening sentence you can think of — intentionally bad. Then write the next one. You now have a draft.",
        "description": "Overthinks starting. High intent but low follow-through.",
        "color": "#BA7517",
    },
}

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🧠 ProcrastiSense")
st.sidebar.caption("AI Procrastination Detection System — v2 (honest rebuild)")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",
    ["🎭 Demo profiles", "📋 Daily check-in", "🧬 Fingerprint", "⚔️ Generic vs AI"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Built with OULAD dataset · Isolation Forest · LogReg · RF · SHAP · Llama 3.3 70B")
st.sidebar.caption("v2: time-based holdout, fake features removed, real cohort grounding")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DEMO PROFILES — REAL model inference on real OULAD students
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎭 Demo profiles":
    st.title("🎭 Meet the Students")
    st.caption("Three real OULAD students from the 2014 holdout set — never seen during training. "
               "Every number below comes from actually running the trained models on their real data.")
    st.markdown("---")

    selected_type = st.radio(
        "Select a student", list(DEMO_FLAVOR.keys()),
        format_func=lambda t: f"{DEMO_FLAVOR[t]['emoji']} {DEMO_FLAVOR[t]['name']} — {DEMO_FLAVOR[t]['type_label']}",
        horizontal=True,
    )
    flavor = DEMO_FLAVOR[selected_type]
    real = inference.get_demo_profile(selected_type)

    st.info(f"Real OULAD student ID `{real['id_student']}` · "
            f"{'true holdout (2014, never seen during fit)' if real['is_true_holdout_2014'] else ''}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Blended anomaly score", f"{real['blended_score']:.0%}", delta="Above baseline", delta_color="inverse")
    with col2:
        st.metric("RF-predicted type", real["rf_predicted_type"].replace("_", " ").title(),
                   delta=f"{real['rf_confidence']:.0%} confidence")
    with col3:
        st.metric("Mood today (demo)", flavor["mood"].title())

    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🔍 What the ML actually found")
        st.info(f"**Task being avoided:** {flavor['task']}")
        st.write(f"**Student profile:** {flavor['description']}")
        st.write("**Top-3 real SHAP contributions (Isolation Forest):**")
        for s in real["shap_explanation"]:
            st.write(f"- *{s['label']}* (value={s['value']:.2f}) — {s['direction']}")
        st.caption(real["disclosure"])
    with col_right:
        st.subheader("💬 ProcrastiSense nudge")
        st.success(f"**{flavor['nudge']}**")
        st.write(f"📌 **Micro-action:** {flavor['micro_action']}")
        st.caption("Nudge text is illustrative copy for this archetype — the detection numbers to the left are the real model output.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DAILY CHECK-IN — self-report type + real cohort grounding
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Daily check-in":
    st.title("📋 Daily Check-in")
    st.caption("60 seconds. Tell us how you are feeling and what you are avoiding.")
    st.markdown("---")
    st.warning(
        "**Honest framing:** the procrastination-detection models on the Demo Profiles page were "
        "trained on OULAD submission history that a brand-new check-in doesn't have. This page asks "
        "you to self-report your pattern, and grounds your nudge with two *real, verified* cohort "
        "statistics — it does not pretend to run the trained model on you.",
        icon="ℹ️"
    )

    with st.form("checkin_form"):
        col1, col2 = st.columns(2)
        with col1:
            mood = st.selectbox("How are you feeling right now?",
                ["stressed", "overwhelmed", "tired", "distracted", "anxious", "okay", "motivated"])
            energy = st.slider("Energy level (1=drained, 10=energised)", 1, 10, 5)
            sleep = st.slider("Hours of sleep last night", 3.0, 10.0, 7.0, 0.5)
            tasks_on_time = st.selectbox(
                "How often do you complete tasks on time?",
                ["Always", "Often", "Sometimes", "Rarely"], index=2
            )
        with col2:
            task_name = st.text_input("What task are you avoiding?",
                placeholder="e.g. Data Structures assignment, ML project report...")
            proc_type = st.selectbox(
                "Which pattern sounds most like you right now? (self-reported)",
                ["deadline_panic", "distraction_escape", "perfectionism_paralysis"],
                format_func=lambda x: {
                    "deadline_panic": "⏰ Deadline-panic — I know the deadline, I just haven't started",
                    "distraction_escape": "📱 Distraction-escape — I keep opening other tabs/apps",
                    "perfectionism_paralysis": "🔄 Perfectionism-paralysis — I don't know how to start perfectly"
                }[x]
            )
        submitted = st.form_submit_button("Generate my nudge →", type="primary", use_container_width=True)

    if submitted:
        if not task_name.strip():
            st.warning("Please enter the task you are avoiding.")
        else:
            with st.spinner("Looking up real cohort statistics..."):
                cohort_result = cohort.build_narrative(sleep_hours=sleep, tasks_on_time_response=tasks_on_time)
            with st.spinner("Generating your personalised nudge..."):
                result = generate_nudge_live(proc_type, mood, task_name, cohort_result["narrative"])

            nudge = result.get("nudge", "")
            action = result.get("micro_action", "")

            st.markdown("---")
            st.subheader("💬 Your nudge")
            st.success(f"**{nudge}**")
            st.info(f"📌 **Micro-action:** {action}")

            with st.expander("📊 Real cohort statistics behind this nudge"):
                st.write(cohort_result["narrative"])
                st.caption(cohort_result["disclosure"])

            save_checkin(mood, energy, sleep, tasks_on_time, task_name, proc_type, nudge, action)
            st.caption("✓ Check-in saved to your history.")

    st.markdown("---")
    st.subheader("📅 Check-in history")
    try:
        import pandas as pd
        conn = sqlite3.connect(DB_PATH)
        hist = pd.read_sql(
            "SELECT ts, mood, tasks_on_time, task_name, proc_type, nudge FROM checkins ORDER BY ts DESC LIMIT 10",
            conn)
        conn.close()
        if len(hist):
            st.dataframe(hist, use_container_width=True)
        else:
            st.caption("No check-ins yet. Complete the form above to get started.")
    except Exception:
        st.caption("No history yet.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FINGERPRINT HEATMAP — real OULAD data, properly masked for low counts
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Fingerprint":
    st.title("🧬 Procrastination Fingerprint")
    st.caption("Real mean anomaly score by day-of-week × assessment type, computed from the v2 model "
               "on the 2014 holdout set. Cells with fewer than 50 data points are blanked rather than "
               "shown as a misleading number.")
    st.markdown("---")

    import pandas as pd
    fp_path = Path("fingerprint_summary_v2.csv")
    if fp_path.exists():
        hmap = pd.read_csv(fp_path, index_col=0)
        st.dataframe(hmap, use_container_width=True)
        st.caption("Rows = day of week, columns = assessment type (CMA = computer-marked, TMA = tutor-marked). "
                   "Blank = fewer than 50 real assessments in that cell — not enough data to report honestly.")
    else:
        st.warning("fingerprint_summary_v2.csv not found in the app directory.")

    st.markdown("---")
    st.subheader("What the fingerprint tells us")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Highest risk day", "Friday", delta="TMA deadlines cluster here", delta_color="inverse")
    with col2:
        st.metric("Highest risk type", "TMA (tutor-marked)", delta="Higher stakes = more avoidance")
    with col3:
        st.metric("Lowest risk day", "Thursday/Saturday", delta="More consistent engagement")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: GENERIC vs AI SIDE-BY-SIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Generic vs AI":
    st.title("⚔️ Generic reminder vs ProcrastiSense")
    st.caption("The difference should be obvious in 10 seconds.")
    st.markdown("---")

    for ptype, flavor in DEMO_FLAVOR.items():
        real = inference.get_demo_profile(ptype)
        st.markdown(f"### {flavor['emoji']} {flavor['name']} — *{flavor['type_label']}*")
        col_gen, col_ai = st.columns(2)
        with col_gen:
            st.markdown("**📱 Generic app reminder**")
            st.warning(f"⏰ Reminder: **{flavor['task'][:40]}...**  \n"
                       f"Due soon. Don't forget to submit!")
            st.caption("No context. No personalisation. Easy to dismiss.")
        with col_ai:
            st.markdown("**🤖 ProcrastiSense nudge**")
            st.success(f"{flavor['nudge']}  \n\n📌 *{flavor['micro_action']}*")
            top_feat = real["shap_explanation"][0]
            st.caption(f"Driven by real SHAP: *{top_feat['label']}* (value={top_feat['value']:.1f})")
        st.markdown("---")

    st.subheader("Why this is different from a to-do app")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🧠 Knows your pattern**")
        st.write("Detects deviation from YOUR personal baseline — not a global threshold.")
    with col2:
        st.markdown("**🔍 Explains why**")
        st.write("Real SHAP tells you which specific signals triggered the alert, on real holdout data.")
    with col3:
        st.markdown("**💬 Honest about limits**")
        st.write("Demo profiles use real model inference. Live check-in uses verified cohort statistics, "
                 "not a fabricated model score — and says so.")
