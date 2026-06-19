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

# ── Groq client (3-key rotation) ───────────────────────────────────────────────
def _collect_groq_keys():
    """Gather any configured Groq keys, trying several common naming schemes
    so this works whether secrets.toml has GROQ_API_KEY_1/_2/_3, a single
    GROQ_API_KEY, or a list under GROQ_API_KEYS."""
    keys = []
    # Scheme 1: GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3
    for i in (1, 2, 3):
        k = st.secrets.get(f"GROQ_API_KEY_{i}", os.environ.get(f"GROQ_API_KEY_{i}", ""))
        if k:
            keys.append(k)
    # Scheme 2: a single GROQ_API_KEY
    single = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if single and single not in keys:
        keys.append(single)
    # Scheme 3: a list/array secret GROQ_API_KEYS = ["...", "...", "..."]
    arr = st.secrets.get("GROQ_API_KEYS", None)
    if arr:
        for k in arr:
            if k and k not in keys:
                keys.append(k)
    return keys

@st.cache_resource
def get_groq_clients():
    keys = _collect_groq_keys()
    if not keys:
        return []
    return [Groq(api_key=k) for k in keys]

def get_groq_client():
    """Returns the first available client (kept for backward compatibility
    with any code below that still calls this directly)."""
    clients = get_groq_clients()
    return clients[0] if clients else None

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
            task_name TEXT, blockers TEXT, recent_activity TEXT,
            deadline_proximity TEXT, clarity_score INTEGER,
            proc_type TEXT, type_confidence REAL,
            nudge TEXT, micro_action TEXT
        )
    """)
    # Backward-compatible upgrade for existing DBs from the old schema
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(checkins)")}
    for col, coltype in [("blockers", "TEXT"), ("recent_activity", "TEXT"),
                          ("deadline_proximity", "TEXT"), ("clarity_score", "INTEGER"),
                          ("type_confidence", "REAL")]:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE checkins ADD COLUMN {col} {coltype}")
    conn.commit()
    conn.close()

def save_checkin(mood, energy, sleep, tasks_on_time, task, blockers, recent_activity,
                  deadline_proximity, clarity_score, proc_type, type_confidence, nudge, action):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO checkins (mood,energy,sleep_hours,tasks_on_time,task_name,blockers,"
        "recent_activity,deadline_proximity,clarity_score,proc_type,type_confidence,nudge,micro_action) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (mood, energy, sleep, tasks_on_time, task, blockers, recent_activity,
         deadline_proximity, clarity_score, proc_type, type_confidence, nudge, action)
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

# ── Diagnostic check-in: real blockers instead of a self-labeled dropdown ─────
BLOCKER_OPTIONS = {
    "no_clue_where_to_start": "I genuinely don't know where to start",
    "keep_getting_distracted": "I keep opening other tabs / my phone / social media",
    "worried_not_good_enough": "I'm worried whatever I make won't be good enough",
    "deadline_not_real_yet": "The deadline doesn't feel real / urgent yet",
    "overwhelmed_by_size": "The task feels too big to even approach",
    "low_energy": "I just don't have the energy right now",
}
RECENT_ACTIVITY_OPTIONS = [
    "Scrolling social media / browsing",
    "Doing other (easier) tasks instead",
    "Sitting with the task open, not writing anything",
    "Researching / re-reading instructions repeatedly",
    "Sleeping or resting",
    "Haven't opened it at all today",
]
DEADLINE_OPTIONS = ["Today", "In 1-2 days", "This week", "More than a week away", "No fixed deadline"]

def infer_type_from_diagnostic(blockers, recent_activity, deadline_proximity, clarity_score, energy):
    """
    Transparent, rule-based scoring over the diagnostic answers above —
    NOT the OULAD-trained Random Forest (that model needs click-stream/
    submission-timing features a check-in form can't produce; see
    docs/MODEL_CARD.md). This is disclosed in the UI as a behavioral
    diagnostic, not presented as the same validated model used on the
    Demo Profiles page.

    Returns (proc_type, confidence 0-1, score_breakdown dict).
    """
    scores = {"deadline_panic": 0.0, "distraction_escape": 0.0, "perfectionism_paralysis": 0.0}

    if "deadline_not_real_yet" in blockers:
        scores["deadline_panic"] += 1.0
    if deadline_proximity in ("Today", "In 1-2 days"):
        scores["deadline_panic"] += 1.0
    if "overwhelmed_by_size" in blockers:
        scores["deadline_panic"] += 0.5

    if "keep_getting_distracted" in blockers:
        scores["distraction_escape"] += 1.5
    if recent_activity == "Scrolling social media / browsing":
        scores["distraction_escape"] += 1.0
    if recent_activity == "Doing other (easier) tasks instead":
        scores["distraction_escape"] += 0.5

    if "worried_not_good_enough" in blockers:
        scores["perfectionism_paralysis"] += 1.5
    if "no_clue_where_to_start" in blockers:
        scores["perfectionism_paralysis"] += 0.5
    if recent_activity == "Researching / re-reading instructions repeatedly":
        scores["perfectionism_paralysis"] += 1.0
    if clarity_score <= 2:
        scores["perfectionism_paralysis"] += 1.0

    total = sum(scores.values())
    if total == 0:
        # No clear signal from the diagnostic — fall back to the single
        # strongest raw input (deadline proximity) rather than a coin flip,
        # and mark confidence as low so the UI can say so honestly.
        proc_type = "deadline_panic" if deadline_proximity in ("Today", "In 1-2 days") else "perfectionism_paralysis"
        return proc_type, 0.34, scores

    proc_type = max(scores, key=scores.get)
    confidence = scores[proc_type] / total
    return proc_type, round(confidence, 2), scores


def generate_nudge_live(proc_type, mood, task_name, cohort_narrative=""):
    clients = get_groq_clients()
    if not clients:
        st.error(
            "DEBUG: No Groq keys found in st.secrets. Checked for "
            "GROQ_API_KEY_1/_2/_3, GROQ_API_KEY, and GROQ_API_KEYS (list). "
            "Open your Streamlit Cloud app -> Settings -> Secrets and confirm "
            "the key names match one of these exactly (case-sensitive)."
        )
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

    last_error = None
    for i, client in enumerate(clients):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL, max_tokens=250, temperature=0.7,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}]
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last_error = e
            # Try the next key (handles per-key rate limits) instead of
            # failing immediately on the first one.
            continue

    st.error(f"DEBUG: All {len(clients)} Groq key(s) failed. Last error: "
             f"{type(last_error).__name__}: {last_error}")
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
    st.caption("A couple minutes, not 60 seconds — but the extra questions are what let us figure out "
               "your actual blocker instead of guessing.")
    st.markdown("---")
    st.warning(
        "**Honest framing:** the trained models on the Demo Profiles page need OULAD-style "
        "submission/click data that a check-in form can't produce. This page no longer asks you to "
        "self-label your pattern — instead it infers it from your actual answers below using a "
        "transparent rule-based diagnostic (shown to you, not hidden), and grounds your nudge with "
        "real cohort statistics. See docs/MODEL_CARD.md for exactly how this differs from the "
        "validated model.",
        icon="ℹ️"
    )

    with st.form("checkin_form"):
        st.subheader("How are you doing?")
        col1, col2 = st.columns(2)
        with col1:
            mood = st.selectbox("How are you feeling right now?",
                ["stressed", "overwhelmed", "tired", "distracted", "anxious", "okay", "motivated"])
            energy = st.slider("Energy level (1=drained, 10=energised)", 1, 10, 5)
        with col2:
            sleep = st.slider("Hours of sleep last night", 3.0, 10.0, 7.0, 0.5)
            tasks_on_time = st.selectbox("How often do you complete tasks on time?",
                ["Always", "Often", "Sometimes", "Rarely"], index=2)

        st.markdown("---")
        st.subheader("What's actually going on with this task?")
        task_name = st.text_input("What task are you avoiding?",
            placeholder="e.g. Data Structures assignment, ML project report...")

        blockers = st.multiselect(
            "What's actually stopping you right now? (pick all that apply — this matters more "
            "than the mood above for figuring out the right nudge)",
            options=list(BLOCKER_OPTIONS.keys()),
            format_func=lambda k: BLOCKER_OPTIONS[k],
        )
        recent_activity = st.selectbox(
            "What have you actually been doing for the last hour, honestly?",
            options=RECENT_ACTIVITY_OPTIONS,
        )
        col3, col4 = st.columns(2)
        with col3:
            deadline_proximity = st.selectbox("When is it actually due?", options=DEADLINE_OPTIONS, index=2)
        with col4:
            clarity_score = st.slider(
                "How clear are you on what 'done' looks like for this task? (1=no idea, 5=totally clear)",
                1, 5, 3)

        submitted = st.form_submit_button("Generate my nudge →", type="primary", use_container_width=True)

    if submitted:
        if not task_name.strip():
            st.warning("Please enter the task you are avoiding.")
        elif not blockers:
            st.warning("Pick at least one thing that's actually stopping you — that's what drives the nudge.")
        else:
            proc_type, type_confidence, score_breakdown = infer_type_from_diagnostic(
                blockers, recent_activity, deadline_proximity, clarity_score, energy)

            with st.spinner("Looking up real cohort statistics..."):
                cohort_result = cohort.build_narrative(sleep_hours=sleep, tasks_on_time_response=tasks_on_time)

            blocker_text = "; ".join(BLOCKER_OPTIONS[b] for b in blockers)
            rich_context = (
                f"Specific blockers reported: {blocker_text}. "
                f"Recent activity: {recent_activity}. "
                f"Deadline: {deadline_proximity}. "
                f"Clarity on 'done' (1-5): {clarity_score}. "
                f"{cohort_result['narrative']}"
            )
            with st.spinner("Generating your personalised nudge..."):
                result = generate_nudge_live(proc_type, mood, task_name, rich_context)

            nudge = result.get("nudge", "")
            action = result.get("micro_action", "")

            st.markdown("---")
            st.subheader("🔍 What we're seeing")
            conf_label = "high" if type_confidence >= 0.6 else ("moderate" if type_confidence >= 0.4 else "low")
            st.info(
                f"Based on your answers, this looks most like **{proc_type.replace('_',' ').title()}** "
                f"({conf_label} confidence — {type_confidence:.0%}). This is inferred from the blockers "
                f"and behavior you reported, not a self-label."
            )
            with st.expander("See the diagnostic scoring breakdown"):
                st.write(score_breakdown)
                st.caption("Rule-based, transparent scoring over your diagnostic answers — "
                           "not the OULAD-trained model used on the Demo Profiles page.")

            st.subheader("💬 Your nudge")
            st.success(f"**{nudge}**")
            st.info(f"📌 **Micro-action:** {action}")

            with st.expander("📊 Real cohort statistics behind this nudge"):
                st.write(cohort_result["narrative"])
                st.caption(cohort_result.get("disclosure", ""))

            save_checkin(mood, energy, sleep, tasks_on_time, task_name, blocker_text, recent_activity,
                         deadline_proximity, clarity_score, proc_type, type_confidence, nudge, action)
            st.caption("✓ Check-in saved to your history.")

    st.markdown("---")
    st.subheader("📅 Check-in history")
    try:
        import pandas as pd
        conn = sqlite3.connect(DB_PATH)
        hist = pd.read_sql(
            "SELECT ts, mood, deadline_proximity, task_name, blockers, proc_type, type_confidence, nudge "
            "FROM checkins ORDER BY ts DESC LIMIT 10",
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