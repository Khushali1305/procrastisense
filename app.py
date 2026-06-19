import streamlit as st
import json, time, os, sqlite3
from pathlib import Path
from groq import Groq

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProcrastiSense",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Groq client ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        return None
    return Groq(api_key=api_key)

GROQ_MODEL = "llama-3.3-70b-versatile"

# ── SQLite check-in store ─────────────────────────────────────────────────────
DB_PATH = Path("procrastisense_checkins.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT DEFAULT (datetime('now')),
            mood TEXT, energy INTEGER, sleep_hours REAL,
            task_name TEXT, proc_type TEXT,
            nudge TEXT, micro_action TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_checkin(mood, energy, sleep, task, proc_type, nudge, action):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO checkins (mood,energy,sleep_hours,task_name,proc_type,nudge,micro_action) VALUES (?,?,?,?,?,?,?)",
        (mood, energy, sleep, task, proc_type, nudge, action)
    )
    conn.commit()
    conn.close()

init_db()

# ── Nudge generator ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ProcrastiSense, a warm and supportive academic coach.
Rules:
- 2-3 sentences for the nudge, warm and specific. Never generic.
- Never shame, never lecture. Acknowledge the mood first.
- Micro-action must be completable in under 10 minutes.
- Be specific to the procrastination TYPE given.
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

def generate_nudge_live(proc_type, mood, task_name, shap_explanation=""):
    client = get_groq_client()
    if client is None:
        return {
            "nudge": f"Hey, it looks like {task_name} has been waiting a while. That's okay — let's take one tiny step together.",
            "micro_action": "Open the document and write just one sentence. Nothing more."
        }
    prompt = f"""Student profile:
- Procrastination type: {proc_type}
- {TYPE_GUIDANCE.get(proc_type, "")}
- Current mood: {mood}
- Task they are avoiding: {task_name}
{f"- Behavioral signals: {shap_explanation}" if shap_explanation else ""}
Generate the nudge + micro_action JSON now."""
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=250,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ]
        )
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {
            "nudge": "We see you are carrying a lot right now. One small step is all it takes.",
            "micro_action": "Open the task document and write just one sentence."
        }

# ── Demo profiles (pre-built — no API call needed for demo mode) ──────────────
DEMO_PROFILES = [
    {
        "name": "Priya",
        "emoji": "🔴",
        "proc_type": "deadline_panic",
        "type_label": "Deadline-panic",
        "anomaly_score": 0.82,
        "mood": "stressed",
        "task": "Data Structures assignment due in 2 days",
        "shap_explanation": "submission delay is 3x your personal average AND last-minute submission rate jumped to 70% this week",
        "nudge": "Priya, I can see the deadline is close and that feels overwhelming — that makes total sense. You have done difficult assignments before and you can break this one down too.",
        "micro_action": "Set a 15-minute timer and write just the function signature and one comment for each method. Nothing else.",
        "description": "Submits everything at the last minute. High anxiety near deadlines.",
        "color": "#E24B4A"
    },
    {
        "name": "Dev",
        "emoji": "🔵",
        "proc_type": "distraction_escape",
        "type_label": "Distraction-escape",
        "anomaly_score": 0.75,
        "mood": "distracted",
        "task": "Machine Learning project report",
        "shap_explanation": "avoidance score is 40% above your baseline AND screen time vs study time ratio doubled this week",
        "nudge": "Dev, it looks like your focus has been pulled in a lot of directions lately — that is a really normal response to a daunting task. Let's just create one small protected window.",
        "micro_action": "Put your phone in another room, close every tab except the report document, and write one paragraph in the next 20 minutes.",
        "description": "High screen time and app-switching. Avoids tasks by scrolling.",
        "color": "#378ADD"
    },
    {
        "name": "Parth",
        "emoji": "🟡",
        "proc_type": "perfectionism_paralysis",
        "type_label": "Perfectionism-paralysis",
        "anomaly_score": 0.79,
        "mood": "overwhelmed",
        "task": "Research paper introduction",
        "shap_explanation": "delay deviation is 12 days above your personal baseline AND focus trend is declining despite high engagement intent",
        "nudge": "Parth, the introduction is sitting there waiting and you know exactly what it needs — the gap is just getting started. A rough first draft you can improve is infinitely more useful than a perfect plan that stays in your head.",
        "micro_action": "Write the worst possible opening sentence you can think of — intentionally bad. Then write the next one. You now have a draft.",
        "description": "Overthinks starting. High intent but low follow-through.",
        "color": "#BA7517"
    }
]

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🧠 ProcrastiSense")
st.sidebar.caption("AI Procrastination Detection System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🎭 Demo profiles", "📋 Daily check-in", "🧬 Fingerprint", "⚔️ Generic vs AI"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with OULAD dataset · Isolation Forest · SHAP · Llama 3.3 70B")
st.sidebar.caption("Stage 6 — Hackathon build · Simulated behavioral signals disclosed")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DEMO PROFILES
# ══════════════════════════════════════════════════════════════════════════════
if page == "🎭 Demo profiles":
    st.title("🎭 Meet the Students")
    st.caption("Three real procrastination types detected by ProcrastiSense. Click any profile to see the full analysis.")
    st.markdown("---")

    selected_name = st.radio(
        "Select a student",
        [p["name"] for p in DEMO_PROFILES],
        format_func=lambda n: f"{[p for p in DEMO_PROFILES if p['name']==n][0]['emoji']} {n} — {[p for p in DEMO_PROFILES if p['name']==n][0]['type_label']}",
        horizontal=True
    )
    profile = [p for p in DEMO_PROFILES if p["name"] == selected_name][0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Anomaly score", f"{profile['anomaly_score']:.0%}", delta="Above baseline", delta_color="inverse")
    with col2:
        st.metric("Procrastination type", profile["type_label"])
    with col3:
        st.metric("Mood today", profile["mood"].title())

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🔍 What the ML found")
        st.info(f"**Task being avoided:** {profile['task']}")
        st.write(f"**SHAP explanation:** {profile['shap_explanation']}")
        st.write(f"**Student profile:** {profile['description']}")

    with col_right:
        st.subheader("💬 ProcrastiSense nudge")
        st.success(f"**{profile['nudge']}**")
        st.write(f"📌 **Micro-action:** {profile['micro_action']}")

    st.markdown("---")
    st.subheader("📊 Anomaly score over time (simulated)")
    import random
    import pandas as pd
    random.seed(hash(profile["name"]) % (10**6))
    weeks = list(range(1, 9))
    scores = [round(random.uniform(0.2, 0.5), 2) for _ in range(5)] + \
             [round(random.uniform(0.6, 0.9), 2) for _ in range(3)]
    chart_df = pd.DataFrame({"Week": weeks, "Anomaly Score": scores})
    st.line_chart(chart_df.set_index("Week"))
    st.caption("Weeks 6-8: consecutive high anomaly scores triggered the procrastination alert.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: DAILY CHECK-IN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Daily check-in":
    st.title("📋 Daily Check-in")
    st.caption("60 seconds. Tell us how you are feeling and what you are avoiding — we will handle the rest.")
    st.markdown("---")

    with st.form("checkin_form"):
        col1, col2 = st.columns(2)
        with col1:
            mood = st.selectbox("How are you feeling right now?",
                ["stressed", "overwhelmed", "tired", "distracted", "anxious", "okay", "motivated"])
            energy = st.slider("Energy level (1=drained, 10=energised)", 1, 10, 5)
            sleep = st.slider("Hours of sleep last night", 3.0, 10.0, 7.0, 0.5)
        with col2:
            task_name = st.text_input("What task are you avoiding?",
                placeholder="e.g. Data Structures assignment, ML project report...")
            proc_type = st.selectbox("Which sounds most like you right now?",
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
            with st.spinner("Generating your personalised nudge..."):
                result = generate_nudge_live(proc_type, mood, task_name)

            nudge = result.get("nudge", "")
            action = result.get("micro_action", "")

            st.markdown("---")
            st.subheader("💬 Your nudge")
            st.success(f"**{nudge}**")
            st.info(f"📌 **Micro-action:** {action}")

            save_checkin(mood, energy, sleep, task_name, proc_type, nudge, action)
            st.caption("✓ Check-in saved to your history.")

    st.markdown("---")
    st.subheader("📅 Check-in history")
    try:
        import pandas as pd
        conn = sqlite3.connect(DB_PATH)
        hist = pd.read_sql("SELECT ts, mood, task_name, proc_type, nudge FROM checkins ORDER BY ts DESC LIMIT 10", conn)
        conn.close()
        if len(hist):
            st.dataframe(hist, use_container_width=True)
        else:
            st.caption("No check-ins yet. Complete the form above to get started.")
    except Exception:
        st.caption("No history yet.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FINGERPRINT HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧬 Fingerprint":
    st.title("🧬 Procrastination Fingerprint")
    st.caption("Each student has a unique delay signature — a pattern of WHEN they procrastinate and on WHAT types of tasks.")
    st.markdown("---")

    import pandas as pd
    import numpy as np

    heatmap_path = Path("plots/stage3_procrastination_heatmap.png")
    parquet_path = Path("stage3_anomaly_scores.parquet")

    if heatmap_path.exists():
        st.image(str(heatmap_path), caption="Real procrastination fingerprint from OULAD data (Stage 3)")
    elif parquet_path.exists():
        df_stage3 = pd.read_parquet(parquet_path)
        if "day_of_week" in df_stage3.columns and "assessment_type" in df_stage3.columns:
            hmap = (
                df_stage3[df_stage3["if_anomaly"] == 1]
                .groupby(["day_of_week", "assessment_type"])["if_score"]
                .mean()
                .unstack(fill_value=0)
            )
            st.dataframe(hmap.round(3), use_container_width=True)
            st.caption("Mean anomaly score by day of week x assessment type. Higher = more procrastination detected.")
    else:
        st.info("Showing simulated fingerprint (Stage 3 output not found in working directory).")
        np.random.seed(42)
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        types = ["TMA", "CMA"]
        data = pd.DataFrame(
            np.random.beta(2, 5, (7, 2)) + np.array([[0.1], [0.05], [0.15], [0.2], [0.3], [0.4], [0.35]]),
            index=days, columns=types
        ).clip(0, 1).round(3)
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(data, annot=True, fmt=".2f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Mean anomaly score"})
        ax.set_title("Procrastination Fingerprint (simulated)", fontweight="bold")
        ax.set_xlabel("Assessment type")
        ax.set_ylabel("Day of week")
        st.pyplot(fig)
        st.caption("Simulated data shown. Run Stage 3 and copy stage3_anomaly_scores.parquet to the app directory for real data.")

    st.markdown("---")
    st.subheader("What the fingerprint tells us")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Highest risk day", "Friday", delta="30% above avg", delta_color="inverse")
    with col2:
        st.metric("Highest risk type", "TMA (tutor-marked)", delta="High-stakes = more avoidance", delta_color="inverse")
    with col3:
        st.metric("Lowest risk day", "Tuesday", delta="Most consistent engagement")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: GENERIC vs AI SIDE-BY-SIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚔️ Generic vs AI":
    st.title("⚔️ Generic reminder vs ProcrastiSense")
    st.caption("The difference should be obvious in 10 seconds.")
    st.markdown("---")

    for profile in DEMO_PROFILES:
        st.markdown(f"### {profile['emoji']} {profile['name']} — *{profile['type_label']}*")
        col_gen, col_ai = st.columns(2)

        with col_gen:
            st.markdown("**📱 Generic app reminder**")
            st.warning(
                f"⏰ Reminder: **{profile['task'][:40]}...**  \n"
                f"Due soon. Don't forget to submit!"
            )
            st.caption("No context. No personalisation. Easy to dismiss.")

        with col_ai:
            st.markdown("**🤖 ProcrastiSense nudge**")
            st.success(
                f"{profile['nudge']}  \n\n"
                f"📌 *{profile['micro_action']}*"
            )
            st.caption(f"Driven by SHAP: *{profile['shap_explanation'][:80]}...*")

        st.markdown("---")

    st.subheader("Why this is different from a to-do app")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🧠 Knows your pattern**")
        st.write("Detects deviation from YOUR personal baseline — not a global threshold.")
    with col2:
        st.markdown("**🔍 Explains why**")
        st.write("SHAP tells you which specific signals triggered the alert. Not just a score.")
    with col3:
        st.markdown("**💬 Speaks to the cause**")
        st.write("Deadline-panic, distraction, and perfectionism need different interventions.")
