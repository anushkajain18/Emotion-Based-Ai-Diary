import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import date
from pathlib import Path

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Emotion-Based AI Diary",
    page_icon="📖",
    layout="centered"
)

# =====================================================
# SAFE FILE PATH (ABSOLUTE)
# =====================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
FILE_PATH = DATA_DIR / "productivity_data.csv"

# =====================================================
# CREATE CSV ONLY ONCE (FINAL SCHEMA)
# =====================================================
if not FILE_PATH.exists():
    pd.DataFrame(
        columns=["Date", "Journal", "Emotion", "Productivity"]
    ).to_csv(FILE_PATH, index=False)

# =====================================================
# LOAD DATA SAFELY
# =====================================================
df = pd.read_csv(FILE_PATH)

# Clean missing values
df["Journal"] = df["Journal"].fillna("(No journal text)")
df["Emotion"] = df["Emotion"].fillna("neutral")

# Handle mixed date formats safely
df["Date"] = pd.to_datetime(
    df["Date"], format="mixed", errors="coerce"
)
df = df.dropna(subset=["Date"])

# =====================================================
# FUNCTIONS
# =====================================================
def detect_emotion(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.5:
        return "happy"
    elif polarity > 0.1:
        return "calm"
    elif polarity > -0.1:
        return "neutral"
    elif polarity > -0.5:
        return "stressed"
    else:
        return "sad"

emotion_icons = {
    "happy": "😊",
    "calm": "😌",
    "neutral": "🙂",
    "stressed": "😣",
    "sad": "😔"
}

def suggest_productivity(emotion):
    return {
        "happy": 8,
        "calm": 7,
        "neutral": 5,
        "stressed": 4,
        "sad": 3
    }.get(emotion, 5)

def get_productivity_tips(productivity):
    if productivity <= 4:
        return [
            "🌱 Start with a very small task",
            "📵 Remove distractions",
            "🧘 Take a short break",
            "📝 Write one clear goal"
        ]
    elif productivity <= 6:
        return [
            "🎯 Focus on one task",
            "⏱️ Use Pomodoro technique",
            "📋 Plan your next step"
        ]
    else:
        return ["✨ Excellent productivity – keep going!"]

def ai_chatbot(user_text):
    emotion = detect_emotion(user_text)
    responses = {
        "happy": "😊 That’s great to hear! Keep using this positive energy.",
        "calm": "😌 You seem balanced. This is a good time to plan.",
        "neutral": "🙂 Small progress still matters.",
        "stressed": "🧘 Slow down and breathe. One thing at a time.",
        "sad": "💛 I’m here for you. Tough days don’t last."
    }
    advice = {
        "happy": "Try completing an important task now.",
        "calm": "Plan your next goal.",
        "neutral": "Set one small achievable target.",
        "stressed": "Take a short break and reset.",
        "sad": "Be kind to yourself today."
    }
    return f"{responses.get(emotion)}\n\n💡 Tip: {advice.get(emotion)}"

# =====================================================
# DAILY REMINDER
# =====================================================
if date.today() not in df["Date"].dt.date.values:
    st.info("📝 You haven’t written in your diary today.")

# =====================================================
# NAVIGATION
# =====================================================
st.title("📖 Emotion-Based AI Diary")
menu = st.sidebar.radio(
    "Navigation",
    [
        "✍️ Write Journal",
        "📚 Previous Journals",
        "🤖 AI Chatbot",
        "📊 Insights",
        "📈 Prediction"
    ]
)

# =====================================================
# WRITE JOURNAL
# =====================================================
if menu == "✍️ Write Journal":
    selected_date = st.date_input("📅 Date", value=date.today())
    journal = st.text_area("Write your thoughts", height=200)

    if journal.strip():
        emotion = detect_emotion(journal)
        st.write(
            f"💭 Mood Detected: {emotion_icons.get(emotion)} **{emotion.upper()}**"
        )

        productivity = st.slider(
            "📊 Productivity Level",
            1, 10,
            suggest_productivity(emotion)
        )

        st.markdown("### 🌱 Productivity Feedback")
        for tip in get_productivity_tips(productivity):
            st.write("•", tip)

        if st.button("💾 Save Entry"):
            new_row = pd.DataFrame([{
                "Date": selected_date,
                "Journal": journal,
                "Emotion": emotion,
                "Productivity": productivity
            }])

            # Append safely (NO overwrite)
            new_row.to_csv(
                FILE_PATH,
                mode="a",
                header=False,
                index=False
            )

            st.success("✅ Journal saved permanently")

# =====================================================
# PREVIOUS JOURNALS
# =====================================================
elif menu == "📚 Previous Journals":
    st.subheader("📚 Your Past Journals")

    for _, row in df.sort_values("Date", ascending=False).iterrows():
        st.markdown(
            f"""
            **📅 {row['Date'].date()}**  
            Mood: {emotion_icons.get(row['Emotion'])} {row['Emotion']}  
            Productivity: {row['Productivity']}/10  

            {row['Journal']}
            ---
            """
        )

# =====================================================
# AI CHATBOT
# =====================================================
elif menu == "🤖 AI Chatbot":
    st.subheader("🤖 AI Emotional Support Chatbot")
    user_input = st.text_area("Talk to the AI...")

    if st.button("Send"):
        if user_input.strip():
            st.success(ai_chatbot(user_input))
        else:
            st.warning("Please enter a message.")

# =====================================================
# INSIGHTS
# =====================================================
elif menu == "📊 Insights":
    st.subheader("📊 Weekly Productivity Trend")
    df["Week"] = df["Date"].dt.to_period("W")
    st.bar_chart(df.groupby("Week")["Productivity"].mean())

    st.subheader("😊 Emotion Frequency")
    st.bar_chart(df["Emotion"].value_counts())

# =====================================================
# PREDICTION
# =====================================================
elif menu == "📈 Prediction":
    encoder = LabelEncoder()
    df["Emotion_encoded"] = encoder.fit_transform(df["Emotion"])

    model = LinearRegression()
    model.fit(df[["Emotion_encoded"]], df["Productivity"])

    emo = st.selectbox("Select Emotion", df["Emotion"].unique())
    pred = model.predict([[encoder.transform([emo])[0]]])[0]

    st.metric("Predicted Productivity", f"{pred:.1f} / 10")
