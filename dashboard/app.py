import streamlit as st
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.classification.data_loader import load_hate_speech_data, split_data
from src.classification.classifier import OffensiveTextClassifier
from src.explainability.explain import explain_prediction

st.set_page_config(page_title="Content Moderation Dashboard", layout="wide")

st.title("🇱🇰 Code-Mixed Content Moderation Dashboard")
st.caption("Sinhala-Tamil-English content moderation pipeline")


@st.cache_resource
def load_trained_classifier():
    comments, labels = load_hate_speech_data("data/raw/sentence-level-annotation.csv")
    train_comments, test_comments, train_labels, test_labels = split_data(comments, labels)
    classifier = OffensiveTextClassifier()
    classifier.train(train_comments, train_labels)
    return classifier, labels


classifier, all_labels = load_trained_classifier()

# ============================================================
# SECTION 1: Live moderation (interactive, changes per input)
# ============================================================
st.header("🔍 Try it live")
st.caption("Type any comment below and click Moderate to see a real-time prediction.")

user_input = st.text_area("Enter a comment to moderate:", "you are so stupid")

if st.button("Moderate", type="primary"):
    label, probs = classifier.predict(user_input)
    top_words = explain_prediction(classifier, user_input)

    result_color = "🔴" if label == "offensive" else "🟢"
    st.subheader(f"{result_color} Prediction: {label.upper()}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Confidence:**")
        st.progress(probs["offensive"], text=f"Offensive: {probs['offensive']:.1%}")
        st.progress(probs["clean"], text=f"Clean: {probs['clean']:.1%}")

    with col2:
        st.write("**Top contributing words:**")
        if top_words:
            for word, score in top_words:
                color = "🔴" if score > 0 else "🟢"
                st.write(f"{color} {word}: {score:.3f}")
        else:
            st.info("None of these words were seen during training — the model has no signal to explain, so it defaults to a 50/50 guess.")

st.divider()

# ============================================================
# SECTION 2: Dataset background info (static, NOT related to
# whatever comment you just tested above)
# ============================================================
st.header("📊 About the Training Data")
st.caption(
    "This section is background context about the dataset used to train the model — "
    "it does **not** change based on what you tested above."
)

clean_count = all_labels.count("clean")
offensive_count = all_labels.count("offensive")

fig = go.Figure(data=[go.Bar(
    x=["Clean", "Offensive"],
    y=[clean_count, offensive_count],
    marker_color=["#2ecc71", "#e74c3c"]
)])
fig.update_layout(title="Label Distribution in Training Data (Class Imbalance)", yaxis_title="Number of comments")
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Total: {len(all_labels)} comments | "
    f"Clean: {clean_count} ({clean_count/len(all_labels):.1%}) | "
    f"Offensive: {offensive_count} ({offensive_count/len(all_labels):.1%})"
)

with st.expander("ℹ️ Why does this matter?"):
    st.write(
        "The training data is heavily imbalanced (~91% clean, ~9% offensive). "
        "This means the model sometimes leans toward 'clean' on borderline comments, "
        "especially longer sentences where a few strongly-offensive words get diluted "
        "by many neutral filler words. This is a known, documented limitation — "
        "see the README for details."
    )