# activation_dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Activation Functions Dashboard",
    
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}

.block-container {
    padding-top: 1rem;
}

h1, h2, h3 {
    color: #38bdf8;
}

.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    padding: 10px;
    border-radius: 10px;
}

.card {
    background: linear-gradient(135deg,#1e293b,#0f172a);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid #334155;
    margin-bottom: 20px;
    box-shadow: 0px 0px 15px rgba(56,189,248,0.2);
}

.highlight {
    color: #22c55e;
    font-weight: bold;
}

.big-font {
    font-size:20px !important;
}

.small-text {
    font-size:16px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.title(" Ultimate Activation Functions Dashboard")
st.markdown("""
<div class='big-font'>
Learn Activation Functions in a <span class='highlight'>Gen-Z Style 🚀</span> with:
<ul>
<li>Interactive Graphs 📊</li>
<li>English + Hinglish Explanation 🇮🇳</li>
<li>Real Life Examples 🌍</li>
<li>Neural Network Intuition 🤖</li>
</ul>
</div>
""", unsafe_allow_html=True)

# =========================
# RANDOM DATASET
# =========================
np.random.seed(42)

study_hours = np.random.randint(1, 12, 50)
marks = study_hours * 8 + np.random.randint(-10, 10, 50)

df = pd.DataFrame({
    "Study Hours": study_hours,
    "Marks": marks
})

st.subheader("📚 Random Dataset Example")
st.write("Imagine AI predicting marks based on study hours.")

fig_data = px.scatter(
    df,
    x="Study Hours",
    y="Marks",
    title="Study Hours vs Marks",
    template="plotly_dark",
    size_max=15
)

st.plotly_chart(fig_data, use_container_width=True)

# =========================
# ACTIVATION FUNCTIONS
# =========================

x = np.linspace(-10, 10, 1000)

# Functions
relu = np.maximum(0, x)

sigmoid = 1 / (1 + np.exp(-x))

tanh = np.tanh(x)

leaky_relu = np.where(x > 0, x, 0.01 * x)

softmax_input = np.array([2.0, 1.0, 0.1])
softmax = np.exp(softmax_input) / np.sum(np.exp(softmax_input))

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚡ Select Activation Function")

selected = st.sidebar.radio(
    "Choose One 👇",
    ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "Softmax"]
)

# =========================
# GRAPH FUNCTION
# =========================
def plot_function(y, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(width=4),
        name=title
    ))

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Input (x)",
        yaxis_title="Output",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# ReLU
# =========================
if selected == "ReLU":

    st.header("⚡ ReLU (Rectified Linear Unit)")

    plot_function(relu, "ReLU Activation")

    st.markdown("""
    <div class='card'>

    ## 📘 English Explanation

    ReLU outputs:
    - 0 for negative values
    - same value for positive values

    Formula:
    f(x) = max(0, x)

    Why it is popular?
    - Very fast ⚡
    - Solves vanishing gradient problem
    - Used in most Deep Learning models

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🇮🇳 Hinglish Explanation

    ReLU ek gatekeeper ki tarah kaam karta hai 🚪

    Agar input negative hai → "Bhai tu andar nahi aa sakta" ❌

    Agar input positive hai → "Aaja bhai full power ke saath" ✅

    Matlab:
    - Negative values ko 0 bana deta hai
    - Positive values ko same rehne deta hai

    Isliye Neural Network fast aur efficient ban jaata hai.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🌍 Real Life Example

    Imagine Instagram Reels Algorithm 📱

    - Bad content → Ignore (0 output)
    - Good content → Push harder 🚀

    Exactly same ReLU karta hai.

    </div>
    """, unsafe_allow_html=True)

# =========================
# SIGMOID
# =========================
elif selected == "Sigmoid":

    st.header("🧠 Sigmoid Function")

    plot_function(sigmoid, "Sigmoid Activation")

    st.markdown("""
    <div class='card'>

    ## 📘 English Explanation

    Sigmoid converts values between 0 and 1.

    Formula:
    f(x) = 1 / (1 + e^-x)

    Mostly used in:
    - Binary Classification
    - Probability prediction

    Output behaves like probability.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🇮🇳 Hinglish Explanation

    Sigmoid ek decision maker ki tarah hai 🤔

    Output:
    - 0 ke paas → "No chance"
    - 1 ke paas → "High chance"

    Example:
    AI bol raha:
    - Spam hai ya nahi?
    - Pass hoga ya fail?

    Sigmoid probability nikalta hai.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🌍 Real Life Example

    Gmail Spam Detection 📧

    - 0.95 → Definitely Spam
    - 0.10 → Safe Mail

    Sigmoid AI ko probability samjhata hai.

    </div>
    """, unsafe_allow_html=True)

# =========================
# TANH
# =========================
elif selected == "Tanh":

    st.header("🔥 Tanh Function")

    plot_function(tanh, "Tanh Activation")

    st.markdown("""
    <div class='card'>

    ## 📘 English Explanation

    Tanh outputs values between:
    -1 and +1

    Formula:
    tanh(x)

    Better than sigmoid because:
    - Zero centered
    - Faster convergence

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🇮🇳 Hinglish Explanation

    Tanh emotions ki tarah kaam karta hai 😎

    - Negative → Sad 😢
    - Positive → Happy 😁
    - Around 0 → Neutral 😐

    Neural Network ko balanced understanding deta hai.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🌍 Real Life Example

    Movie Review Analysis 🎬

    - Negative Review → -1
    - Positive Review → +1

    Tanh sentiments ko samajhne me useful hai.

    </div>
    """, unsafe_allow_html=True)

# =========================
# LEAKY RELU
# =========================
elif selected == "Leaky ReLU":

    st.header("💧 Leaky ReLU")

    plot_function(leaky_relu, "Leaky ReLU Activation")

    st.markdown("""
    <div class='card'>

    ## 📘 English Explanation

    Leaky ReLU fixes ReLU problem.

    ReLU issue:
    Negative neurons become dead forever.

    Leaky ReLU allows tiny negative values.

    Formula:
    f(x)=x if x>0
    else 0.01x

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🇮🇳 Hinglish Explanation

    ReLU negative values ko completely ignore karta tha ❌

    Leaky ReLU bolta hai:
    "Thoda toh chance do yaar" 😂

    Isliye negative values ko tiny output deta hai.

    Network smarter ban jaata hai.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🌍 Real Life Example

    College Attendance System 🎓

    ReLU:
    - Below attendance? Direct reject ❌

    Leaky ReLU:
    - Thoda grace marks de do 😅

    </div>
    """, unsafe_allow_html=True)

# =========================
# SOFTMAX
# =========================
elif selected == "Softmax":

    st.header("🎯 Softmax Function")

    categories = ["Cat 🐱", "Dog 🐶", "Tiger 🐯"]

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=softmax
        )
    ])

    fig.update_layout(
        template="plotly_dark",
        title="Softmax Probability Distribution",
        yaxis_title="Probability"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='card'>

    ## 📘 English Explanation

    Softmax converts outputs into probabilities.

    Total probability always equals 1.

    Used in:
    - Multi-class classification
    - Image recognition
    - NLP

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🇮🇳 Hinglish Explanation

    Softmax AI ka final judge hota hai 👑

    Example:
    AI image dekhta hai aur bolta hai:
    - Cat = 80%
    - Dog = 15%
    - Tiger = 5%

    Highest probability wala final answer hota hai.

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>

    ## 🌍 Real Life Example

    Face Unlock System 📱

    AI decide karta hai:
    - Owner?
    - Stranger?
    - Family Member?

    Softmax final probability decision deta hai.

    </div>
    """, unsafe_allow_html=True)

# =========================
# COMPARISON SECTION
# =========================
st.divider()

st.header("📊 Activation Functions Comparison")

comparison_df = pd.DataFrame({
    "Function": ["ReLU", "Sigmoid", "Tanh", "Leaky ReLU", "Softmax"],
    "Range": [
        "0 to ∞",
        "0 to 1",
        "-1 to 1",
        "-∞ to ∞",
        "0 to 1"
    ],
    "Best Use": [
        "Hidden Layers",
        "Binary Classification",
        "Balanced Output",
        "Avoid Dead Neurons",
        "Multi-class Classification"
    ],
    "Speed": [
        "Very Fast",
        "Slow",
        "Medium",
        "Fast",
        "Medium"
    ]
})

st.dataframe(
    comparison_df,
    use_container_width=True
)

# =========================
# FOOTER
# =========================
st.markdown("""
---
# 🚀 Final Gen-Z Summary

| Function | Simple Meaning |
|---|---|
| ReLU | Ignore negatives 😎 |
| Sigmoid | Gives probability 🎯 |
| Tanh | Emotional balance ⚖️ |
| Leaky ReLU | Gives second chance 😂 |
| Softmax | Final decision maker 👑 |

---
""")

st.success("🔥 Dashboard Ready for Gen-Z ML Learning!")