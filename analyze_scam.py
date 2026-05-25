
# ─── 0. IMPORTS ──────────────────────────────────────────────────────────────
import os, re, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import nltk

# Download resources (one-time)
for pkg in ["vader_lexicon", "punkt", "stopwords", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# ─── 1. CUSTOM SCAM-DOMAIN LEXICON ───────────────────────────────────────────
# Tăng độ chính xác cho context du lịch / fintech
SCAM_BOOST = {
    # Strong negative – scam actions
    "scam": -3.5, "scammed": -3.5, "fraud": -3.5, "fraudulent": -3.5,
    "robbed": -3.2, "stolen": -3.2, "stole": -3.2, "theft": -3.0,
    "ripped": -2.8, "rip-off": -3.0, "ripoff": -3.0, "overcharged": -2.5,
    "overcharge": -2.5, "duped": -3.0, "deceived": -2.8, "tricked": -2.8,
    "swindled": -3.2, "conned": -3.0, "hustled": -2.5, "pickpocket": -3.0,
    "fake": -2.5, "counterfeit": -2.8, "forged": -2.8,
    "threatening": -2.5, "aggressive": -2.5, "harassed": -2.8,
    # Moderate negative – warning signals
    "warning": -1.5, "beware": -1.8, "avoid": -1.5, "careful": -0.8,
    "suspicious": -1.8, "sketchy": -2.0, "shady": -2.0,
    "dangerous": -2.2, "unsafe": -2.5, "terrible": -2.5, "awful": -2.5,
    # Fintech / payment context
    "unauthorized": -2.5, "charged": -1.2, "overpriced": -2.0,
    "hidden fee": -2.2, "refused refund": -3.0,
    # Positive safety signals
    "safe": 1.5, "trustworthy": 2.0, "reliable": 1.8, "legitimate": 1.5,
    "helpful": 1.2, "honest": 1.8, "transparent": 1.5,
    "refunded": 1.5, "resolved": 1.2, "protected": 1.5,
}

# ─── 2. LOAD & CLEAN DATA ────────────────────────────────────────────────────
df = pd.read_csv("/Users/lethihongnhung/Downloads/code_AI/vietnam_scam_reviews.csv")
df.columns = df.columns.str.strip().str.lstrip("\ufeff")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.to_period("M")
df["review_text"] = df["review_text"].fillna("").astype(str)
df["title"] = df["title"].fillna("").astype(str)
df["full_text"] = df["title"] + " " + df["review_text"]

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s\'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["full_text"].apply(clean_text)

# ─── 3. VADER + CUSTOM LEXICON ───────────────────────────────────────────────
sid = SentimentIntensityAnalyzer()
sid.lexicon.update(SCAM_BOOST)          # inject domain lexicon

def get_vader(text):
    scores = sid.polarity_scores(text[:512])   # VADER input cap
    return scores["compound"], scores["neg"], scores["neu"], scores["pos"]

def get_textblob(text):
    tb = TextBlob(text[:1000])
    return tb.sentiment.polarity, tb.sentiment.subjectivity

# ─── 4. ENSEMBLE SCORING ─────────────────────────────────────────────────────
VADER_W  = 0.70   # VADER mạnh hơn cho informal text
TB_W     = 0.30

vader_results = df["clean_text"].apply(get_vader)
df["vader_compound"] = [r[0] for r in vader_results]
df["vader_neg"]      = [r[1] for r in vader_results]
df["vader_neu"]      = [r[2] for r in vader_results]
df["vader_pos"]      = [r[3] for r in vader_results]

tb_results = df["clean_text"].apply(get_textblob)
df["tb_polarity"]     = [r[0] for r in tb_results]
df["tb_subjectivity"] = [r[1] for r in tb_results]

df["ensemble_score"] = (
    VADER_W * df["vader_compound"] +
    TB_W    * df["tb_polarity"]
)

# ─── 5. LABELS ───────────────────────────────────────────────────────────────
def label(score):
    if score >=  0.20: return "Positive"
    if score <= -0.20: return "Negative"
    return "Neutral"

df["sentiment"]        = df["ensemble_score"].apply(label)
df["vader_sentiment"]  = df["vader_compound"].apply(label)

# ─── 6. SCAM TYPE EXTRACTION ─────────────────────────────────────────────────
SCAM_PATTERNS = {
    "Fake Grab / Taxi":    r"fake grab|fake taxi|grab scam|taxi scam|fake driver",
    "Overcharging":        r"overcharg|ripped off|rip.off|overpriced|inflated price",
    "Currency / Change":   r"change|currency|note|bill|exchange|short.chang",
    "Theft / Pickpocket":  r"stole|stolen|theft|pickpocket|robbed|bag snatch",
    "Fake Products":       r"fake|counterfeit|knock.off|replica",
    "Phone / Digital":     r"phone scam|call scam|digital fraud|online scam",
    "Accommodation":       r"hotel scam|hostel|booking scam|accommodation",
    "Tour / Activity":     r"tour scam|activity scam|guide scam|tourist trap",
}

for stype, pattern in SCAM_PATTERNS.items():
    df[stype] = df["clean_text"].str.lower().str.contains(pattern, regex=True).astype(int)

# ─── 7. RISK SCORE (for fintech use-case) ────────────────────────────────────
df["risk_score"] = (
    -df["ensemble_score"] * 50          # sentiment contribution
    + df["vader_neg"] * 30              # explicit negativity
    + df[[*SCAM_PATTERNS]].sum(axis=1) * 5   # scam type count
).clip(0, 100).round(1)

# ─── 8. SAVE RESULTS CSV ─────────────────────────────────────────────────────
out_cols = [
    "source","subreddit","date","title",
    "vader_compound","tb_polarity","ensemble_score","sentiment",
    "tb_subjectivity","risk_score",
    *SCAM_PATTERNS.keys(),
    "score","num_comments","url"
]
df[out_cols].to_csv("/mnt/user-data/outputs/vietnam_scam_sentiment_results.csv", index=False)
print(f"✅ Processed {len(df)} reviews")
print(df["sentiment"].value_counts())
print(f"Mean ensemble score : {df['ensemble_score'].mean():.3f}")
print(f"Mean risk score     : {df['risk_score'].mean():.1f}/100")

# ─── 9. VISUALIZATION ────────────────────────────────────────────────────────
# Palette & style
NEG_COL  = "#E63946"
POS_COL  = "#2DC653"
NEU_COL  = "#F4A261"
DARK_BG  = "#0D1117"
CARD_BG  = "#161B22"
TEXT_COL = "#E6EDF3"
MUTED    = "#8B949E"
ACCENT   = "#58A6FF"
GRID_COL = "#21262D"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "axes.titlecolor":   TEXT_COL,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT_COL,
    "grid.color":        GRID_COL,
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor(DARK_BG)

gs = gridspec.GridSpec(
    4, 3,
    figure=fig,
    hspace=0.50,
    wspace=0.35,
    top=0.91, bottom=0.04,
    left=0.06, right=0.97
)

SENT_COLORS = {"Negative": NEG_COL, "Neutral": NEU_COL, "Positive": POS_COL}

# ── Helper ──
def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10, color=TEXT_COL)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, axis="y", alpha=0.3, color=GRID_COL)

# ─── Chart 1: Sentiment Distribution (donut) ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
cnt = df["sentiment"].value_counts()
colors_donut = [SENT_COLORS.get(l, MUTED) for l in cnt.index]
wedges, texts, autotexts = ax1.pie(
    cnt.values, labels=cnt.index, colors=colors_donut,
    autopct="%1.1f%%", startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=DARK_BG, linewidth=2),
    textprops=dict(color=TEXT_COL, fontsize=9),
    pctdistance=0.75,
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_color(DARK_BG)
    at.set_fontweight("bold")
ax1.set_facecolor(CARD_BG)
ax1.set_title("Sentiment Distribution\n(Ensemble Model)", fontsize=11,
              fontweight="bold", pad=10, color=TEXT_COL)
# Centre label
total = len(df)
ax1.text(0, 0, f"{total}\nreviews", ha="center", va="center",
         fontsize=10, fontweight="bold", color=TEXT_COL)

# ─── Chart 2: Ensemble Score Distribution ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
neg_data = df.loc[df["sentiment"]=="Negative", "ensemble_score"]
neu_data = df.loc[df["sentiment"]=="Neutral",  "ensemble_score"]
pos_data = df.loc[df["sentiment"]=="Positive", "ensemble_score"]
ax2.hist(neg_data, bins=25, color=NEG_COL, alpha=0.75, label="Negative", edgecolor=DARK_BG)
ax2.hist(neu_data, bins=25, color=NEU_COL, alpha=0.75, label="Neutral",  edgecolor=DARK_BG)
ax2.hist(pos_data, bins=25, color=POS_COL, alpha=0.75, label="Positive", edgecolor=DARK_BG)
ax2.axvline(0, color=MUTED, linestyle="--", linewidth=1.2, alpha=0.7)
ax2.axvline(df["ensemble_score"].mean(), color=ACCENT, linestyle="-", linewidth=1.5,
            label=f"Mean={df['ensemble_score'].mean():.2f}")
ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
style_ax(ax2, "Ensemble Sentiment Score Distribution", "Score", "Count")

# ─── Chart 3: Sentiment by Subreddit ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
sub_sent = (df.groupby(["subreddit","sentiment"])
              .size().unstack(fill_value=0))
sub_sent = sub_sent.reindex(columns=["Negative","Neutral","Positive"], fill_value=0)
bar_colors = [NEG_COL, NEU_COL, POS_COL]
sub_sent.plot(kind="bar", ax=ax3, color=bar_colors, edgecolor=DARK_BG,
              linewidth=0.5, width=0.7)
ax3.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
ax3.set_xticklabels(sub_sent.index, rotation=15, ha="right")
style_ax(ax3, "Sentiment by Subreddit", "", "Count")

# ─── Chart 4: Scam Type Frequency ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
scam_counts = df[list(SCAM_PATTERNS.keys())].sum().sort_values(ascending=True)
bars = ax4.barh(scam_counts.index, scam_counts.values,
                color=ACCENT, edgecolor=DARK_BG, linewidth=0.5, height=0.65)
for bar, val in zip(bars, scam_counts.values):
    ax4.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             str(int(val)), va="center", color=TEXT_COL, fontsize=8)
ax4.set_facecolor(CARD_BG)
ax4.set_title("Scam Type Frequency", fontsize=11, fontweight="bold",
              pad=10, color=TEXT_COL)
ax4.tick_params(colors=MUTED, labelsize=8)
for spine in ax4.spines.values():
    spine.set_edgecolor(GRID_COL)
ax4.grid(True, axis="x", alpha=0.3, color=GRID_COL)

# ─── Chart 5: Sentiment by Scam Type ─────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
scam_scores = {}
for stype in SCAM_PATTERNS:
    subset = df[df[stype] == 1]["ensemble_score"]
    if len(subset) >= 3:
        scam_scores[stype] = subset.mean()
scam_df = pd.Series(scam_scores).sort_values()
bar_c = [NEG_COL if v < -0.1 else (POS_COL if v > 0.1 else NEU_COL)
         for v in scam_df.values]
bars5 = ax5.barh(scam_df.index, scam_df.values, color=bar_c,
                 edgecolor=DARK_BG, linewidth=0.5, height=0.65)
ax5.axvline(0, color=MUTED, linestyle="--", linewidth=1, alpha=0.7)
for bar, val in zip(bars5, scam_df.values):
    ax5.text(val + (0.01 if val >= 0 else -0.01),
             bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center",
             ha="left" if val >= 0 else "right",
             color=TEXT_COL, fontsize=8)
ax5.set_facecolor(CARD_BG)
ax5.set_title("Avg Sentiment Score\nper Scam Type", fontsize=11,
              fontweight="bold", pad=10, color=TEXT_COL)
ax5.tick_params(colors=MUTED, labelsize=8)
for spine in ax5.spines.values():
    spine.set_edgecolor(GRID_COL)
ax5.grid(True, axis="x", alpha=0.3, color=GRID_COL)

# ─── Chart 6: Risk Score Distribution ────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
risk_bins = [0, 20, 40, 60, 80, 100]
risk_labels = ["Very Low\n(0-20)", "Low\n(20-40)", "Medium\n(40-60)",
               "High\n(60-80)", "Critical\n(80-100)"]
risk_colors_bar = [POS_COL, "#80C784", NEU_COL, "#FF8A65", NEG_COL]
risk_cats = pd.cut(df["risk_score"], bins=risk_bins, labels=risk_labels, include_lowest=True)
risk_cnt = risk_cats.value_counts().reindex(risk_labels, fill_value=0)
bars6 = ax6.bar(risk_labels, risk_cnt.values, color=risk_colors_bar,
                edgecolor=DARK_BG, linewidth=0.5, width=0.7)
for bar, val in zip(bars6, risk_cnt.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(int(val)), ha="center", color=TEXT_COL, fontsize=9, fontweight="bold")
style_ax(ax6, "Risk Score Distribution\n(Fintech Alert Level)", "", "Count")
ax6.tick_params(axis="x", labelsize=7)

# ─── Chart 7: Timeline — Monthly Negative Review Trend ───────────────────────
ax7 = fig.add_subplot(gs[2, :2])
df_time = df[df["date"].notna()].copy()
monthly = (df_time.groupby(["month","sentiment"])
               .size().unstack(fill_value=0)
               .reindex(columns=["Negative","Neutral","Positive"], fill_value=0))
monthly.index = monthly.index.astype(str)
x = range(len(monthly))
ax7.fill_between(x, monthly["Negative"], alpha=0.25, color=NEG_COL)
ax7.fill_between(x, monthly["Positive"], alpha=0.25, color=POS_COL)
ax7.plot(x, monthly["Negative"], color=NEG_COL, linewidth=2,
         marker="o", markersize=4, label="Negative")
ax7.plot(x, monthly["Neutral"],  color=NEU_COL, linewidth=1.5,
         marker="s", markersize=3, label="Neutral",  linestyle="--")
ax7.plot(x, monthly["Positive"], color=POS_COL, linewidth=2,
         marker="^", markersize=4, label="Positive")
step = max(1, len(monthly) // 8)
ax7.set_xticks(list(range(0, len(monthly), step)))
ax7.set_xticklabels(monthly.index[::step], rotation=30, ha="right", fontsize=8)
ax7.legend(fontsize=9, facecolor=CARD_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
style_ax(ax7, "Monthly Review Sentiment Trend", "Month", "Count")

# ─── Chart 8: VADER vs TextBlob Scatter ──────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
for label_name, color in SENT_COLORS.items():
    mask = df["sentiment"] == label_name
    ax8.scatter(
        df.loc[mask, "vader_compound"],
        df.loc[mask, "tb_polarity"],
        c=color, alpha=0.55, s=18, label=label_name, edgecolors="none"
    )
ax8.axhline(0, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.6)
ax8.axvline(0, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.6)
# Regression line
from numpy.polynomial.polynomial import polyfit
x_v = df["vader_compound"].values
y_v = df["tb_polarity"].values
mask_finite = np.isfinite(x_v) & np.isfinite(y_v)
if mask_finite.sum() > 5:
    coef = np.polyfit(x_v[mask_finite], y_v[mask_finite], 1)
    x_line = np.linspace(-1, 1, 100)
    ax8.plot(x_line, np.polyval(coef, x_line), color=ACCENT,
             linewidth=1.5, linestyle="-", alpha=0.7, label="Trend")
corr = np.corrcoef(x_v[mask_finite], y_v[mask_finite])[0,1]
ax8.text(0.03, 0.92, f"r = {corr:.3f}", transform=ax8.transAxes,
         color=ACCENT, fontsize=9, fontweight="bold")
ax8.legend(fontsize=8, facecolor=CARD_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL,
           markerscale=1.5)
style_ax(ax8, "VADER vs TextBlob\nModel Agreement", "VADER Compound", "TextBlob Polarity")

# ─── Chart 9: Word Cloud (negative reviews) ──────────────────────────────────
ax9 = fig.add_subplot(gs[3, :2])
neg_text = " ".join(df[df["sentiment"]=="Negative"]["clean_text"].tolist())
stop = set(stopwords.words("english"))
stop.update(["got", "get", "one", "us", "said", "told", "went", "back",
             "also", "would", "could", "just", "really", "like", "going",
             "ve", "re", "ll", "didn", "don", "wasn", "couldn", "didn", "even",
             "went", "came", "come", "took", "take", "want", "wanted",
             "vietnam", "vietnamese"])
wc = WordCloud(
    width=900, height=350,
    background_color=CARD_BG,
    colormap="RdYlGn_r",
    stopwords=stop,
    max_words=100,
    prefer_horizontal=0.8,
    relative_scaling=0.5,
    min_font_size=9,
    margin=4,
).generate(neg_text)
ax9.imshow(wc, interpolation="bilinear")
ax9.axis("off")
ax9.set_facecolor(CARD_BG)
ax9.set_title("Word Cloud — Negative Reviews (Top Keywords)",
              fontsize=11, fontweight="bold", pad=10, color=TEXT_COL)

# ─── Chart 10: Summary KPI cards ─────────────────────────────────────────────
ax10 = fig.add_subplot(gs[3, 2])
ax10.axis("off")
ax10.set_facecolor(CARD_BG)

neg_pct  = (df["sentiment"]=="Negative").mean()*100
pos_pct  = (df["sentiment"]=="Positive").mean()*100
avg_risk = df["risk_score"].mean()
top_scam = scam_counts.idxmax()
high_risk_n = (df["risk_score"] >= 60).sum()

kpis = [
    ("🔴  Negative Reviews",  f"{neg_pct:.1f}%",   NEG_COL),
    ("🟢  Positive Reviews",  f"{pos_pct:.1f}%",   POS_COL),
    ("⚠️   Avg Risk Score",   f"{avg_risk:.1f}/100", NEU_COL),
    ("🚨  High-Risk Posts",   f"{high_risk_n}",     "#FF8A65"),
    ("🔍  Top Scam Type",     top_scam,             ACCENT),
]
y_pos = 0.95
for icon_label, value, color in kpis:
    ax10.text(0.05, y_pos, icon_label, transform=ax10.transAxes,
              fontsize=9, color=MUTED, va="top")
    ax10.text(0.95, y_pos - 0.03, value, transform=ax10.transAxes,
              fontsize=11, fontweight="bold", color=color, va="top", ha="right")
    ax10.plot([0.03, 0.97], [y_pos - 0.12, y_pos - 0.12],
             color=GRID_COL, linewidth=0.8,
             transform=ax10.transAxes, clip_on=False)
    y_pos -= 0.185

ax10.set_title("Key Metrics Summary", fontsize=11, fontweight="bold",
               pad=10, color=TEXT_COL)

# ─── TITLE ───────────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.955,
    "Vietnam Travel Scam Reviews — Sentiment & Risk Analysis",
    ha="center", va="center",
    fontsize=17, fontweight="bold", color=TEXT_COL,
)
fig.text(
    0.5, 0.935,
    f"Ensemble Model: VADER (70%) + TextBlob (30%) with custom scam-domain lexicon  |  n = {len(df)} reviews",
    ha="center", va="center",
    fontsize=9, color=MUTED,
)

# ─── SAVE ─────────────────────────────────────────────────────────────────────
out_path = "/mnt/user-data/outputs/vietnam_scam_analysis.png"
plt.savefig(out_path, dpi=160, bbox_inches="tight",
            facecolor=DARK_BG, edgecolor="none")
plt.close()
print(f"\n✅ Chart saved → {out_path}")
print("✅ All done.")