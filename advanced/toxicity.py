import re
import pandas as pd
import numpy as np

# Optional better_profanity import
try:
    from better_profanity import profanity
    profanity.load_censor_words()
    _HAS_PROFANITY_LIB = True
except Exception:
    _HAS_PROFANITY_LIB = False

# Optional detoxify import
try:
    from detoxify import Detoxify
    import torch
    _detox_model = Detoxify('original-small') if torch.cuda.is_available() else Detoxify('original')
    _HAS_DETOXIFY = True
except Exception:
    _detox_model = None
    _HAS_DETOXIFY = False

# Multilingual slang / profanity vocabulary (English + Hinglish/Hindi slang)
CUSTOM_SWEAR_WORDS = {
    'fuck', 'fucking', 'fucked', 'fucker', 'fucks', 'shit', 'shitty', 'bitch', 'bitches',
    'asshole', 'ass', 'bastard', 'dick', 'pussy', 'cunt', 'slut', 'whore', 'motherfucker',
    'bullshit', 'damn', 'crap', 'stfu', 'wtf', 'wth', 'lmao',
    # Hinglish / Hindi slang
    'bc', 'mc', 'bkl', 'bsdk', 'bhosdike', 'bhosadike', 'chutiya', 'chutiye', 'chutiyapa',
    'madarchod', 'behenchod', 'behen ke lode', 'bhenchod', 'gandu', 'gaand', 'gand', 'lund',
    'lauda', 'lode', 'chut', 'harami', 'kamina', 'kamine', 'saala', 'saale', 'kutte', 'kutta',
    'randi', 'tatti', 'jhant', 'jhaat', 'jhaatu', 'suar', 'bakchod', 'bakchodi'
}


def _contains_swear_words(text):
    if not text or not isinstance(text, str):
        return False
    clean_text = text.lower()
    # Check better_profanity if loaded
    if _HAS_PROFANITY_LIB and profanity.contains_profanity(clean_text):
        return True
    # Token check against custom multilingual list
    words = re.findall(r'\b\w+\b', clean_text)
    for w in words:
        if w in CUSTOM_SWEAR_WORDS:
            return True
    return False


def _heuristic_toxicity(text):
    """Fallback lexicon-based toxicity/insult estimation (0.0 to 1.0)."""
    if not text or not isinstance(text, str):
        return 0.0
    lower = text.lower()
    words = re.findall(r'\b\w+\b', lower)
    if not words:
        return 0.0

    swear_count = sum(1 for w in words if w in CUSTOM_SWEAR_WORDS)
    insult_words = {'idiot', 'moron', 'stupid', 'loser', 'dumb', 'fool', 'trash', 'shut up', 'hate you', 'clown'}
    insult_count = sum(1 for w in words if w in insult_words)

    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    excl_ratio = min(text.count('!') * 0.1, 0.3)

    raw_score = (swear_count * 0.45) + (insult_count * 0.35) + (caps_ratio * 0.2) + excl_ratio
    return min(round(float(raw_score), 3), 1.0)


def analyze_toxicity(df):
    """
    Computes Vulgarity and Toxicity statistics:
    1. Rule-based Vulgarity Score (% of messages containing profanity).
    2. Model/heuristic Toxicity Score (0.0 to 1.0).
    3. User vulgarity & toxicity leaderboards.
    4. Timeline of vulgarity & spice levels over time.
    5. Playful group title & spice badges.
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'vulgarity_leaderboard': pd.DataFrame(),
            'spiciest_user': "N/A",
            'top_vulgarity_pct': 0.0,
            'top_toxic_user': "N/A",
            'group_vulgarity_pct': 0.0,
            'timeline': pd.DataFrame(),
            'insights': "Not enough message data for vulgarity analysis."
        }

    # Remove media omitted markers
    media_pattern = r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted|Contact card omitted|sticker omitted'
    df_clean = df_clean[~df_clean['message'].str.contains(media_pattern, case=False, na=False, regex=True)].copy()

    if df_clean.empty:
        return {
            'vulgarity_leaderboard': pd.DataFrame(),
            'spiciest_user': "N/A",
            'top_vulgarity_pct': 0.0,
            'top_toxic_user': "N/A",
            'group_vulgarity_pct': 0.0,
            'timeline': pd.DataFrame(),
            'insights': "No text messages available."
        }

    # 1. Vulgarity flag
    df_clean['has_swear'] = df_clean['message'].apply(_contains_swear_words)

    # 2. Toxicity score
    if _HAS_DETOXIFY and _detox_model is not None:
        try:
            # Batch inference
            msgs = df_clean['message'].tolist()
            preds = _detox_model.predict(msgs)
            df_clean['toxicity_score'] = [round(float(s), 3) for s in preds['toxicity']]
        except Exception:
            df_clean['toxicity_score'] = df_clean['message'].apply(_heuristic_toxicity)
    else:
        df_clean['toxicity_score'] = df_clean['message'].apply(_heuristic_toxicity)

    # User breakdown
    users = df_clean['user'].value_counts()
    active_users = users[users >= 2].index.tolist()
    if not active_users:
        active_users = users.index.tolist()

    user_stats = []
    for user in active_users:
        u_df = df_clean[df_clean['user'] == user]
        u_total = len(u_df)
        u_swear = int(u_df['has_swear'].sum())
        u_vulg_pct = round((u_swear / u_total) * 100.0, 1) if u_total > 0 else 0.0
        u_avg_tox = round(float(u_df['toxicity_score'].mean() * 100.0), 1) if u_total > 0 else 0.0

        if u_vulg_pct >= 15.0 or u_avg_tox >= 30.0:
            spice_badge = "🌶️🌶️🌶️ Flaming Hot"
        elif u_vulg_pct >= 5.0 or u_avg_tox >= 15.0:
            spice_badge = "🌶️ Spicy"
        elif u_vulg_pct > 0.0:
            spice_badge = "🧂 Mild Seasoning"
        else:
            spice_badge = "😇 Saintly Clean"

        user_stats.append({
            'User': user,
            'Total Messages': u_total,
            'Swear Messages': u_swear,
            'Vulgarity %': u_vulg_pct,
            'Toxicity Index (0-100)': u_avg_tox,
            'Spice Rating': spice_badge
        })

    vulg_df = pd.DataFrame(user_stats)
    vulg_df.sort_values(by='Vulgarity %', ascending=False, inplace=True)
    vulg_df.reset_index(drop=True, inplace=True)

    # Group metrics
    total_group_msgs = len(df_clean)
    total_group_swears = int(df_clean['has_swear'].sum())
    group_vulg_pct = round((total_group_swears / total_group_msgs) * 100.0, 1) if total_group_msgs > 0 else 0.0

    spiciest_user = vulg_df.iloc[0]['User'] if not vulg_df.empty else "N/A"
    top_vulgarity_pct = vulg_df.iloc[0]['Vulgarity %'] if not vulg_df.empty else 0.0

    top_toxic_user = vulg_df.sort_values(by='Toxicity Index (0-100)', ascending=False).iloc[0]['User'] if not vulg_df.empty else "N/A"

    # Timeline of vulgarity & toxicity
    df_clean['time_period'] = df_clean['month'] + '-' + df_clean['year'].astype(str)
    timeline = df_clean.groupby(['year', 'month_num', 'month', 'time_period']).agg(
        total_msgs=('message', 'count'),
        swear_msgs=('has_swear', 'sum'),
        avg_toxicity=('toxicity_score', 'mean')
    ).reset_index()

    timeline.sort_values(by=['year', 'month_num'], inplace=True)
    timeline['vulgarity_pct'] = ((timeline['swear_msgs'] / timeline['total_msgs']) * 100.0).round(1)
    timeline['toxicity_index'] = (timeline['avg_toxicity'] * 100.0).round(1)

    insights = (
        f"🤬 **{spiciest_user}** is the group's Spiciest Texter ({top_vulgarity_pct}% of their messages contain profanity). "
        f"Overall, the group averages {group_vulg_pct}% vulgarity."
    )

    return {
        'vulgarity_leaderboard': vulg_df,
        'spiciest_user': spiciest_user,
        'top_vulgarity_pct': top_vulgarity_pct,
        'top_toxic_user': top_toxic_user,
        'group_vulgarity_pct': group_vulg_pct,
        'timeline': timeline,
        'insights': insights
    }
