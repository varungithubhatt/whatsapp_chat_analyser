import pandas as pd
import numpy as np

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
except ImportError:
    _analyzer = None


def _get_vader_score(text):
    """Calculates VADER compound score for a text, with simple fallback if library missing."""
    if not text or not isinstance(text, str):
        return 0.0
    if _analyzer is not None:
        try:
            return _analyzer.polarity_scores(text)['compound']
        except Exception:
            return 0.0

    # Simple heuristic fallback if vaderSentiment is unavailable
    pos_words = {'good', 'great', 'awesome', 'happy', 'love', 'nice', 'cool', 'excellent', 'congrats', 'thanks', 'thank', 'yay', 'haha', 'lol'}
    neg_words = {'bad', 'terrible', 'awful', 'sad', 'hate', 'angry', 'worst', 'annoying', 'sorry', 'pain', 'fail', 'upset'}
    words = text.lower().split()
    pos = sum(1 for w in words if w in pos_words)
    neg = sum(1 for w in words if w in neg_words)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def analyze_sentiment(selected_user, df):
    """
    Performs comprehensive sentiment and mood analysis on chat messages.
    Returns overall mood indicators, swing index, timelines, user leaderboards, and highlights.
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    media_pattern = r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted|Contact card omitted|sticker omitted'
    df_clean = df_clean[~df_clean['message'].str.contains(media_pattern, case=False, na=False, regex=True)]

    if df_clean.empty:
        return {
            'avg_score': 0.0,
            'pos_pct': 0.0,
            'neu_pct': 100.0,
            'neg_pct': 0.0,
            'mood_badge': "😐 Neutral",
            'mood_swing_index': 0.0,
            'timeline': pd.DataFrame(),
            'user_leaderboard': pd.DataFrame(),
            'pos_highlights': [],
            'neg_highlights': []
        }

    # Compute sentiment scores for all messages
    df_clean['sentiment_score'] = df_clean['message'].apply(_get_vader_score)

    def categorize_score(s):
        if s >= 0.05:
            return 'Positive'
        elif s <= -0.05:
            return 'Negative'
        return 'Neutral'

    df_clean['sentiment_label'] = df_clean['sentiment_score'].apply(categorize_score)

    # Filter for selected user if not 'Overall'
    if selected_user != 'Overall':
        df_target = df_clean[df_clean['user'] == selected_user].copy()
    else:
        df_target = df_clean.copy()

    if df_target.empty:
        df_target = df_clean.copy()

    total_msgs = len(df_target)
    pos_count = (df_target['sentiment_label'] == 'Positive').sum()
    neu_count = (df_target['sentiment_label'] == 'Neutral').sum()
    neg_count = (df_target['sentiment_label'] == 'Negative').sum()

    pos_pct = round((pos_count / total_msgs) * 100, 1) if total_msgs > 0 else 0.0
    neu_pct = round((neu_count / total_msgs) * 100, 1) if total_msgs > 0 else 0.0
    neg_pct = round((neg_count / total_msgs) * 100, 1) if total_msgs > 0 else 0.0
    avg_score = round(float(df_target['sentiment_score'].mean()), 3) if total_msgs > 0 else 0.0
    mood_swing_index = round(float(df_target['sentiment_score'].std()), 3) if total_msgs > 1 else 0.0
    if np.isnan(mood_swing_index):
        mood_swing_index = 0.0

    # Determine Mood Badge
    if avg_score >= 0.25:
        mood_badge = f"🌟 Highly Positive & Energetic ({pos_pct}% Pos)"
    elif avg_score >= 0.08:
        mood_badge = f"😊 Warm & Pleasant ({pos_pct}% Pos)"
    elif avg_score >= -0.08:
        mood_badge = f"😐 Balanced & Casual ({neu_pct}% Neu)"
    elif avg_score >= -0.25:
        mood_badge = f"🙁 Slightly Critical / Somber ({neg_pct}% Neg)"
    else:
        mood_badge = f"🚨 Tense / Highly Negative ({neg_pct}% Neg)"

    # Timeline of sentiment over time
    df_target['time_period'] = df_target['month'] + '-' + df_target['year'].astype(str)
    timeline = df_target.groupby(['year', 'month_num', 'month', 'time_period']).agg(
        avg_sentiment=('sentiment_score', 'mean'),
        message_count=('sentiment_score', 'count'),
        positive_count=('sentiment_label', lambda x: (x == 'Positive').sum()),
        negative_count=('sentiment_label', lambda x: (x == 'Negative').sum())
    ).reset_index()

    timeline.sort_values(by=['year', 'month_num'], inplace=True)
    timeline['avg_sentiment'] = timeline['avg_sentiment'].round(3)
    timeline['pos_pct'] = ((timeline['positive_count'] / timeline['message_count']) * 100).round(1)
    timeline['neg_pct'] = ((timeline['negative_count'] / timeline['message_count']) * 100).round(1)

    # Per-user leaderboard (computed across all users)
    user_stats = []
    for user, group in df_clean.groupby('user'):
        u_total = len(group)
        if u_total < 2:
            continue
        u_avg = group['sentiment_score'].mean()
        u_pos = (group['sentiment_label'] == 'Positive').sum() / u_total * 100
        u_neu = (group['sentiment_label'] == 'Neutral').sum() / u_total * 100
        u_neg = (group['sentiment_label'] == 'Negative').sum() / u_total * 100
        u_vol = group['sentiment_score'].std()
        if np.isnan(u_vol):
            u_vol = 0.0

        user_stats.append({
            'User': user,
            'Messages': u_total,
            'Avg Sentiment': round(u_avg, 3),
            'Positive %': round(u_pos, 1),
            'Neutral %': round(u_neu, 1),
            'Negative %': round(u_neg, 1),
            'Mood Swing Index': round(u_vol, 3),
            'Mood Vibe': '😊 Cheerful' if u_avg >= 0.15 else ('😐 Calm' if u_avg >= -0.05 else '😤 Critical')
        })

    user_leaderboard = pd.DataFrame(user_stats)
    if not user_leaderboard.empty:
        user_leaderboard.sort_values(by='Avg Sentiment', ascending=False, inplace=True)
        user_leaderboard.reset_index(drop=True, inplace=True)

    # Top Positive and Negative Highlights
    pos_highlights = []
    neg_highlights = []

    pos_candidates = df_target[df_target['sentiment_score'] > 0.4].sort_values(by='sentiment_score', ascending=False).head(3)
    for _, row in pos_candidates.iterrows():
        pos_highlights.append({
            'user': row['user'],
            'date': str(row['messages_date']),
            'message': row['message'],
            'score': round(row['sentiment_score'], 2)
        })

    neg_candidates = df_target[df_target['sentiment_score'] < -0.3].sort_values(by='sentiment_score', ascending=True).head(3)
    for _, row in neg_candidates.iterrows():
        neg_highlights.append({
            'user': row['user'],
            'date': str(row['messages_date']),
            'message': row['message'],
            'score': round(row['sentiment_score'], 2)
        })

    return {
        'avg_score': avg_score,
        'pos_pct': pos_pct,
        'neu_pct': neu_pct,
        'neg_pct': neg_pct,
        'mood_badge': mood_badge,
        'mood_swing_index': mood_swing_index,
        'timeline': timeline,
        'user_leaderboard': user_leaderboard,
        'pos_highlights': pos_highlights,
        'neg_highlights': neg_highlights
    }
