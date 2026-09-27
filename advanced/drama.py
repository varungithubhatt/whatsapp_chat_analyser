import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()


def detect_drama(df, window_hours=2):
    """
    Drama Detector: Detects high-tension conversational spikes by scanning rolling
    time windows for simultaneous:
    1. Message volume surges (rapid back-and-forth).
    2. Negative sentiment drops (tense debates, complaints, arguments).
    3. High exclamation and ALL-CAPS intensity.

    Returns detected drama episodes, participant drama leaderboard, and drama timeline.
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'drama_events': [],
            'drama_timeline': pd.DataFrame(),
            'drama_leaderboard': pd.DataFrame(),
            'group_drama_level': "🕊️ Completely Peaceful",
            'top_instigator': "N/A",
            'insights': "Not enough message data to detect drama episodes."
        }

    # Filter omitted media
    media_pattern = r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted|Contact card omitted|sticker omitted'
    df_clean = df_clean[~df_clean['message'].str.contains(media_pattern, case=False, na=False, regex=True)].copy()

    if len(df_clean) < 5:
        return {
            'drama_events': [],
            'drama_timeline': pd.DataFrame(),
            'drama_leaderboard': pd.DataFrame(),
            'group_drama_level': "🕊️ Completely Peaceful",
            'top_instigator': "N/A",
            'insights': "Not enough text messages to identify drama patterns."
        }

    df_clean.sort_values(by='messages_date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    # Compute per-message sentiment and tension signals
    def get_msg_tension(msg):
        m_str = str(msg)
        scores = _vader.polarity_scores(m_str)
        neg = scores['neg']
        compound = scores['compound']
        excl = min(m_str.count('!') * 0.2, 0.6)
        caps = 0.3 if sum(1 for c in m_str if c.isupper()) / max(len(m_str), 1) > 0.4 else 0.0
        return neg, compound, excl + caps

    sent_results = df_clean['message'].apply(get_msg_tension)
    df_clean['neg_score'] = [r[0] for r in sent_results]
    df_clean['compound_sentiment'] = [r[1] for r in sent_results]
    df_clean['shout_score'] = [r[2] for r in sent_results]

    # Resample / group into fixed time windows (e.g. 2 hours)
    df_clean.set_index('messages_date', inplace=True)

    # Calculate group baseline volume per 2h window
    grouped = df_clean.rolling(f'{window_hours}h')

    # Let's resample by 1-hour intervals for smooth drama evaluation
    resampled = df_clean.resample('1h').agg(
        msg_count=('message', 'count'),
        neg_mean=('neg_score', 'mean'),
        sentiment_mean=('compound_sentiment', 'mean'),
        shout_mean=('shout_score', 'mean'),
        user_list=('user', list),
        sample_messages=('message', list)
    ).dropna(subset=['msg_count'])

    resampled = resampled[resampled['msg_count'] >= 3].copy()
    if resampled.empty:
        # Fallback to daily or looser threshold if sparse
        resampled = df_clean.resample('6h').agg(
            msg_count=('message', 'count'),
            neg_mean=('neg_score', 'mean'),
            sentiment_mean=('compound_sentiment', 'mean'),
            shout_mean=('shout_score', 'mean'),
            user_list=('user', list),
            sample_messages=('message', list)
        ).dropna(subset=['msg_count'])
        resampled = resampled[resampled['msg_count'] >= 2].copy()

    if resampled.empty:
        df_clean.reset_index(inplace=True)
        return {
            'drama_events': [],
            'drama_timeline': pd.DataFrame(),
            'drama_leaderboard': pd.DataFrame(),
            'group_drama_level': "🕊️ Calm & Low Conflict",
            'top_instigator': "N/A",
            'insights': "No significant volume or negativity spikes found."
        }

    # Calculate Drama Score per interval:
    # High message volume + negative sentiment + shouting / exclamation
    avg_group_vol = resampled['msg_count'].median() or 1.0

    resampled['vol_ratio'] = (resampled['msg_count'] / avg_group_vol).clip(upper=4.0)
    # Negativity component: compound < 0 or neg_mean > 0.15
    resampled['neg_factor'] = (resampled['neg_mean'] * 2.5 + np.maximum(0, -resampled['sentiment_mean']) * 1.5).clip(0, 1.0)
    resampled['shout_factor'] = resampled['shout_mean'].clip(0, 1.0)

    resampled['drama_intensity'] = (
        (0.40 * (resampled['vol_ratio'] / 4.0) +
         0.40 * resampled['neg_factor'] +
         0.20 * resampled['shout_factor']) * 100.0
    ).round(1)

    drama_events = []
    instigator_counter = {}

    for ts, row in resampled.iterrows():
        if row['drama_intensity'] >= 25.0:  # Noticeable tension
            users_in_window = [u for u in row['user_list'] if u]
            top_users = pd.Series(users_in_window).value_counts().head(3).index.tolist()

            for u in top_users:
                instigator_counter[u] = instigator_counter.get(u, 0) + 1

            # Extract sample heated quotes
            quotes = [m for m in row['sample_messages'] if len(str(m)) > 5][:2]

            if row['drama_intensity'] >= 65.0:
                severity = "🔥 High Voltage Drama / Heated Clash"
            elif row['drama_intensity'] >= 40.0:
                severity = "⚡ Moderate Debate / Friction"
            else:
                severity = "👀 Spicy Banter"

            drama_events.append({
                'timestamp': ts.strftime("%b %d, %Y (%I:%M %p)"),
                'drama_intensity': row['drama_intensity'],
                'severity': severity,
                'message_count': int(row['msg_count']),
                'sentiment_score': round(float(row['sentiment_mean']), 2),
                'key_participants': ", ".join(top_users),
                'sample_snippets': quotes
            })

    # Sort drama events by intensity
    drama_events.sort(key=lambda x: x['drama_intensity'], reverse=True)

    # Participant Drama Leaderboard
    instigator_records = []
    for user, count in instigator_counter.items():
        instigator_records.append({
            'User': user,
            'Drama Episodes Involved': count,
            'Drama Presence': "🔥 High Drama Magnet" if count >= 3 else "⚡ Occasional Friction"
        })

    lead_df = pd.DataFrame(instigator_records)
    if not lead_df.empty:
        lead_df.sort_values(by='Drama Episodes Involved', ascending=False, inplace=True)
        lead_df.reset_index(drop=True, inplace=True)
        top_instigator = lead_df.iloc[0]['User']
    else:
        top_instigator = "N/A"

    # Overall drama rating
    max_drama = drama_events[0]['drama_intensity'] if drama_events else 0.0
    if max_drama >= 65.0 or len(drama_events) >= 5:
        group_drama_level = "🌶️🌶️🌶️ High Drama & Spicy Debates"
    elif max_drama >= 35.0 or len(drama_events) >= 2:
        group_drama_level = "⚡ Moderate Group Spice & Banter"
    else:
        group_drama_level = "🕊️ Zen, Harmonious & Low Conflict"

    # Format timeline for display
    resampled_reset = resampled.reset_index()
    resampled_reset['time_str'] = resampled_reset['messages_date'].dt.strftime('%b %d %H:%M')
    timeline_df = resampled_reset[['time_str', 'drama_intensity', 'msg_count']].copy()

    df_clean.reset_index(inplace=True)

    insights = (
        f"🎭 The group's overall climate is **{group_drama_level}**. "
        f"Detected **{len(drama_events)} distinct spicy tension events**. "
        f"Top participant in heated windows: **{top_instigator}**."
    )

    return {
        'drama_events': drama_events[:10],
        'drama_timeline': timeline_df,
        'drama_leaderboard': lead_df,
        'group_drama_level': group_drama_level,
        'top_instigator': top_instigator,
        'insights': insights
    }
