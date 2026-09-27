import pandas as pd
import numpy as np


def analyze_chaos(df):
    """
    Computes the Chaos Score (0-100) per user based on 5 behavioral signals:
    1. ALL CAPS message ratio (20%)
    2. Exclamation mark density (20%)
    3. Late-night message percentage (12 AM - 5 AM) (20%)
    4. Consecutive multi-texting rate (2+ msgs before others reply) (20%)
    5. Rapid-fire burst frequency (<1 min gap from own previous message) (20%)
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'chaos_leaderboard': pd.DataFrame(),
            'chaos_champion': "N/A",
            'top_chaos_score': 0.0,
            'insights': "Not enough message data to compute chaos metrics."
        }

    df_clean.sort_values(by='messages_date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    users = df_clean['user'].value_counts()
    active_users = users[users >= 2].index.tolist()
    if not active_users:
        active_users = users.index.tolist()

    # Track sequential senders and gaps
    df_clean['prev_user'] = df_clean['user'].shift(1)
    df_clean['prev_date'] = df_clean['messages_date'].shift(1)
    df_clean['gap_seconds'] = (df_clean['messages_date'] - df_clean['prev_date']).dt.total_seconds()

    # Rapid fire own burst: same user speaking within 60s of their previous message
    df_clean['is_rapid_burst'] = (
        (df_clean['prev_user'] == df_clean['user']) &
        (df_clean['gap_seconds'] <= 60.0)
    )

    # Multi-texting: same user sending consecutive messages
    df_clean['is_multi_text'] = (df_clean['prev_user'] == df_clean['user'])

    user_records = []

    for user in active_users:
        u_df = df_clean[df_clean['user'] == user].copy()
        msg_count = len(u_df)

        if msg_count == 0:
            continue

        # 1. ALL CAPS ratio
        def is_caps_msg(msg):
            words = [w for w in str(msg).split() if len(w) > 1 and w.isalpha()]
            if not words:
                return False
            caps_count = sum(1 for w in words if w.isupper())
            return (caps_count / len(words)) >= 0.5

        caps_msgs = u_df['message'].apply(is_caps_msg).sum()
        caps_pct = (caps_msgs / msg_count) * 100.0

        # 2. Exclamation density
        total_excl = u_df['message'].apply(lambda m: str(m).count('!')).sum()
        excl_density = (total_excl / msg_count)  # e.g., 0.5 exclamations per message

        # 3. Late-night % (00:00 to 05:00)
        late_night_msgs = u_df[(u_df['hour'] >= 0) & (u_df['hour'] < 5)]
        late_night_pct = (len(late_night_msgs) / msg_count) * 100.0

        # 4. Multi-texting rate
        multi_texts = u_df['is_multi_text'].sum()
        multi_text_pct = (multi_texts / msg_count) * 100.0

        # 5. Rapid burst rate (<1 min)
        rapid_bursts = u_df['is_rapid_burst'].sum()
        rapid_burst_pct = (rapid_bursts / msg_count) * 100.0

        # Normalize components (0 to 1)
        norm_caps = min(caps_pct / 30.0, 1.0)
        norm_excl = min(excl_density / 1.5, 1.0)
        norm_late = min(late_night_pct / 35.0, 1.0)
        norm_multi = min(multi_text_pct / 60.0, 1.0)
        norm_rapid = min(rapid_burst_pct / 50.0, 1.0)

        # Weighted Chaos Score (0 to 100)
        chaos_score = round(
            (0.20 * norm_caps +
             0.20 * norm_excl +
             0.20 * norm_late +
             0.20 * norm_multi +
             0.20 * norm_rapid) * 100.0,
            1
        )

        # Classification label
        if chaos_score >= 60.0:
            chaos_tier = "🌪️ Unhinged Chaos"
        elif chaos_score >= 30.0:
            chaos_tier = "⚡ Mild Chaos"
        else:
            chaos_tier = "🧘 Zen & Composed"

        user_records.append({
            'User': user,
            'Chaos Score': chaos_score,
            'Chaos Tier': chaos_tier,
            'ALL CAPS %': round(caps_pct, 1),
            'Exclamations/Msg': round(excl_density, 2),
            'Late Night %': round(late_night_pct, 1),
            'Multi-Text %': round(multi_text_pct, 1),
            'Rapid Burst %': round(rapid_burst_pct, 1)
        })

    chaos_df = pd.DataFrame(user_records)
    if not chaos_df.empty:
        chaos_df.sort_values(by='Chaos Score', ascending=False, inplace=True)
        chaos_df.reset_index(drop=True, inplace=True)
        chaos_champion = chaos_df.iloc[0]['User']
        top_chaos_score = chaos_df.iloc[0]['Chaos Score']
        top_tier = chaos_df.iloc[0]['Chaos Tier']
        insights = (
            f"🌪️ **{chaos_champion}** is crowned the **Chaos Champion** with a score of {top_chaos_score}/100 ({top_tier}). "
            f"They send {chaos_df.iloc[0]['Multi-Text %']}% multi-texts and {chaos_df.iloc[0]['Late Night %']}% late-night messages."
        )
    else:
        chaos_champion = "N/A"
        top_chaos_score = 0.0
        insights = "Not enough variation to determine chaos levels."

    return {
        'chaos_leaderboard': chaos_df,
        'chaos_champion': chaos_champion,
        'top_chaos_score': top_chaos_score,
        'insights': insights
    }
