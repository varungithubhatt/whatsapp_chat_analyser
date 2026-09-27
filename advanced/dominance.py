import re
import pandas as pd
import numpy as np


def analyze_dominance(df, burst_gap_minutes=30):
    """
    Computes Group Dominance and Influence scores based on multi-factor behavioral signals:
    1. Message Volume (25%)
    2. Word Volume (15%)
    3. Conversation Initiation (20%)
    4. Reply Magnet / Follow-up trigger (20%)
    5. Initiator vs Reactor Ratio (10%)
    6. Media & Resource Sharing (10%)
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'leaderboard': pd.DataFrame(),
            'top_leader': "N/A",
            'top_initiator': "N/A",
            'top_magnet': "N/A",
            'insights': "Not enough message data to compute dominance."
        }

    # Ensure sorted by date
    df_clean.sort_values(by='messages_date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    # 1. Message Volume
    total_messages = len(df_clean)
    user_msg_counts = df_clean['user'].value_counts()
    users = user_msg_counts.index.tolist()

    if len(users) == 0:
        return {'leaderboard': pd.DataFrame(), 'top_leader': "N/A"}

    # 2. Word Volume
    df_clean['word_count'] = df_clean['message'].apply(lambda m: len(str(m).split()))
    user_word_counts = df_clean.groupby('user')['word_count'].sum()
    total_words = user_word_counts.sum() if user_word_counts.sum() > 0 else 1

    # 3. Conversation Initiation (Bursts)
    df_clean['prev_date'] = df_clean['messages_date'].shift(1)
    df_clean['gap_minutes'] = (df_clean['messages_date'] - df_clean['prev_date']).dt.total_seconds() / 60.0
    df_clean['is_burst_start'] = (df_clean['gap_minutes'].isna()) | (df_clean['gap_minutes'] > burst_gap_minutes)

    burst_df = df_clean[df_clean['is_burst_start']]
    user_burst_counts = burst_df['user'].value_counts().reindex(users, fill_value=0)
    total_bursts = len(burst_df) if len(burst_df) > 0 else 1

    # 4. Reply Magnet (Messages that immediately triggered a reply from another person within 30 min)
    df_clean['next_user'] = df_clean['user'].shift(-1)
    df_clean['next_gap'] = (df_clean['messages_date'].shift(-1) - df_clean['messages_date']).dt.total_seconds() / 60.0
    df_clean['triggered_reply'] = (
        (df_clean['next_user'].notna()) &
        (df_clean['next_user'] != df_clean['user']) &
        (df_clean['next_gap'] <= burst_gap_minutes)
    )

    magnet_counts = df_clean[df_clean['triggered_reply']]['user'].value_counts().reindex(users, fill_value=0)
    total_magnet = magnet_counts.sum() if magnet_counts.sum() > 0 else 1

    # 5. Media / Link Sharing
    media_pattern = r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted|Contact card omitted|sticker omitted'
    url_pattern = r'https?://\S+|www\.\S+'

    def count_media_and_links(msg):
        msg_str = str(msg)
        is_media = 1 if re.search(media_pattern, msg_str, re.IGNORECASE) else 0
        urls = len(re.findall(url_pattern, msg_str))
        return is_media + urls

    df_clean['shares'] = df_clean['message'].apply(count_media_and_links)
    user_share_counts = df_clean.groupby('user')['shares'].sum().reindex(users, fill_value=0)
    total_shares = user_share_counts.sum() if user_share_counts.sum() > 0 else 1

    # Compile signals
    records = []
    for user in users:
        msg_cnt = int(user_msg_counts.get(user, 0))
        word_cnt = int(user_word_counts.get(user, 0))
        burst_cnt = int(user_burst_counts.get(user, 0))
        magnet_cnt = int(magnet_counts.get(user, 0))
        share_cnt = int(user_share_counts.get(user, 0))

        # Percentage signals (0 to 100)
        msg_pct = (msg_cnt / total_messages) * 100.0
        word_pct = (word_cnt / total_words) * 100.0
        init_pct = (burst_cnt / total_bursts) * 100.0
        magnet_pct = (magnet_cnt / total_magnet) * 100.0
        share_pct = (share_cnt / total_shares) * 100.0

        # Initiator vs Reactor index: how much they start vs just reply
        init_ratio = (burst_cnt / (msg_cnt + 1)) * 100.0

        # Weighted Dominance Score formula:
        # 25% Msg Volume + 15% Word Volume + 20% Initiation + 20% Reply Magnet + 10% Media Sharing + 10% Init Ratio
        raw_dominance = (
            0.25 * msg_pct +
            0.15 * word_pct +
            0.20 * init_pct +
            0.20 * magnet_pct +
            0.10 * share_pct +
            0.10 * min(init_ratio * 3, 100.0) # scaled
        )

        records.append({
            'User': user,
            'Dominance Score': raw_dominance,
            'Messages': msg_cnt,
            'Msg %': round(msg_pct, 1),
            'Words': word_cnt,
            'Word %': round(word_pct, 1),
            'Sessions Started': burst_cnt,
            'Initiation %': round(init_pct, 1),
            'Replies Triggered': magnet_cnt,
            'Magnet %': round(magnet_pct, 1),
            'Shares': share_cnt,
            'Share %': round(share_pct, 1)
        })

    leaderboard = pd.DataFrame(records)

    # Scale dominance scores to 0-100 relative to top performer or max
    max_score = leaderboard['Dominance Score'].max()
    if max_score > 0:
        leaderboard['Dominance Score'] = ((leaderboard['Dominance Score'] / max_score) * 100).round(1)
    else:
        leaderboard['Dominance Score'] = 0.0

    leaderboard.sort_values(by='Dominance Score', ascending=False, inplace=True)
    leaderboard.reset_index(drop=True, inplace=True)

    # Assign leadership roles
    top_leader = leaderboard.iloc[0]['User'] if not leaderboard.empty else "N/A"
    top_initiator = leaderboard.sort_values(by='Sessions Started', ascending=False).iloc[0]['User'] if not leaderboard.empty else "N/A"
    top_magnet = leaderboard.sort_values(by='Replies Triggered', ascending=False).iloc[0]['User'] if not leaderboard.empty else "N/A"

    leader_init_pct = leaderboard.iloc[0]['Initiation %'] if not leaderboard.empty else 0
    leader_mag_pct = leaderboard.iloc[0]['Magnet %'] if not leaderboard.empty else 0

    insights = (
        f"👑 **{top_leader}** holds the highest group dominance with a score of {leaderboard.iloc[0]['Dominance Score']}/100. "
        f"They initiate {leader_init_pct}% of new conversation sessions and trigger {leader_mag_pct}% of follow-up replies."
    )

    return {
        'leaderboard': leaderboard,
        'top_leader': top_leader,
        'top_initiator': top_initiator,
        'top_magnet': top_magnet,
        'total_sessions': total_bursts,
        'insights': insights
    }
