import pandas as pd
import numpy as np


def analyze_ghost_and_revival(df):
    """
    Analyzes:
    1. Ghost Score: Percentage of messages that received no reply within 30 minutes,
       and average response waiting time when someone does reply.
    2. Chat Reviver Score: Who breaks conversational silences (>2 hours of group inactivity)
       and breathes life back into the group.
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'ghost_leaderboard': pd.DataFrame(),
            'top_ghosted_user': "N/A",
            'top_ghost_rate': 0.0,
            'reviver_leaderboard': pd.DataFrame(),
            'top_reviver': "N/A",
            'total_revivals': 0,
            'insights': "Not enough message data to compute ghost and reviver metrics."
        }

    df_clean.sort_values(by='messages_date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    n = len(df_clean)
    users = df_clean['user'].value_counts()
    active_users = users[users >= 2].index.tolist()
    if not active_users:
        active_users = users.index.tolist()

    # Track next sender and gap
    df_clean['next_user'] = df_clean['user'].shift(-1)
    df_clean['next_date'] = df_clean['messages_date'].shift(-1)
    df_clean['forward_gap_mins'] = (df_clean['next_date'] - df_clean['messages_date']).dt.total_seconds() / 60.0

    # Track previous sender and backward gap for silence revival (>120 mins)
    df_clean['prev_user'] = df_clean['user'].shift(1)
    df_clean['prev_date'] = df_clean['messages_date'].shift(1)
    df_clean['backward_gap_mins'] = (df_clean['messages_date'] - df_clean['prev_date']).dt.total_seconds() / 60.0

    # A message is a "Revival" if it occurs after a silence of >= 120 minutes (2 hours)
    df_clean['is_revival'] = df_clean['backward_gap_mins'] >= 120.0

    # 1. GHOST / LEFT-ON-READ ANALYSIS
    ghost_records = []
    for user in active_users:
        u_msgs = df_clean[df_clean['user'] == user]
        total_sent = len(u_msgs)
        if total_sent == 0:
            continue

        # A message is ghosted if:
        # either no one replied within 30 min, or the next message was after a long gap / by same user
        ghosted_count = 0
        wait_times = []

        for idx, row in u_msgs.iterrows():
            next_u = row['next_user']
            gap = row['forward_gap_mins']

            if pd.isna(next_u) or gap > 30.0:
                ghosted_count += 1
            elif next_u != user and gap <= 30.0:
                wait_times.append(gap)

        ghost_rate = round((ghosted_count / total_sent) * 100.0, 1)
        avg_wait = round(float(np.mean(wait_times)), 1) if wait_times else 0.0

        if ghost_rate >= 40.0:
            ghost_badge = "👻 Ghost Magnet (High Ignored Rate)"
        elif ghost_rate >= 20.0:
            ghost_badge = "👀 Occasional Echo"
        else:
            ghost_badge = "⚡ Instant Responder Favorite"

        ghost_records.append({
            'User': user,
            'Total Sent': total_sent,
            'Ghosted Msgs': ghosted_count,
            'Ghost Rate %': ghost_rate,
            'Avg Reply Wait (min)': avg_wait,
            'Ghost Status': ghost_badge
        })

    ghost_df = pd.DataFrame(ghost_records)
    if not ghost_df.empty:
        ghost_df.sort_values(by='Ghost Rate %', ascending=False, inplace=True)
        ghost_df.reset_index(drop=True, inplace=True)
        top_ghosted_user = ghost_df.iloc[0]['User']
        top_ghost_rate = ghost_df.iloc[0]['Ghost Rate %']
    else:
        top_ghosted_user = "N/A"
        top_ghost_rate = 0.0

    # 2. REVIVER ANALYSIS
    revival_df = df_clean[df_clean['is_revival']].copy()
    total_revivals = len(revival_df)

    reviver_records = []
    for user in active_users:
        u_revivals = len(revival_df[revival_df['user'] == user])
        u_total = len(df_clean[df_clean['user'] == user])
        revival_share = round((u_revivals / total_revivals) * 100.0, 1) if total_revivals > 0 else 0.0

        if u_revivals >= 5:
            revive_badge = "🩺 Chat Defibrillator (Group Resuscitator)"
        elif u_revivals >= 2:
            revive_badge = "🌅 Regular Morning/Silence Breaker"
        else:
            revive_badge = "🛋️ Silent Observer"

        reviver_records.append({
            'User': user,
            'Silences Broken (>2 hrs)': u_revivals,
            'Share of Group Revivals %': revival_share,
            'Reviver Badge': revive_badge
        })

    reviver_df = pd.DataFrame(reviver_records)
    if not reviver_df.empty:
        reviver_df.sort_values(by='Silences Broken (>2 hrs)', ascending=False, inplace=True)
        reviver_df.reset_index(drop=True, inplace=True)
        top_reviver = reviver_df.iloc[0]['User']
    else:
        top_reviver = "N/A"

    insights = (
        f"👻 **{top_ghosted_user}** faces the highest ghost rate at {top_ghost_rate}%, "
        f"while 🩺 **{top_reviver}** is the primary **Chat Reviver**, breaking group silence {reviver_df.iloc[0]['Silences Broken (>2 hrs)'] if not reviver_df.empty else 0} times."
    )

    return {
        'ghost_leaderboard': ghost_df,
        'top_ghosted_user': top_ghosted_user,
        'top_ghost_rate': top_ghost_rate,
        'reviver_leaderboard': reviver_df,
        'top_reviver': top_reviver,
        'total_revivals': total_revivals,
        'insights': insights
    }
