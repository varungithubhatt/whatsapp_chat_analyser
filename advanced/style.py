import re
import unicodedata
import pandas as pd
import numpy as np


def _is_emoji(char):
    """Checks if a character is an emoji using unicode categorization."""
    cat = unicodedata.category(char)
    if cat in ('So', 'Sk'):
        return True
    code = ord(char)
    return (
        0x1F600 <= code <= 0x1F64F or  # Emoticons
        0x1F300 <= code <= 0x1F5FF or  # Misc Symbols and Pictographs
        0x1F680 <= code <= 0x1F6FF or  # Transport and Map
        0x1F1E0 <= code <= 0x1F1FF or  # Flags
        0x2600 <= code <= 0x26FF or    # Misc symbols
        0x2700 <= code <= 0x27BF or    # Dingbats
        0xFE00 <= code <= 0xFE0F or    # Variation Selectors
        0x1F900 <= code <= 0x1F9FF or  # Supplemental Symbols and Pictographs
        0x1FA70 <= code <= 0x1FAFF     # Symbols and Pictographs Extended-A
    )


def analyze_communication_styles(df):
    """
    Analyzes individual communication styles and assigns behavioral personality badges:
    - 📜 The Essayist: Writes long, detailed messages
    - 💬 Rapid Texter: Sends multiple short, rapid-fire messages
    - 😂 Emoji Enthusiast: Uses lots of emojis
    - ❓ The Curious One: Asks lots of questions
    - ⚡ The Energetic One: High exclamation mark and ALL-CAPS usage
    - 🦉 Night Owl: Chats predominantly late at night (00:00 - 06:00)
    - 🌅 Early Bird: Most active early in the morning (06:00 - 11:00)
    - 📸 Media Mogul: Shares lots of photos, links, and documents
    - 🧘 The Balanced One: Steady, adaptable conversationalist
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'style_profiles': pd.DataFrame(),
            'user_badges': {},
            'user_details': {},
            'insights': "Not enough message data to compute communication styles."
        }

    media_pattern = r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted|Contact card omitted|sticker omitted'
    url_pattern = r'https?://\S+|www\.\S+'

    users = df_clean['user'].value_counts()
    active_users = users[users >= 2].index.tolist()

    if not active_users:
        active_users = users.index.tolist()

    profiles = []
    user_badges = {}
    user_details = {}

    for user in active_users:
        user_df = df_clean[df_clean['user'] == user].copy()
        msg_count = len(user_df)

        # Word count stats
        words_per_msg = user_df['message'].apply(lambda m: len(str(m).split()))
        avg_words = float(words_per_msg.mean()) if msg_count > 0 else 0.0

        # Emojis count
        def count_emojis(text):
            return sum(1 for c in str(text) if _is_emoji(c))

        emojis_per_msg = user_df['message'].apply(count_emojis)
        avg_emojis = float(emojis_per_msg.mean()) if msg_count > 0 else 0.0
        total_emojis = int(emojis_per_msg.sum())

        # Punctuation & Caps
        q_count = user_df['message'].apply(lambda m: str(m).count('?')).sum()
        excl_count = user_df['message'].apply(lambda m: str(m).count('!')).sum()

        def caps_word_count(text):
            words = [w for w in str(text).split() if len(w) > 1 and w.isupper() and not w.isdigit()]
            return len(words)

        caps_count = user_df['message'].apply(caps_word_count).sum()

        q_per_100 = round((q_count / msg_count) * 100, 1) if msg_count > 0 else 0.0
        excl_per_100 = round((excl_count / msg_count) * 100, 1) if msg_count > 0 else 0.0
        caps_per_100 = round((caps_count / msg_count) * 100, 1) if msg_count > 0 else 0.0

        # Time of day
        night_msgs = user_df[(user_df['hour'] >= 0) & (user_df['hour'] < 6)]
        morning_msgs = user_df[(user_df['hour'] >= 6) & (user_df['hour'] < 12)]
        night_pct = round((len(night_msgs) / msg_count) * 100, 1) if msg_count > 0 else 0.0
        morning_pct = round((len(morning_msgs) / msg_count) * 100, 1) if msg_count > 0 else 0.0

        # Media & Link Shares
        shares = user_df['message'].apply(
            lambda m: 1 if re.search(media_pattern, str(m), re.IGNORECASE) or re.search(url_pattern, str(m)) else 0
        ).sum()
        share_pct = round((shares / msg_count) * 100, 1) if msg_count > 0 else 0.0

        # Assign Badges
        badges = []

        if avg_words >= 15:
            badges.append({"badge": "📜 The Essayist", "desc": f"Writes detailed messages (avg {round(avg_words,1)} words/msg)"})
        elif avg_words <= 4.5 and msg_count >= 10:
            badges.append({"badge": "💬 Rapid Texter", "desc": f"Brisk and punchy (avg {round(avg_words,1)} words/msg)"})

        if avg_emojis >= 1.0:
            badges.append({"badge": "😂 Emoji Enthusiast", "desc": f"Expressive chatter (avg {round(avg_emojis,1)} emojis/msg)"})

        if q_per_100 >= 30.0:
            badges.append({"badge": "❓ The Curious One", "desc": f"Asks questions often ({q_per_100}% questions asked)"})

        if excl_per_100 >= 35.0 or caps_per_100 >= 20.0:
            badges.append({"badge": "⚡ The Energetic One", "desc": "High excitement, exclamation & caps energy!"})

        if night_pct >= 25.0:
            badges.append({"badge": "🦉 Night Owl", "desc": f"{night_pct}% of messages sent between midnight and 6 AM"})
        elif morning_pct >= 40.0:
            badges.append({"badge": "🌅 Early Bird", "desc": f"{morning_pct}% of messages sent in early morning"})

        if share_pct >= 15.0:
            badges.append({"badge": "📸 Media Mogul", "desc": f"Frequently shares photos, videos, and links ({share_pct}%)"})

        if not badges:
            badges.append({"badge": "🧘 The Balanced One", "desc": "Steady, adaptable, and composed conversationalist"})

        user_badges[user] = badges

        user_details[user] = {
            'avg_words': round(avg_words, 1),
            'avg_emojis': round(avg_emojis, 1),
            'total_emojis': total_emojis,
            'question_rate': q_per_100,
            'energy_rate': excl_per_100,
            'night_activity': night_pct,
            'morning_activity': morning_pct,
            'share_pct': share_pct,
            'primary_badge': badges[0]['badge']
        }

        profiles.append({
            'User': user,
            'Primary Style': badges[0]['badge'],
            'Avg Words/Msg': round(avg_words, 1),
            'Emojis/Msg': round(avg_emojis, 2),
            'Questions %': q_per_100,
            'Energy Score %': excl_per_100,
            'Night Owl %': night_pct,
            'Media %': share_pct
        })

    profiles_df = pd.DataFrame(profiles)

    insights = "Each group member brings a unique tone and signature communication rhythm."

    return {
        'style_profiles': profiles_df,
        'user_badges': user_badges,
        'user_details': user_details,
        'insights': insights
    }
