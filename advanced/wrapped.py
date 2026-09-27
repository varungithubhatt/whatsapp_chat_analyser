import io
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_group_wrapped(df, group_title="WhatsApp Group"):
    """
    Generates a Spotify-Wrapped style annual/group recap:
    1. Comprehensive Group Superlatives & Awards.
    2. Core Year/Lifetime Chat Milestones (total messages, words, peak times, vibe).
    3. High-resolution downloadable Image Card (PNG) synthesized via Pillow.
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'awards': [],
            'metrics': {},
            'vibe': "No Data",
            'image_bytes': None,
            'insights': "Not enough message data for Group Wrapped."
        }

    df_clean.sort_values(by='messages_date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    # 1. Basic Stats
    total_messages = len(df_clean)
    total_words = df_clean['message'].apply(lambda x: len(str(x).split())).sum()

    media_pattern = r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted|Contact card omitted|sticker omitted'
    total_media = int(df_clean['message'].str.contains(media_pattern, case=False, na=False, regex=True).sum())

    users = df_clean['user'].value_counts()
    active_users = users.index.tolist()

    # Dates & Timings
    start_date = df_clean['messages_date'].min().strftime('%b %Y')
    end_date = df_clean['messages_date'].max().strftime('%b %Y')
    busiest_day = df_clean['day_name'].mode()[0] if 'day_name' in df_clean.columns else "N/A"
    busiest_hour = f"{int(df_clean['hour'].mode()[0]):02d}:00" if 'hour' in df_clean.columns else "N/A"

    # 2. Compute Superlatives
    user_stats = {}
    for user in active_users:
        u_df = df_clean[df_clean['user'] == user]
        m_count = len(u_df)
        w_count = u_df['message'].apply(lambda x: len(str(x).split())).sum()
        avg_w = w_count / max(m_count, 1)

        # Late night (00:00 to 05:00)
        late_night = len(u_df[(u_df['hour'] >= 0) & (u_df['hour'] < 5)])

        # Emojis count (simple ascii / regex count for emoji estimation)
        import unicodedata
        emojis_count = sum(
            1 for msg in u_df['message']
            for char in str(msg)
            if unicodedata.category(char) in ('So', 'Sm', 'Sk') or ord(char) > 0x1F000
        )

        user_stats[user] = {
            'messages': m_count,
            'words': w_count,
            'avg_words': round(avg_w, 1),
            'late_night': late_night,
            'emojis': emojis_count
        }

    # Sequential gaps for rapid burst & multi-text
    df_clean['prev_user'] = df_clean['user'].shift(1)
    df_clean['prev_date'] = df_clean['messages_date'].shift(1)
    df_clean['gap_s'] = (df_clean['messages_date'] - df_clean['prev_date']).dt.total_seconds()
    df_clean['backward_gap_mins'] = df_clean['gap_s'] / 60.0

    multi_text_counts = df_clean[df_clean['prev_user'] == df_clean['user']]['user'].value_counts()
    revival_counts = df_clean[df_clean['backward_gap_mins'] >= 120.0]['user'].value_counts()

    # Assign awards
    awards = []

    # 1. Main Character
    main_char = users.index[0] if len(users) > 0 else "N/A"
    awards.append({
        'icon': '👑',
        'title': 'The Main Character',
        'winner': main_char,
        'subtitle': f'{users.iloc[0]:,} messages ({round((users.iloc[0]/total_messages)*100, 1)}% of all chat)'
    })

    # 2. The Novelist (Longest average messages)
    novelist = max(user_stats.items(), key=lambda x: x[1]['avg_words'])[0] if user_stats else "N/A"
    awards.append({
        'icon': '📜',
        'title': 'The Novelist',
        'winner': novelist,
        'subtitle': f'Averages {user_stats[novelist]["avg_words"]} words per message'
    })

    # 3. The Machine Gun (Most multi-texts / consecutive bursts)
    if not multi_text_counts.empty:
        machine_gun = multi_text_counts.index[0]
        awards.append({
            'icon': '⚡',
            'title': 'The Machine Gun',
            'winner': machine_gun,
            'subtitle': f'{multi_text_counts.iloc[0]:,} rapid consecutive messages'
        })

    # 4. The Night Owl / Vampire
    vampire = max(user_stats.items(), key=lambda x: x[1]['late_night'])[0] if user_stats else "N/A"
    if user_stats[vampire]['late_night'] > 0:
        awards.append({
            'icon': '🦉',
            'title': 'The Night Owl',
            'winner': vampire,
            'subtitle': f'{user_stats[vampire]["late_night"]:,} messages sent after midnight'
        })

    # 5. The Chat Defibrillator
    if not revival_counts.empty:
        defib = revival_counts.index[0]
        awards.append({
            'icon': '🩺',
            'title': 'The Chat Defibrillator',
            'winner': defib,
            'subtitle': f'Resuscitated dead chat {revival_counts.iloc[0]:,} times'
        })

    # 6. Emoji Enthusiast
    emoji_king = max(user_stats.items(), key=lambda x: x[1]['emojis'])[0] if user_stats else "N/A"
    if user_stats[emoji_king]['emojis'] > 0:
        awards.append({
            'icon': '😂',
            'title': 'The Emoji Maestro',
            'winner': emoji_king,
            'subtitle': f'{user_stats[emoji_king]["emojis"]:,} emojis dropped in chat'
        })

    # Group Vibe
    if total_messages > 1000:
        group_vibe = "🌪️ Highly Chaotic & Ultra Active"
    elif busiest_day in ['Saturday', 'Sunday']:
        group_vibe = "🎉 Weekend Party & Chill Crew"
    else:
        group_vibe = "☕ Warm, Steady & Everyday Banter"

    metrics = {
        'total_messages': total_messages,
        'total_words': total_words,
        'total_media': total_media,
        'timespan': f"{start_date} - {end_date}",
        'busiest_day': busiest_day,
        'busiest_hour': busiest_hour,
        'group_vibe': group_vibe,
        'total_members': len(active_users)
    }

    # 3. Generate PNG Image Card with Pillow
    image_bytes = _render_wrapped_card(group_title, metrics, awards)

    insights = (
        f"🎁 **WhatsApp Group Wrapped Recap**: From {start_date} to {end_date}, your group exchanged "
        f"**{total_messages:,} messages** ({total_words:,} words). The overall vibe is **{group_vibe}**. "
        f"👑 **{main_char}** leads as The Main Character."
    )

    return {
        'awards': awards,
        'metrics': metrics,
        'vibe': group_vibe,
        'image_bytes': image_bytes,
        'insights': insights
    }


def _render_wrapped_card(group_title, metrics, awards):
    """
    Renders an 800x1100 high-aesthetic dark card using Pillow with gradient backgrounds,
    glowing borders, metric chips, and top awards.
    """
    width, height = 800, 1150
    # Create base dark image
    img = Image.new('RGB', (width, height), color='#0B1120')
    draw = ImageDraw.Draw(img)

    # Draw gradient background effect
    for y in range(height):
        # Vertical gradient from #0b1120 to #1e1b4b
        r = int(11 + (y / height) * 20)
        g = int(17 + (y / height) * 10)
        b = int(32 + (y / height) * 45)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Outer decorative glow border
    draw.rounded_rectangle([(20, 20), (width - 20, height - 20)], radius=24, outline='#38BDF8', width=2)
    draw.rounded_rectangle([(24, 24), (width - 24, height - 24)], radius=20, outline='#1E293B', width=1)

    # Header fonts
    try:
        font_huge = ImageFont.truetype("arial.ttf", 36)
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_sub = ImageFont.truetype("arial.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font_huge = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Header Text
    draw.text((width // 2, 60), "WHATSAPP GROUP WRAPPED", fill='#38BDF8', font=font_title, anchor='mm')
    draw.text((width // 2, 105), f"✨ {group_title} ✨", fill='#FFFFFF', font=font_huge, anchor='mm')
    draw.text((width // 2, 145), f"Period: {metrics.get('timespan', '')}  •  Vibe: {metrics.get('group_vibe', '')}", fill='#94A3B8', font=font_sub, anchor='mm')

    # Draw Separator
    draw.line([(60, 175), (width - 60, 175)], fill='#334155', width=1)

    # Top Metrics Grid (4 Boxes)
    box_w, box_h = 160, 90
    box_y = 195
    coords = [
        (50, box_y, "Total Messages", f"{metrics.get('total_messages', 0):,}"),
        (230, box_y, "Total Words", f"{metrics.get('total_words', 0):,}"),
        (410, box_y, "Media Shared", f"{metrics.get('total_media', 0):,}"),
        (590, box_y, "Peak Hour", f"{metrics.get('busiest_hour', 'N/A')}")
    ]

    for (bx, by, label, val) in coords:
        draw.rounded_rectangle([(bx, by), (bx + box_w, by + box_h)], radius=12, fill='#1E293B', outline='#475569', width=1)
        draw.text((bx + box_w // 2, by + 28), val, fill='#38BDF8', font=font_bold, anchor='mm')
        draw.text((bx + box_w // 2, by + 62), label, fill='#94A3B8', font=font_small, anchor='mm')

    # Section Title: Hall of Fame / Superlatives
    draw.text((width // 2, 325), "🏆 GROUP HALL OF FAME & SUPERLATIVES", fill='#F59E0B', font=font_bold, anchor='mm')
    draw.line([(width // 2 - 180, 345), (width // 2 + 180, 345)], fill='#D97706', width=2)

    # Draw Award Cards
    start_y = 365
    card_h = 100
    card_w = width - 100

    for idx, award in enumerate(awards[:6]):
        curr_y = start_y + idx * (card_h + 16)
        # Background card
        draw.rounded_rectangle([(50, curr_y), (50 + card_w, curr_y + card_h)], radius=14, fill='#0F172A', outline='#334155', width=1)

        # Left accent stripe
        draw.rounded_rectangle([(50, curr_y), (58, curr_y + card_h)], radius=4, fill='#38BDF8')

        # Award Title & Winner
        draw.text((75, curr_y + 25), f"{award.get('icon', '🎖️')}  {award.get('title', '')}", fill='#F8FAFC', font=font_bold)
        draw.text((75, curr_y + 55), f"Winner: {award.get('winner', '')}", fill='#38BDF8', font=font_bold)
        draw.text((75, curr_y + 80), award.get('subtitle', ''), fill='#94A3B8', font=font_small)

    # Footer
    draw.line([(60, height - 65), (width - 60, height - 65)], fill='#334155', width=1)
    draw.text((width // 2, height - 40), "Generated with WhatsApp Group Intelligence Engine 🚀", fill='#64748B', font=font_small, anchor='mm')

    # Save to buffer
    buf = io.BytesIO()
    img.save(buf, format='PNG', quality=95)
    buf.seek(0)
    return buf.getvalue()
