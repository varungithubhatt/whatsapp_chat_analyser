import re
import unicodedata
from collections import Counter
import pandas as pd
from wordcloud import WordCloud

# Standard English stop words and WhatsApp specific terms to filter out
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 'before',
    'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'could',
    'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn',
    "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have',
    'haven', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here',
    "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i',
    "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', 'isn', "isn't", 'it',
    "it's", 'its', 'itself', 'let', "let's", 'me', 'more', 'most', 'mustn',
    "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
    'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over',
    'own', 'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's", 'should',
    'shouldn', "shouldn't", 'so', 'some', 'such', 't', 'than', 'that', "that's",
    'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's",
    'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those',
    'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn',
    "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', 'weren', "weren't",
    'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while',
    'who', "who's", 'whom', 'why', "why's", 'with', 'won', "won't", 'would',
    'wouldn', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your',
    'yours', 'yourself', 'yourselves', 'media', 'omitted', 'message', 'deleted',
    'pm', 'am', 'ok', 'okay', 'yeah', 'yes', 'null', 'image', 'video', 'audio',
    'sticker', 'gif', 'document'
}


def is_emoji(char):
    """Checks if a given character is an emoji."""
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


def fetch_stats(selected_user, df):
    """Returns total messages, total words, media messages count, and links shared count."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    media_pattern = r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted|Contact card omitted|sticker omitted'
    num_media_messages = df[df['message'].str.contains(media_pattern, case=False, na=False, regex=True)].shape[0]

    url_pattern = r'https?://\S+|www\.\S+'
    links = []
    for message in df['message']:
        links.extend(re.findall(url_pattern, message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    """Returns the top 5 most active users and percentage contribution of all users."""
    df_filtered = df[df['user'] != 'group_notification']
    user_counts = df_filtered['user'].value_counts()
    top_5 = user_counts.head()

    percent_df = round((user_counts / df_filtered.shape[0]) * 100, 2).reset_index()
    percent_df.columns = ['User', 'Percentage']

    return top_5, percent_df


def create_wordcloud(selected_user, df):
    """Generates and returns a WordCloud object for the given user."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df_filtered = df[df['user'] != 'group_notification']
    media_pattern = r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted|Contact card omitted|sticker omitted'
    df_filtered = df_filtered[~df_filtered['message'].str.contains(media_pattern, case=False, na=False, regex=True)]

    cleaned_words = []
    for message in df_filtered['message']:
        for word in message.lower().split():
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word not in STOP_WORDS:
                cleaned_words.append(clean_word)

    text = " ".join(cleaned_words)
    if not text.strip():
        text = "NoWordsAvailable"

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(text)
    return df_wc


def most_common_words(selected_user, df):
    """Returns a DataFrame containing the top 20 most frequent words."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df_filtered = df[df['user'] != 'group_notification']
    media_pattern = r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted|Contact card omitted|sticker omitted'
    df_filtered = df_filtered[~df_filtered['message'].str.contains(media_pattern, case=False, na=False, regex=True)]

    words = []
    for message in df_filtered['message']:
        for word in message.lower().split():
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word and clean_word not in STOP_WORDS and not clean_word.isdigit():
                words.append(clean_word)

    if not words:
        return pd.DataFrame(columns=['Word', 'Frequency'])

    most_common_df = pd.DataFrame(Counter(words).most_common(20), columns=['Word', 'Frequency'])
    return most_common_df


def emoji_helper(selected_user, df):
    """Returns a DataFrame containing emoji counts."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if is_emoji(c)])

    if not emojis:
        return pd.DataFrame(columns=['Emoji', 'Count'])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))), columns=['Emoji', 'Count'])
    return emoji_df


def monthly_timeline(selected_user, df):
    """Returns monthly timeline dataframe with total messages per month."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month'])['message'].count().reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline


def daily_timeline(selected_user, df):
    """Returns daily timeline dataframe with total messages per day."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily = df.groupby('only_date')['message'].count().reset_index()
    daily.rename(columns={'only_date': 'date', 'message': 'count'}, inplace=True)
    return daily


def week_activity_map(selected_user, df):
    """Returns message counts grouped by day of the week."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_name'].value_counts()
    return day_counts.reindex(days_order).fillna(0).astype(int)


def month_activity_map(selected_user, df):
    """Returns message counts grouped by month name."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    months_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    month_counts = df['month'].value_counts()
    return month_counts.reindex(months_order).fillna(0).astype(int)


def activity_heatmap(selected_user, df):
    """Returns a pivot table representing weekly activity heatmap (day vs period)."""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(
        index='day_name',
        columns='period',
        values='message',
        aggfunc='count'
    ).fillna(0)

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    user_heatmap = user_heatmap.reindex([d for d in days_order if d in user_heatmap.index])
    return user_heatmap
