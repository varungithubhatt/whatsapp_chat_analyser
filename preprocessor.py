import re
import pandas as pd


def preprocess(data):
    """
    Preprocesses raw WhatsApp chat export data into a structured Pandas DataFrame.
    Supports iOS and Android export formats in 12-hour (AM/PM) and 24-hour time notations.
    """
    # Normalize unicode spaces and line breaks
    data = data.replace(' ', ' ').replace('\xa0', ' ').replace('\r', '')

    # Regex patterns for various WhatsApp date/time formats:
    # 1. iOS 12-hr / 24-hr format with brackets: [dd/mm/yy, hh:mm:ss AM/PM] or [dd/mm/yyyy, ...]
    ios_pattern = r'\[(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4},\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s+[APap][Mm])?)\]\s*'
    # 2. Android 12-hr / 24-hr format with hyphen: dd/mm/yy, hh:mm am - or dd/mm/yyyy, hh:mm -
    android_pattern = r'(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4},\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s+[APap][Mm])?)\s*-\s*'

    patterns = [ios_pattern, android_pattern]

    matched_pattern = None
    for p in patterns:
        if len(re.findall(p, data)) > 0:
            matched_pattern = p
            break

    if matched_pattern is None:
        matched_pattern = ios_pattern

    split_result = re.split(matched_pattern, data)

    if len(split_result) > 1:
        dates = split_result[1::2]
        user_messages = split_result[2::2]
    else:
        dates = []
        user_messages = []

    dates = [d.strip('[] ') for d in dates]

    df = pd.DataFrame({
        'user_messages': user_messages,
        'messages_date': dates
    })

    df['user_messages'] = (
        df['user_messages']
        .str.replace(r'^-\s*', '', regex=True)
        .str.strip()
    )

    users = []
    messages = []

    for message in df['user_messages']:
        # Match user name up to the colon (supports phone numbers with '+', emojis, names with symbols)
        entry = re.split(r'^([^:]+?):\s', message, maxsplit=1)
        if len(entry) > 2:
            users.append(entry[1].strip())
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_messages'], inplace=True)

    # Convert messages_date to datetime with flexible fallback
    try:
        df['messages_date'] = pd.to_datetime(df['messages_date'], format='mixed', dayfirst=True)
    except Exception:
        try:
            df['messages_date'] = pd.to_datetime(df['messages_date'], dayfirst=True)
        except Exception:
            df['messages_date'] = pd.to_datetime(df['messages_date'], errors='coerce')

    df.dropna(subset=['messages_date'], inplace=True)

    df['year'] = df['messages_date'].dt.year
    df['month_num'] = df['messages_date'].dt.month
    df['month'] = df['messages_date'].dt.month_name()
    df['day'] = df['messages_date'].dt.day
    df['day_name'] = df['messages_date'].dt.day_name()
    df['hour'] = df['messages_date'].dt.hour
    df['minute'] = df['messages_date'].dt.minute
    df['only_date'] = df['messages_date'].dt.date

    period = []
    for hour in df['hour']:
        if hour == 23:
            period.append('23-00')
        elif hour == 0:
            period.append('00-01')
        else:
            period.append(f'{hour:02d}-{(hour + 1):02d}')

    df['period'] = period

    return df


# Alias for backward compatibility
preprocessor = preprocess
