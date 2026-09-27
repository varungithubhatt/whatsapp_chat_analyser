import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


# Default custom stop words list for chat analysis
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'ok', 'okay', 'yeah', 'yes',
    'no', 'like', 'get', 'got', 'go', 'going', 'know', 'see', 'come', 'one', 'also', 'u', 'ur', 'r', 'k', 'bro',
    'dude', 'bhai', 'hai', 'ki', 'ko', 'se', 'ka', 'ke', 'aur', 'kya', 'toh', 'bhi', 'ho', 'tha', 'thi', 'kar',
    'media', 'omitted', 'null', 'nan', 'image', 'video', 'sticker', 'audio', 'document'
}


def _clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<media omitted>|image omitted|video omitted|sticker omitted', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [w for w in text.split() if len(w) > 2 and w not in STOP_WORDS and not w.isdigit()]
    return " ".join(tokens)


def analyze_topics(df, n_topics=4, top_words_count=6):
    """
    Analyzes Interest Alignment and extracts Conversation Topics:
    1. Per-User TF-IDF Vectorization & Pairwise Cosine Similarity Matrix.
    2. Most aligned pairs ("On the same wavelength").
    3. Latent Dirichlet Allocation (LDA) Topic Extraction.
    4. Per-User Topic Distribution & Topic Champions.
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'topics': [],
            'similarity_matrix': pd.DataFrame(),
            'top_pairs': [],
            'user_topic_dist': pd.DataFrame(),
            'insights': "Not enough message data for topic modeling."
        }

    df_clean['cleaned_message'] = df_clean['message'].apply(_clean_text)
    meaningful_df = df_clean[df_clean['cleaned_message'].str.len() > 3].copy()

    users = df_clean['user'].value_counts()
    active_users = users[users >= 2].index.tolist()

    # 1. User Interest Alignment via TF-IDF & Cosine Similarity
    user_docs = {}
    for user in active_users:
        user_text = " ".join(meaningful_df[meaningful_df['user'] == user]['cleaned_message'])
        if len(user_text.strip()) > 0:
            user_docs[user] = user_text

    valid_users = list(user_docs.keys())

    if len(valid_users) >= 2:
        tfidf = TfidfVectorizer(max_features=500, stop_words=list(STOP_WORDS))
        user_tfidf_matrix = tfidf.fit_transform([user_docs[u] for u in valid_users])
        sim_matrix_vals = cosine_similarity(user_tfidf_matrix)

        similarity_matrix = pd.DataFrame(
            sim_matrix_vals,
            index=valid_users,
            columns=valid_users
        ).round(3)

        # Extract top pairs
        pairs = []
        for i in range(len(valid_users)):
            for j in range(i + 1, len(valid_users)):
                score = round(float(sim_matrix_vals[i][j]) * 100, 1)
                pairs.append({
                    'User 1': valid_users[i],
                    'User 2': valid_users[j],
                    'Alignment %': score,
                    'Pair': f"{valid_users[i]} & {valid_users[j]}"
                })
        top_pairs = sorted(pairs, key=lambda x: x['Alignment %'], reverse=True)
    else:
        similarity_matrix = pd.DataFrame()
        top_pairs = []

    # 2. Topic Modeling via Latent Dirichlet Allocation (LDA)
    all_texts = meaningful_df['cleaned_message'].tolist()

    topics = []
    user_topic_dist = pd.DataFrame()

    if len(all_texts) >= 10:
        actual_topics = min(n_topics, max(2, len(all_texts) // 15))
        vectorizer = CountVectorizer(max_df=0.85, min_df=2, max_features=1000, stop_words=list(STOP_WORDS))
        try:
            dtm = vectorizer.fit_transform(all_texts)
            feature_names = vectorizer.get_feature_names_out()

            lda = LatentDirichletAllocation(n_components=actual_topics, random_state=42, max_iter=15)
            lda.fit(dtm)

            # Assign topic to each message in meaningful_df
            topic_distribution = lda.transform(dtm)
            meaningful_df['dominant_topic'] = topic_distribution.argmax(axis=1)

            # Top keywords per topic
            for topic_idx, topic_weights in enumerate(lda.components_):
                top_indices = topic_weights.argsort()[:-top_words_count - 1:-1]
                top_keywords = [feature_names[i] for i in top_indices]
                topic_title = f"Topic {topic_idx + 1}: " + ", ".join(top_keywords[:3]).title()

                # Find Topic Champion (User with most messages in this topic)
                topic_msgs = meaningful_df[meaningful_df['dominant_topic'] == topic_idx]
                topic_vol = len(topic_msgs)
                topic_pct = round((topic_vol / len(meaningful_df)) * 100, 1) if len(meaningful_df) > 0 else 0

                if not topic_msgs.empty:
                    champion = topic_msgs['user'].value_counts().index[0]
                    champ_msgs = topic_msgs['user'].value_counts().iloc[0]
                else:
                    champion = "N/A"
                    champ_msgs = 0

                topics.append({
                    'id': topic_idx + 1,
                    'title': topic_title,
                    'keywords': top_keywords,
                    'volume': topic_vol,
                    'percentage': topic_pct,
                    'champion': champion,
                    'champion_messages': champ_msgs
                })

            # User vs Topic Distribution Matrix
            user_topic_counts = meaningful_df.groupby(['user', 'dominant_topic']).size().unstack(fill_value=0)
            user_topic_percentages = user_topic_counts.div(user_topic_counts.sum(axis=1), axis=0) * 100.0
            user_topic_percentages.columns = [f"Topic {i+1}" for i in user_topic_percentages.columns]
            user_topic_dist = user_topic_percentages.round(1).reset_index()

        except Exception:
            # Fallback if vocabulary is too sparse for LDA
            topics = []
            user_topic_dist = pd.DataFrame()

    # Insight summary
    if top_pairs:
        best_pair = top_pairs[0]
        insights = (
            f"🤝 **{best_pair['User 1']}** and **{best_pair['User 2']}** have the highest conversational alignment "
            f"({best_pair['Alignment %']}% vocabulary & interest overlap)."
        )
    else:
        insights = "Conversations show diverse and distributed interests across participants."

    return {
        'topics': topics,
        'similarity_matrix': similarity_matrix,
        'top_pairs': top_pairs,
        'user_topic_dist': user_topic_dist,
        'insights': insights
    }
