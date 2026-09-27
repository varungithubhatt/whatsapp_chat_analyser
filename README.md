# WhatsApp Chat Analyzer

A Python + Streamlit project that turns exported WhatsApp chats into interactive insights. It parses raw chat text, structures it into a usable DataFrame, and then visualizes activity, participation, sentiment, network behavior, and personality-style patterns across a group.

## Why this project

This project is designed for:

- analyzing personal or group chat history
- understanding who is most active and dominant
- spotting conversation trends over time
- detecting emotional tone and tension in messages
- mapping social interaction patterns and subgroups
- building a rich dashboard from exported WhatsApp logs

---

## Core features

### 1. Chat parsing and data prep
- Parses WhatsApp exports from Android and iOS formats
- Handles both 12-hour and 24-hour timestamps
- Normalizes messy text exports into structured columns like user, message, date, hour, and day
- Filters out media-omitted notifications and invalid rows

### 2. Dashboard analytics
- Total messages, total words, media count, and links shared
- Most active users with contribution percentages
- Monthly and daily message timelines
- Day-of-week activity breakdown
- Monthly activity drilldown
- Weekly activity heatmap by day and hour
- Word cloud generation
- Frequent word analysis
- Emoji analysis and counts

### 3. Sentiment and mood intelligence
- VADER-based sentiment scoring
- Positive, neutral, and negative sentiment split
- Average sentiment score per chat or user
- Mood badge generation such as highly positive, calm, tense, or negative
- Monthly sentiment trajectory
- User sentiment leaderboard
- Highlighting the most positive and negative messages

### 4. Dominance and influence metrics
- Weighted dominance score based on message volume, word count, initiation activity, reply triggers, and sharing behavior
- Top leaders, initiators, and reply magnets
- Group dominance ranking by user
- Leadership signals for who drives conversation momentum

### 5. Topic modeling and alignment
- TF-IDF-based user similarity analysis
- Pairwise conversation alignment score between participants
- Topic extraction using Latent Dirichlet Allocation (LDA)
- Top keywords for each discovered discussion theme
- Topic champion detection
- User-topic distribution matrix

### 6. Relationship dynamics and network analysis
- Directed interaction tracking between users
- Reply gap calculation and response latency
- Strongest pairwise interaction bonds
- Ghost score / left-on-read rate
- Centrality metrics such as popularity, social glue, and bridge influence
- Community detection and subgroup discovery
- Interactive network graph visualizations with PyVis

### 7. Communication style and personality badges
- Essayist vs. rapid texter detection
- Emoji enthusiast analysis
- Question-heavy chatters
- Energetic or expressive message patterns
- Night owl and early bird behavior
- Media-heavy participants
- Balanced conversational personalities

### 8. Chaos, drama, and toxicity indicators
- Chaos score calculation based on caps, exclamations, late-night behavior, multi-texting, and rapid bursts
- Drama detection for heated periods and tense spikes
- Toxicity and vulgarity analysis using profanity and heuristic detection
- Spice rating by user
- Timeline of toxicity and vulgarity trends

### 9. Group wrapped recap
- Spotify-style wrapped summary for the group
- Top superlatives such as main character, novelist, night owl, and emoji maestro
- Group vibe summary and chat milestones
- Downloadable wrapped-style image card rendered with Pillow

---

## Advanced module overview

The project includes a dedicated advanced package with specialized modules:

```text
advanced/
├── sentiment.py       # VADER-based sentiment, mood badges, mood swing, highlight extraction
├── dominance.py       # Weighted dominance scoring and leadership insights
├── topics.py          # TF-IDF alignment, LDA topics, topic champions
├── dynamics.py        # Social graph, reply networks, ghost rates, centrality, community graph HTML
├── style.py           # Personality and communication style badges
├── chaos.py           # Chaos score for noisy and hyperactive users
├── toxicity.py        # Vulgarity and toxicity estimation
├── drama.py           # Drama spike detection and tension windows
├── ghost.py           # Ghost score and revival / silence-breaker metrics
├── cliques.py         # Clique and subgroup detection based on interaction structure
├── wrapped.py         # Group wrapped card and superlative awards
└── __init__.py
```

### Module-specific capabilities

- `advanced/sentiment.py`
  - sentiment breakdown by user and month
  - mood badge and swing index
  - positive and negative highlight extraction

- `advanced/dominance.py`
  - dominance leaderboard
  - top initiator, top magnet, top leader
  - weighted conversation influence model

- `advanced/topics.py`
  - alignment matrix between participants
  - top topic keywords
  - dominant topic assignment per message
  - user-topic distribution

- `advanced/dynamics.py`
  - reply graph analysis
  - strongest interaction pairs
  - ghost rate and centrality metrics
  - network map HTML export

- `advanced/style.py`
  - essayist, rapid texter, emoji fan, night owl, early bird, media mogul badges
  - communication profile summaries for each user

- `advanced/chaos.py`
  - chaos score ranking
  - all-caps, exclamation, late-night, multi-text, and burst-based behavior analysis

- `advanced/toxicity.py`
  - obscene and profane content detection
  - toxicity index by user and month
  - spicy / saintly group labeling

- `advanced/drama.py`
  - tense debate and conflict detection across time windows
  - sample snippets from heated conversations
  - group drama level classification

- `advanced/ghost.py`
  - left-on-read rate
  - average response wait time
  - revival and silence-breaking detection

- `advanced/cliques.py`
  - subgroup discovery
  - communication cohesion within cliques
  - clique leader detection

- `advanced/wrapped.py`
  - group awards and superlatives
  - wrapped summary metrics
  - generated PNG card for sharing

---

## Project structure

```text
whatsapp_chat_analyser/
├── app.py
├── helper.py
├── preprocessor.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── advanced/
│   ├── __init__.py
│   ├── chaos.py
│   ├── cliques.py
│   ├── dominance.py
│   ├── drama.py
│   ├── dynamics.py
│   ├── ghost.py
│   ├── sentiment.py
│   ├── style.py
│   ├── topics.py
│   ├── toxicity.py
│   └── wrapped.py
└── __pycache__/
```

---

## Tech stack

- Python 3
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
- scikit-learn
- vaderSentiment
- NetworkX
- PyVis
- Pillow
- better-profanity

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/whatsapp_chat_analyser.git
cd whatsapp_chat_analyser
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, typically:

```text
http://localhost:8501
```

---

## Exporting a WhatsApp chat

To analyze a chat, export it from WhatsApp without media:

### Android
1. Open the chat
2. Tap the three-dot menu
3. Select Export chat
4. Choose Without media
5. Save the .txt file

### iPhone
1. Open the chat
2. Tap the contact or group name
3. Select Export Chat
4. Choose Without Media
5. Save the .txt file

---

## Notes

This project combines basic analytics with a larger experimental feature set. Some modules are focused on group behavior analysis, conversational intelligence, and social network understanding, while the user-facing dashboard provides a simplified front-end for quick insights.

---

## License

This project is available under the MIT License.
