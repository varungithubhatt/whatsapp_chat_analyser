# WhatsApp Chat Analyzer

A Python + Streamlit project for analyzing exported WhatsApp chats. Upload a chat log, parse the messages, and explore activity patterns, engagement, word usage, and emoji trends in a lightweight dashboard.

## Features

- Parse WhatsApp exports from Android and iOS formats
- Clean raw chat text into a structured DataFrame
- Show overall summary metrics:
  - total messages
  - total words
  - media messages
  - shared links
- Identify the busiest users and their contribution percentages
- Generate word clouds and top word frequency tables
- Analyze emoji activity
- View monthly, daily, and weekday trends
- Display a week/day activity heatmap
- Launch a simple Streamlit web dashboard

## Project structure

```text
whatsapp_chat_analyser/
├── app.py
├── helper.py
├── preprocessor.py
├── requirements.txt
├── README.md
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
└── .gitignore
```

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

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

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

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Exporting a WhatsApp chat

To analyze a conversation, export it from WhatsApp without media:

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

## GitHub-ready notes

This repo is structured as a simple public project starter. To publish it on GitHub, add your repository name, update the clone URL in the instructions, and optionally add screenshots or a demo GIF in a "Demo" section.

## License

This project is available under the MIT License.
