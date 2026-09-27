import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import preprocessor
import helper
import advanced

# Configure Streamlit page settings
st.set_page_config(
    page_title="WhatsApp Group Intelligence & Chat Analyser",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin-bottom: 10px;
        border: 1px solid #334155;
    }
    .badge-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #38bdf8;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .insight-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 12px 0;
        font-size: 15px;
    }
    .spice-card {
        background: linear-gradient(135deg, #311313 0%, #1a0b0b 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .wrapped-card {
        background: linear-gradient(135deg, #0b1120 0%, #1e1b4b 100%);
        border: 2px solid #818cf8;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# App Title & Description
st.title("🚀 WhatsApp Group Intelligence & Chat Analyser")
st.markdown(
    "Turn raw WhatsApp chat exports into actionable **Group Intelligence**: analyze mood & sentiment, "
    "group dominance, interest alignment, interaction social graphs, chaos levels, drama spikes, "
    "clique subgroups, and generate custom **Spotify-style Wrapped Cards**."
)

# Sidebar setup
st.sidebar.title("💬 WhatsApp Analyser")
uploaded_file = st.sidebar.file_uploader("Upload a WhatsApp Chat export file (.txt)", type=["txt"])

if uploaded_file is not None:
    try:
        bytes_data = uploaded_file.getvalue()
        data = bytes_data.decode("utf-8")
        df = preprocessor.preprocess(data)

        if df.empty:
            st.error("⚠️ Could not parse messages from the uploaded file. Please ensure it is a valid WhatsApp export (.txt).")
        else:
            # User selection list
            user_list = df['user'].unique().tolist()
            if 'group_notification' in user_list:
                user_list.remove('group_notification')
            user_list.sort()
            user_list.insert(0, "Overall")

            selected_user = st.sidebar.selectbox("Filter analysis with respect to:", user_list)
            run_button = st.sidebar.button("✨ Show Full Intelligence Analysis", use_container_width=True)

            if run_button:
                # Create main navigation tabs
                (
                    tab_overview,
                    tab_sentiment,
                    tab_dominance,
                    tab_topics,
                    tab_dynamics,
                    tab_chaos_drama,
                    tab_style,
                    tab_wrapped
                ) = st.tabs([
                    "📊 Activity & Stats",
                    "🎭 Mood & Sentiment",
                    "👑 Dominance & Influence",
                    "🧠 Topics & Alignment",
                    "🕸️ Dynamics & Cliques",
                    "🌪️ Chaos & Drama",
                    "🎖️ Personality & Style",
                    "🎁 Group Wrapped"
                ])

                # =========================================================================
                # TAB 1: OVERVIEW & ACTIVITY STATS
                # =========================================================================
                with tab_overview:
                    st.header("📈 Overview Statistics")
                    num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("Total Messages", f"{num_messages:,}")
                    with c2:
                        st.metric("Total Words", f"{words:,}")
                    with c3:
                        st.metric("Media Shared", f"{num_media_messages:,}")
                    with c4:
                        st.metric("Links Shared", f"{num_links:,}")

                    st.divider()

                    # Timelines
                    st.subheader("⏳ Conversation Timelines")
                    t_col1, t_col2 = st.columns(2)

                    with t_col1:
                        st.write("**Monthly Message Timeline**")
                        timeline = helper.monthly_timeline(selected_user, df)
                        if not timeline.empty:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(timeline['time'], timeline['message'], color='#38BDF8', marker='o', linewidth=2)
                            plt.xticks(rotation=45, ha='right')
                            plt.ylabel("Messages")
                            plt.grid(True, linestyle='--', alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("No monthly timeline data.")

                    with t_col2:
                        st.write("**Daily Activity Trend**")
                        daily = helper.daily_timeline(selected_user, df)
                        if not daily.empty:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.plot(daily['date'], daily['count'], color='#A855F7', linewidth=1.5)
                            plt.xticks(rotation=45, ha='right')
                            plt.ylabel("Messages")
                            plt.grid(True, linestyle='--', alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("No daily timeline data.")

                    st.divider()

                    # Activity Maps
                    st.subheader("🗓️ Activity Heatmaps & Distribution")
                    act_col1, act_col2 = st.columns(2)

                    with act_col1:
                        st.write("**Busiest Days of Week**")
                        busy_day = helper.week_activity_map(selected_user, df)
                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        ax.bar(busy_day.index, busy_day.values, color='#3B82F6', edgecolor='#1D4ED8')
                        plt.xticks(rotation=45, ha='right')
                        plt.ylabel("Messages")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    with act_col2:
                        st.write("**Busiest Months of Year**")
                        busy_month = helper.month_activity_map(selected_user, df)
                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        ax.bar(busy_month.index, busy_month.values, color='#F59E0B', edgecolor='#D97706')
                        plt.xticks(rotation=45, ha='right')
                        plt.ylabel("Messages")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    # Hourly Heatmap
                    st.write("**Weekly Activity Heatmap (Day vs Hour Window)**")
                    user_heatmap = helper.activity_heatmap(selected_user, df)
                    if not user_heatmap.empty and user_heatmap.shape[1] > 0:
                        fig, ax = plt.subplots(figsize=(12, 5))
                        sns.heatmap(user_heatmap, ax=ax, cmap="YlGnBu", annot=False, cbar=True)
                        plt.xlabel("Hourly Window")
                        plt.ylabel("Day of Week")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    st.divider()

                    # Group Top Members (if Overall)
                    if selected_user == 'Overall':
                        st.subheader("👥 Most Active Participants")
                        top_users, percent_df = helper.most_busy_users(df)
                        if not top_users.empty:
                            u_col1, u_col2 = st.columns(2)
                            with u_col1:
                                fig, ax = plt.subplots(figsize=(7, 4))
                                ax.bar(top_users.index, top_users.values, color='#EF4444', edgecolor='#B91C1C')
                                plt.xticks(rotation=45, ha='right')
                                plt.ylabel("Messages Sent")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)
                            with u_col2:
                                st.dataframe(percent_df, use_container_width=True)

                    st.divider()

                    # Word Cloud & Common Words
                    wc_col, cw_col = st.columns(2)
                    with wc_col:
                        st.subheader("☁️ Vocabulary Word Cloud")
                        try:
                            df_wc = helper.create_wordcloud(selected_user, df)
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.imshow(df_wc, interpolation='bilinear')
                            ax.axis('off')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            st.info(f"Could not render word cloud: {e}")

                    with cw_col:
                        st.subheader("🔤 Top 20 Common Words")
                        most_common_df = helper.most_common_words(selected_user, df)
                        if not most_common_df.empty:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.barh(most_common_df['Word'], most_common_df['Frequency'], color='#10B981')
                            plt.gca().invert_yaxis()
                            plt.xlabel("Occurrences")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info("No common words available.")

                    # Emoji Breakdown
                    st.subheader("😀 Emoji Frequency")
                    emoji_df = helper.emoji_helper(selected_user, df)
                    if not emoji_df.empty:
                        e1, e2 = st.columns(2)
                        with e1:
                            st.dataframe(emoji_df.head(15), use_container_width=True)
                        with e2:
                            top_5_emojis = emoji_df.head(5)
                            fig, ax = plt.subplots(figsize=(5, 5))
                            ax.pie(
                                top_5_emojis['Count'],
                                labels=top_5_emojis['Emoji'],
                                autopct='%0.1f%%',
                                startangle=140
                            )
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                # =========================================================================
                # TAB 2: SENTIMENT & MOOD ANALYSIS (PHASE 1)
                # =========================================================================
                with tab_sentiment:
                    st.header("🎭 Mood & Sentiment Intelligence")
                    sent_data = advanced.analyze_sentiment(selected_user, df)

                    st.markdown(f"""
                    <div class="insight-box">
                        <b>Current Mood Pulse:</b> {sent_data['mood_badge']}<br>
                        <b>Average Compound Sentiment:</b> {sent_data['avg_score']} &nbsp;|&nbsp;
                        <b>Mood Volatility (Swing Index):</b> {sent_data['mood_swing_index']}
                    </div>
                    """, unsafe_allow_html=True)

                    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
                    with s_col1:
                        st.metric("Positive Messages", f"{sent_data['pos_pct']}%", delta=f"{sent_data['pos_pct']}%")
                    with s_col2:
                        st.metric("Neutral Messages", f"{sent_data['neu_pct']}%")
                    with s_col3:
                        st.metric("Negative Messages", f"{sent_data['neg_pct']}%", delta=f"-{sent_data['neg_pct']}%", delta_color="inverse")
                    with s_col4:
                        st.metric("Mood Swing Index", f"{sent_data['mood_swing_index']}")

                    st.divider()

                    # Sentiment timeline
                    st.subheader("📈 Sentiment Evolution Over Time")
                    s_timeline = sent_data['timeline']
                    if not s_timeline.empty:
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(s_timeline['time_period'], s_timeline['avg_sentiment'], marker='o', color='#10B981', linewidth=2, label='Avg Sentiment')
                        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                        plt.xticks(rotation=45, ha='right')
                        plt.ylabel("Sentiment Score (-1 to +1)")
                        plt.title("Monthly Average Sentiment Trajectory")
                        plt.grid(True, linestyle=':', alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                    st.divider()

                    # User Sentiment Leaderboard
                    st.subheader("🏆 Group Participant Sentiment Leaderboard")
                    u_lead = sent_data['user_leaderboard']
                    if not u_lead.empty:
                        st.dataframe(u_lead, use_container_width=True)

                    # Highlights
                    st.divider()
                    st.subheader("🌟 Top Sentiment Highlights")
                    hl_col1, hl_col2 = st.columns(2)

                    with hl_col1:
                        st.write("##### 💖 Most Positive Moments")
                        if sent_data['pos_highlights']:
                            for h in sent_data['pos_highlights']:
                                st.success(f"**{h['user']}** ({h['date']}) [Score: +{h['score']}]:\n\n\"{h['message']}\"")
                        else:
                            st.info("No strong positive highlights found.")

                    with hl_col2:
                        st.write("##### ⚠️ Critical / Somber Moments")
                        if sent_data['neg_highlights']:
                            for h in sent_data['neg_highlights']:
                                st.error(f"**{h['user']}** ({h['date']}) [Score: {h['score']}]:\n\n\"{h['message']}\"")
                        else:
                            st.info("No strong negative moments found.")

                # =========================================================================
                # TAB 3: DOMINANCE & INFLUENCE (PHASE 2)
                # =========================================================================
                with tab_dominance:
                    st.header("👑 Group Dominance & Influence Analysis")
                    dom_data = advanced.analyze_dominance(df)

                    st.markdown(f"""
                    <div class="insight-box">
                        {dom_data['insights']}
                    </div>
                    """, unsafe_allow_html=True)

                    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                    with d_col1:
                        st.metric("Group Leader 👑", str(dom_data['top_leader']))
                    with d_col2:
                        st.metric("Top Initiator 🚀", str(dom_data['top_initiator']))
                    with d_col3:
                        st.metric("Top Reply Magnet 🧲", str(dom_data['top_magnet']))
                    with d_col4:
                        st.metric("Total Chat Sessions", f"{dom_data['total_sessions']:,}")

                    st.divider()

                    # Leaderboard Chart & Table
                    leaderboard = dom_data['leaderboard']
                    if not leaderboard.empty:
                        l_col1, l_col2 = st.columns([1, 1])

                        with l_col1:
                            st.subheader("Dominance Score Ranking (0-100)")
                            fig, ax = plt.subplots(figsize=(7, 4.5))
                            ax.barh(leaderboard['User'], leaderboard['Dominance Score'], color='#F59E0B', edgecolor='#D97706')
                            plt.gca().invert_yaxis()
                            plt.xlabel("Weighted Dominance Index")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        with l_col2:
                            st.subheader("Influence Signals Breakdown")
                            st.dataframe(
                                leaderboard[['User', 'Dominance Score', 'Msg %', 'Word %', 'Initiation %', 'Magnet %', 'Share %']],
                                use_container_width=True
                            )

                # =========================================================================
                # TAB 4: TOPIC MODELING & INTEREST ALIGNMENT (PHASE 3)
                # =========================================================================
                with tab_topics:
                    st.header("🧠 Topics & Conversational Alignment")
                    topic_data = advanced.analyze_topics(df)

                    st.markdown(f"""
                    <div class="insight-box">
                        {topic_data['insights']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Similarity Heatmap & Top Pairs
                    st.subheader("🤝 Interest Alignment Matrix (Cosine Similarity)")
                    sim_matrix = topic_data['similarity_matrix']
                    if not sim_matrix.empty and len(sim_matrix) >= 2:
                        h_col1, h_col2 = st.columns([1.2, 0.8])
                        with h_col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(sim_matrix, ax=ax, cmap="mako", annot=True, fmt=".2f", cbar=True)
                            plt.title("Wavelength / Topic Overlap Heatmap")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        with h_col2:
                            st.write("**Top Aligned Pairs (Same Wavelength)**")
                            top_pairs_df = pd.DataFrame(topic_data['top_pairs'])
                            if not top_pairs_df.empty:
                                st.dataframe(top_pairs_df[['Pair', 'Alignment %']].head(8), use_container_width=True)
                    else:
                        st.info("Not enough distinct participants to generate an alignment matrix.")

                    st.divider()

                    # Extracted Topics
                    st.subheader("🏷️ Discovered Conversation Topics")
                    topics_list = topic_data['topics']
                    if topics_list:
                        t_cols = st.columns(min(len(topics_list), 4))
                        for idx, t in enumerate(topics_list[:4]):
                            with t_cols[idx]:
                                st.markdown(f"""
                                <div class="badge-card">
                                    <h4>{t['title']}</h4>
                                    <p><b>Keywords:</b><br>{", ".join(t['keywords'])}</p>
                                    <p><b>Volume:</b> {t['volume']} msgs ({t['percentage']}%)</p>
                                    <p>👑 <b>Topic Champion:</b><br><b>{t['champion']}</b> ({t['champion_messages']} msgs)</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No distinct topics discovered. More chat history needed.")

                    # User topic distribution
                    if not topic_data['user_topic_dist'].empty:
                        st.divider()
                        st.subheader("📊 Participant Topic Distribution (%)")
                        st.dataframe(topic_data['user_topic_dist'], use_container_width=True)

                # =========================================================================
                # TAB 5: RELATIONSHIP DYNAMICS & CLIQUES (PHASE 4 + FEATURE 3 & 4)
                # =========================================================================
                with tab_dynamics:
                    st.header("🕸️ Relationship Dynamics, Social Graph & Cliques")
                    dyn_data = advanced.analyze_dynamics(df)
                    clique_data = advanced.analyze_cliques(df)
                    ghost_data = advanced.analyze_ghost_and_revival(df)

                    st.markdown(f"""
                    <div class="insight-box">
                        <b>Graph Insight:</b> {dyn_data['insights']}<br>
                        <b>Clique Insight:</b> {clique_data['insights']}<br>
                        <b>Ghost/Reviver Insight:</b> {ghost_data['insights']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Subgroup / Clique Detection
                    st.subheader("👥 Detected Subgroups & Cliques (Louvain Community Detection)")
                    if clique_data['cliques']:
                        c_cols = st.columns(min(len(clique_data['cliques']), 3))
                        for idx, clq in enumerate(clique_data['cliques']):
                            col_t = c_cols[idx % 3]
                            with col_t:
                                st.markdown(f"""
                                <div class="badge-card">
                                    <h4>{clq['name']}</h4>
                                    <p><b>Members ({clq['member_count']}):</b><br>{", ".join(clq['members'])}</p>
                                    <p>👑 <b>Clique Core:</b> {clq['leader']}</p>
                                    <p>🔗 <b>Internal Cohesion:</b> {clq['cohesion_pct']}%</p>
                                </div>
                                """, unsafe_allow_html=True)

                        if not clique_data['clique_df'].empty:
                            st.dataframe(clique_data['clique_df'], use_container_width=True)

                    st.divider()

                    # Strongest Bonds & Ghost Scores
                    b_col1, b_col2 = st.columns(2)

                    with b_col1:
                        st.subheader("🔗 Strongest Interaction Bonds")
                        if dyn_data['top_bonds']:
                            st.dataframe(
                                pd.DataFrame(dyn_data['top_bonds'])[['Pair', 'Total Replies', 'Reciprocity %', 'Avg Reply Time (min)']],
                                use_container_width=True
                            )
                        else:
                            st.info("No direct pairwise interaction data.")

                    with b_col2:
                        st.subheader("👻 Ghost Rate (Left-on-Read) & Revivers")
                        if not ghost_data['ghost_leaderboard'].empty:
                            st.dataframe(ghost_data['ghost_leaderboard'][['User', 'Total Sent', 'Ghost Rate %', 'Avg Reply Wait (min)', 'Ghost Status']], use_container_width=True)

                    # Revivers Table
                    st.subheader("🩺 Chat Revivers & Silence Breakers (>2 hr silences)")
                    if not ghost_data['reviver_leaderboard'].empty:
                        st.dataframe(ghost_data['reviver_leaderboard'], use_container_width=True)

                    st.divider()

                    # Centrality Metrics
                    st.subheader("🎯 Network Centrality & Social Roles")
                    if not dyn_data['centrality_table'].empty:
                        st.dataframe(dyn_data['centrality_table'], use_container_width=True)

                # =========================================================================
                # TAB 6: CHAOS & DRAMA (ROADMAP V2 FEATURES 1, 2, 5)
                # =========================================================================
                with tab_chaos_drama:
                    st.header("🌪️ Chaos Score, Drama Detector & Spice Radar")

                    chaos_data = advanced.analyze_chaos(df)
                    drama_data = advanced.detect_drama(df)
                    tox_data = advanced.analyze_toxicity(df)

                    st.markdown(f"""
                    <div class="insight-box">
                        <b>Chaos Insight:</b> {chaos_data['insights']}<br>
                        <b>Drama Pulse:</b> {drama_data['insights']}<br>
                        <b>Spice Level:</b> {tox_data['insights']}
                    </div>
                    """, unsafe_allow_html=True)

                    cd_col1, cd_col2, cd_col3, cd_col4 = st.columns(4)
                    with cd_col1:
                        st.metric("Chaos Champion 🌪️", str(chaos_data['chaos_champion']))
                    with cd_col2:
                        st.metric("Top Chaos Score", f"{chaos_data['top_chaos_score']}/100")
                    with cd_col3:
                        st.metric("Spiciest Texter 🌶️", str(tox_data['spiciest_user']))
                    with cd_col4:
                        st.metric("Drama Climate", str(drama_data['group_drama_level']))

                    st.divider()

                    # Chaos Leaderboard
                    st.subheader("🌪️ Chaos Meter Leaderboard (0-100)")
                    chaos_df = chaos_data['chaos_leaderboard']
                    if not chaos_df.empty:
                        ch_col1, ch_col2 = st.columns([1, 1.2])
                        with ch_col1:
                            fig, ax = plt.subplots(figsize=(7, 4.5))
                            ax.barh(chaos_df['User'], chaos_df['Chaos Score'], color='#EC4899', edgecolor='#BE185D')
                            plt.gca().invert_yaxis()
                            plt.xlabel("Chaos Index (0-100)")
                            plt.title("Unhinged Texting Meter")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        with ch_col2:
                            st.dataframe(
                                chaos_df[['User', 'Chaos Score', 'Chaos Tier', 'ALL CAPS %', 'Late Night %', 'Multi-Text %', 'Rapid Burst %']],
                                use_container_width=True
                            )

                    st.divider()

                    # Drama Episodes Detector
                    st.subheader("🔥 High-Tension Drama Episodes & Arguments Detected")
                    if drama_data['drama_events']:
                        for ep in drama_data['drama_events']:
                            st.markdown(f"""
                            <div class="spice-card">
                                <h4>{ep['severity']} — {ep['timestamp']}</h4>
                                <p><b>Drama Intensity:</b> {ep['drama_intensity']}/100 &nbsp;|&nbsp; <b>Messages in Window:</b> {ep['message_count']} &nbsp;|&nbsp; <b>Sentiment:</b> {ep['sentiment_score']}</p>
                                <p><b>Key Participants:</b> {ep['key_participants']}</p>
                                <p><i>Sample Excerpts:</i><br>{"<br>".join([f'• "{q}"' for q in ep['sample_snippets']])}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("🕊️ No high-tension drama spikes detected in this chat export.")

                    st.divider()

                    # Vulgarity & Toxicity Leaderboards
                    st.subheader("🌶️ Vulgarity & Spice Leaderboard")
                    vulg_df = tox_data['vulgarity_leaderboard']
                    if not vulg_df.empty:
                        st.dataframe(vulg_df, use_container_width=True)

                    # Vulgarity Timeline
                    if not tox_data['timeline'].empty:
                        st.subheader("📈 Vulgarity & Toxicity Evolution Over Time")
                        fig, ax = plt.subplots(figsize=(12, 4))
                        ax.plot(tox_data['timeline']['time_period'], tox_data['timeline']['vulgarity_pct'], marker='s', color='#EF4444', linewidth=2, label='Vulgarity %')
                        plt.xticks(rotation=45, ha='right')
                        plt.ylabel("Vulgarity %")
                        plt.title("Monthly Profanity & Spice Level Trajectory")
                        plt.grid(True, linestyle=':', alpha=0.5)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                # =========================================================================
                # TAB 7: PERSONALITY & STYLE BADGES (PHASE 5)
                # =========================================================================
                with tab_style:
                    st.header("🎖️ Communication Style & Personality Flavor")
                    style_data = advanced.analyze_communication_styles(df)

                    st.subheader("👤 Participant Personality Showcase")
                    user_badges = style_data['user_badges']
                    user_details = style_data['user_details']

                    if user_badges:
                        card_cols = st.columns(min(len(user_badges), 3))
                        for idx, (usr, badges) in enumerate(user_badges.items()):
                            col_target = card_cols[idx % 3]
                            details = user_details.get(usr, {})
                            with col_target:
                                badges_html = "".join([f"<li><b>{b['badge']}</b>: <i>{b['desc']}</i></li>" for b in badges])
                                st.markdown(f"""
                                <div class="badge-card">
                                    <h3 style="margin-top:0; color:#38BDF8;">👤 {usr}</h3>
                                    <p><b>Primary Style:</b> {details.get('primary_badge', 'N/A')}</p>
                                    <ul>{badges_html}</ul>
                                    <hr style="border: 0.5px solid #334155;">
                                    <small>
                                        📝 Avg Words: {details.get('avg_words', 0)} |
                                        😂 Emojis/msg: {details.get('avg_emojis', 0)} |
                                        ❓ Questions: {details.get('question_rate', 0)}% |
                                        🦉 Night Owl: {details.get('night_activity', 0)}%
                                    </small>
                                </div>
                                """, unsafe_allow_html=True)

                    st.divider()

                    st.subheader("📋 Comprehensive Behavioral Profiles Matrix")
                    if not style_data['style_profiles'].empty:
                        st.dataframe(style_data['style_profiles'], use_container_width=True)

                # =========================================================================
                # TAB 8: GROUP WRAPPED (FEATURE 6)
                # =========================================================================
                with tab_wrapped:
                    st.header("🎁 WhatsApp Group Wrapped Recap")
                    group_title = uploaded_file.name.replace(".txt", "").replace("WhatsApp Chat with ", "")
                    wrapped_data = advanced.generate_group_wrapped(df, group_title=group_title)

                    st.markdown(f"""
                    <div class="insight-box">
                        {wrapped_data['insights']}
                    </div>
                    """, unsafe_allow_html=True)

                    # Top stats banner
                    m = wrapped_data['metrics']
                    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
                    with w_col1:
                        st.metric("Total Messages", f"{m.get('total_messages', 0):,}")
                    with w_col2:
                        st.metric("Total Words Exchanged", f"{m.get('total_words', 0):,}")
                    with w_col3:
                        st.metric("Peak Activity Time", f"{m.get('busiest_day', '')} @ {m.get('busiest_hour', '')}")
                    with w_col4:
                        st.metric("Group Vibe", f"{m.get('group_vibe', '')}")

                    st.divider()

                    w_left, w_right = st.columns([1.1, 0.9])

                    with w_left:
                        st.subheader("🏆 Hall of Fame Superlatives")
                        for a in wrapped_data['awards']:
                            st.markdown(f"""
                            <div class="wrapped-card">
                                <h3>{a['icon']} {a['title']}</h3>
                                <h2 style="color: #38BDF8; margin: 4px 0;">👑 {a['winner']}</h2>
                                <p style="color: #94A3B8; margin: 0;">{a['subtitle']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    with w_right:
                        st.subheader("📸 Downloadable Status Card")
                        if wrapped_data['image_bytes']:
                            st.image(wrapped_data['image_bytes'], caption="WhatsApp Group Wrapped Recap Card", use_container_width=True)

                            st.download_button(
                                label="📥 Download Wrapped PNG Card (HD)",
                                data=wrapped_data['image_bytes'],
                                file_name=f"WhatsApp_Wrapped_{group_title.replace(' ', '_')}.png",
                                mime="image/png",
                                use_container_width=True
                            )
                        else:
                            st.info("Image generation unavailable.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.info("👆 Please upload a WhatsApp exported chat text file (.txt) using the sidebar to begin analysis.")
