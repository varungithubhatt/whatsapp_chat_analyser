import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import community
import json


def analyze_dynamics(df, reply_gap_minutes=20):
    """
    Computes Social Network Graph & Interaction Dynamics:
    1. Directed reply network & adjacency matrix.
    2. Response latency (avg minutes to reply per pair).
    3. Pairwise reciprocity & strongest bonds.
    4. Ghost rate (messages left unanswered).
    5. Network Centrality (Magnet, Social Glue, Connector/Bridge).
    6. Sub-clique / community detection.
    7. PyVis interactive network visualization HTML.
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'graph_html': "",
            'adjacency_matrix': pd.DataFrame(),
            'top_bonds': [],
            'ghost_scores': pd.DataFrame(),
            'centrality_table': pd.DataFrame(),
            'communities': [],
            'insights': "Not enough message data to compute relationship dynamics."
        }

    df_clean.sort_values(by='messages_date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    users = df_clean['user'].unique().tolist()
    total_messages_per_user = df_clean['user'].value_counts()

    # 1. Track sequential interactions (A -> B within reply_gap_minutes)
    df_clean['next_user'] = df_clean['user'].shift(-1)
    df_clean['next_date'] = df_clean['messages_date'].shift(-1)
    df_clean['gap_minutes'] = (df_clean['next_date'] - df_clean['messages_date']).dt.total_seconds() / 60.0

    # Replies from a different user within threshold
    replies_df = df_clean[
        (df_clean['next_user'].notna()) &
        (df_clean['next_user'] != df_clean['user']) &
        (df_clean['gap_minutes'] <= reply_gap_minutes)
    ].copy()

    # Directed interactions: sender -> responder
    interaction_counts = {}
    pair_latencies = {}

    for _, row in replies_df.iterrows():
        sender = row['user']
        responder = row['next_user']
        gap = row['gap_minutes']

        pair = (sender, responder)
        interaction_counts[pair] = interaction_counts.get(pair, 0) + 1
        pair_latencies.setdefault(pair, []).append(gap)

    # 2. Build Adjacency Matrix
    adj_matrix = pd.DataFrame(0, index=users, columns=users)
    for (sender, responder), count in interaction_counts.items():
        if sender in users and responder in users:
            adj_matrix.loc[sender, responder] = count

    # 3. Pairwise Strongest Bonds & Reciprocity
    pairs_processed = set()
    bonds = []

    for u1 in users:
        for u2 in users:
            if u1 == u2:
                continue
            pair_key = tuple(sorted([u1, u2]))
            if pair_key in pairs_processed:
                continue
            pairs_processed.add(pair_key)

            fwd = interaction_counts.get((u1, u2), 0)
            rev = interaction_counts.get((u2, u1), 0)
            total_pair = fwd + rev

            if total_pair >= 2:
                reciprocity = (min(fwd, rev) / max(fwd, rev)) * 100.0 if max(fwd, rev) > 0 else 0
                latencies = pair_latencies.get((u1, u2), []) + pair_latencies.get((u2, u1), [])
                avg_latency = round(float(np.mean(latencies)), 1) if latencies else 0.0

                bonds.append({
                    'Pair': f"{u1} ⇄ {u2}",
                    'User 1': u1,
                    'User 2': u2,
                    'Total Replies': total_pair,
                    f'{u1} → {u2}': fwd,
                    f'{u2} → {u1}': rev,
                    'Reciprocity %': round(reciprocity, 1),
                    'Avg Reply Time (min)': avg_latency
                })

    bonds_df = pd.DataFrame(bonds)
    if not bonds_df.empty:
        bonds_df.sort_values(by='Total Replies', ascending=False, inplace=True)
        bonds_df.reset_index(drop=True, inplace=True)
        top_bonds = bonds_df.head(10).to_dict(orient='records')
    else:
        top_bonds = []

    # 4. Ghost Score (Messages that received NO response from anyone within 30 min)
    ghost_conditions = (
        (df_clean['next_user'] != df_clean['user']) &
        ((df_clean['gap_minutes'] > 30.0) | (df_clean['gap_minutes'].isna()))
    )
    ghost_df = df_clean[ghost_conditions]
    ghost_counts = ghost_df['user'].value_counts()

    ghost_records = []
    for user in users:
        total_u = int(total_messages_per_user.get(user, 0))
        ghost_u = int(ghost_counts.get(user, 0))
        rate = round((ghost_u / total_u) * 100.0, 1) if total_u > 0 else 0.0
        ghost_records.append({
            'User': user,
            'Total Sent': total_u,
            'Ignored / Left on Read': ghost_u,
            'Ghost Rate %': rate
        })

    ghost_scores = pd.DataFrame(ghost_records)
    ghost_scores.sort_values(by='Ghost Rate %', ascending=False, inplace=True)
    ghost_scores.reset_index(drop=True, inplace=True)

    # 5. NetworkX Graph Analysis
    G = nx.DiGraph()
    for user in users:
        G.add_node(user, weight=int(total_messages_per_user.get(user, 0)))

    for (sender, responder), count in interaction_counts.items():
        G.add_edge(sender, responder, weight=count)

    # Centrality metrics
    in_deg = nx.in_degree_centrality(G) if len(G) > 1 else {u: 0 for u in users}
    out_deg = nx.out_degree_centrality(G) if len(G) > 1 else {u: 0 for u in users}
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight') if len(G) > 2 else {u: 0 for u in users}
    except Exception:
        betweenness = {u: 0 for u in users}

    centrality_records = []
    for user in users:
        centrality_records.append({
            'User': user,
            'In-Degree (Popularity / Magnet)': round(in_deg.get(user, 0) * 100, 1),
            'Out-Degree (Social Glue / Responsive)': round(out_deg.get(user, 0) * 100, 1),
            'Betweenness (Bridge / Connector)': round(betweenness.get(user, 0) * 100, 1)
        })

    centrality_table = pd.DataFrame(centrality_records)
    centrality_table.sort_values(by='In-Degree (Popularity / Magnet)', ascending=False, inplace=True)
    centrality_table.reset_index(drop=True, inplace=True)

    # Community Detection (Sub-cliques)
    communities_list = []
    try:
        undirected_G = G.to_undirected()
        if len(undirected_G.edges) > 1:
            comms = community.greedy_modularity_communities(undirected_G)
            for idx, c in enumerate(comms):
                if len(c) > 1:
                    communities_list.append({
                        'group_id': idx + 1,
                        'members': list(c),
                        'size': len(c)
                    })
    except Exception:
        communities_list = []

    # 6. Generate PyVis Interactive Network HTML
    graph_html = ""
    try:
        from pyvis.network import Network
        net = Network(height="520px", width="100%", bgcolor="#0E1117", font_color="#FFFFFF", directed=True)

        palette = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"]

        # Map user to community color
        user_color = {}
        for c_idx, comm_item in enumerate(communities_list):
            color = palette[c_idx % len(palette)]
            for member in comm_item['members']:
                user_color[member] = color

        for user in users:
            msg_count = int(total_messages_per_user.get(user, 1))
            size = min(max(15, int(np.log1p(msg_count) * 7)), 45)
            color = user_color.get(user, "#38BDF8")
            ghost_pct = ghost_scores[ghost_scores['User'] == user]['Ghost Rate %'].values
            ghost_val = ghost_pct[0] if len(ghost_pct) > 0 else 0

            title_text = (
                f"<b>{user}</b><br>"
                f"Messages: {msg_count}<br>"
                f"Ghost Rate: {ghost_val}%<br>"
                f"In-Degree: {round(in_deg.get(user,0)*100, 1)}%"
            )
            net.add_node(user, label=user, title=title_text, value=size, color=color)

        for (sender, responder), weight in interaction_counts.items():
            if sender in users and responder in users and weight > 0:
                width = min(max(1, int(np.log1p(weight) * 2)), 8)
                net.add_edge(
                    sender,
                    responder,
                    value=weight,
                    title=f"{sender} → {responder}: {weight} replies",
                    color="rgba(148, 163, 184, 0.45)",
                    arrows="to"
                )

        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3500,
              "centralGravity": 0.25,
              "springLength": 120,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "minVelocity": 0.75
          },
          "edges": {
            "smooth": {"type": "continuous"}
          }
        }
        """)

        graph_html = net.generate_html()
    except Exception:
        graph_html = ""

    # Top insight
    if top_bonds:
        lead_pair = top_bonds[0]
        insights = (
            f"🔗 **{lead_pair['User 1']}** and **{lead_pair['User 2']}** have the strongest interaction bond "
            f"with {lead_pair['Total Replies']} direct cross-replies and {lead_pair['Reciprocity %']}% reciprocity."
        )
    else:
        insights = "Interactions are evenly spread throughout the participants."

    return {
        'graph_html': graph_html,
        'adjacency_matrix': adj_matrix,
        'top_bonds': top_bonds,
        'bonds_df': bonds_df,
        'ghost_scores': ghost_scores,
        'centrality_table': centrality_table,
        'communities': communities_list,
        'insights': insights
    }
