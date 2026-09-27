import pandas as pd
import numpy as np
import networkx as nx


def analyze_cliques(df, max_response_mins=30.0):
    """
    Detects conversational cliques and subgroups in the WhatsApp group using
    community detection algorithms (Louvain or Greedy Modularity) on the reply graph.

    Computes:
    1. Subgroups / Cliques and their member list.
    2. Clique cohesion: In-clique vs Out-clique communication ratio.
    3. Core driver of each clique.
    4. Pairwise clique cross-talk metrics.
    """
    df_clean = df[df['user'] != 'group_notification'].copy()
    if df_clean.empty:
        return {
            'cliques': [],
            'clique_df': pd.DataFrame(),
            'modularity_score': 0.0,
            'insights': "Not enough message data for clique detection."
        }

    df_clean.sort_values(by='messages_date', inplace=True)
    df_clean.reset_index(drop=True, inplace=True)

    users = df_clean['user'].value_counts()
    active_users = users[users >= 2].index.tolist()
    if len(active_users) < 2:
        return {
            'cliques': [],
            'clique_df': pd.DataFrame(),
            'modularity_score': 0.0,
            'insights': "At least 2 active participants are required to analyze subgroups."
        }

    # Build interaction edges between sequential senders within max_response_mins
    df_clean['prev_user'] = df_clean['user'].shift(1)
    df_clean['prev_date'] = df_clean['messages_date'].shift(1)
    df_clean['gap_mins'] = (df_clean['messages_date'] - df_clean['prev_date']).dt.total_seconds() / 60.0

    valid_replies = df_clean[
        (df_clean['prev_user'].notna()) &
        (df_clean['prev_user'] != df_clean['user']) &
        (df_clean['gap_mins'] <= max_response_mins) &
        (df_clean['user'].isin(active_users)) &
        (df_clean['prev_user'].isin(active_users))
    ]

    # Create undirected weighted graph for community detection
    G = nx.Graph()
    for u in active_users:
        G.add_node(u)

    edge_weights = {}
    for _, row in valid_replies.iterrows():
        u1, u2 = sorted([row['prev_user'], row['user']])
        edge_weights[(u1, u2)] = edge_weights.get((u1, u2), 0) + 1

    for (u1, u2), weight in edge_weights.items():
        G.add_edge(u1, u2, weight=weight)

    if G.number_of_edges() == 0:
        return {
            'cliques': [],
            'clique_df': pd.DataFrame(),
            'modularity_score': 0.0,
            'insights': "No strong pairwise replies detected within response windows."
        }

    # Community Detection: Louvain or Greedy Modularity
    try:
        from networkx.algorithms.community import louvain_communities
        raw_communities = list(louvain_communities(G, weight='weight', seed=42))
    except Exception:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            raw_communities = list(greedy_modularity_communities(G, weight='weight'))
        except Exception:
            raw_communities = [set(c) for c in nx.connected_components(G)]

    # Filter out empty communities and sort by size
    communities = [list(c) for c in raw_communities if len(c) > 0]
    communities.sort(key=lambda c: len(c), reverse=True)

    # User to community mapping
    user_community = {}
    for idx, comm in enumerate(communities):
        for u in comm:
            user_community[u] = idx + 1

    # Analyze each clique
    cliques_info = []
    user_clique_records = []

    # Count internal vs external edges
    for idx, comm in enumerate(communities):
        clique_id = idx + 1
        comm_set = set(comm)
        internal_msgs = 0
        external_msgs = 0

        # Subgraph activity
        sub_msgs = df_clean[df_clean['user'].isin(comm_set)]
        total_comm_msgs = len(sub_msgs)

        # Internal edges
        for _, row in valid_replies.iterrows():
            u_from = row['prev_user']
            u_to = row['user']
            if u_from in comm_set and u_to in comm_set:
                internal_msgs += 1
            elif u_from in comm_set or u_to in comm_set:
                external_msgs += 1

        total_interactions = internal_msgs + external_msgs
        cohesion_pct = round((internal_msgs / total_interactions * 100.0), 1) if total_interactions > 0 else 0.0

        # Find clique leader (highest messages within the clique)
        leader = sub_msgs['user'].value_counts().index[0] if not sub_msgs.empty else comm[0]

        # Fun clique name based on size & rank
        if len(comm) >= 4:
            clique_name = f"Subgroup #{clique_id}: The Core Council"
        elif len(comm) == 3:
            clique_name = f"Subgroup #{clique_id}: The Trio"
        elif len(comm) == 2:
            clique_name = f"Subgroup #{clique_id}: The Dynamic Duo"
        else:
            clique_name = f"Subgroup #{clique_id}: Solo Operative"

        cliques_info.append({
            'clique_id': clique_id,
            'name': clique_name,
            'members': comm,
            'member_count': len(comm),
            'leader': leader,
            'internal_replies': internal_msgs,
            'external_replies': external_msgs,
            'cohesion_pct': cohesion_pct,
            'total_messages': total_comm_msgs
        })

        for u in comm:
            user_clique_records.append({
                'User': u,
                'Subgroup': f"Subgroup #{clique_id}",
                'Clique Members': ", ".join(comm),
                'Clique Cohesion %': cohesion_pct,
                'Clique Leader': leader
            })

    clique_df = pd.DataFrame(user_clique_records)

    # Summary insight
    num_cliques = len(communities)
    largest_clique = cliques_info[0]['name'] if cliques_info else "N/A"
    insights = (
        f"🔍 Detected **{num_cliques} natural subgroups / conversational clusters** in the chat. "
        f"The primary subgroup is **{largest_clique}** with {cliques_info[0]['member_count'] if cliques_info else 0} members and "
        f"{cliques_info[0]['cohesion_pct'] if cliques_info else 0}% internal reply cohesion."
    )

    return {
        'cliques': cliques_info,
        'clique_df': clique_df,
        'num_subgroups': num_cliques,
        'insights': insights
    }
