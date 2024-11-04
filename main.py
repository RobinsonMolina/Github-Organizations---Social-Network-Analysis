import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from networkx.algorithms import community

# Cargar los datos
df = pd.read_csv('organization.csv')

# Crear el grafo bipartito
B = nx.Graph()

# Añadir nodos para organizaciones y colaboradores
organizations = set(df['Organisation'])
collaborators = set(df['member'])

B.add_nodes_from(organizations, bipartite=0)  # Nodo tipo 0 para organizaciones
B.add_nodes_from(collaborators, bipartite=1)  # Nodo tipo 1 para colaboradores

# Añadir aristas entre organizaciones y colaboradores
edges = list(df.itertuples(index=False, name=None))  # Convertir cada fila en una tupla (org, col)
B.add_edges_from(edges)

# Dibujar el grafo bipartito con colores específicos para cada tipo de nodo
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(B, k=0.5, seed=42)  # Layout para el grafo

# Dibujar nodos de organizaciones (azul) y colaboradores (verde)
nx.draw_networkx_nodes(B, pos, nodelist=organizations, node_color="blue", node_size=100)
nx.draw_networkx_nodes(B, pos, nodelist=collaborators, node_color="green", node_size=100)
nx.draw_networkx_edges(B, pos, edge_color="gray")
nx.draw_networkx_labels(B, pos, font_size=8)

plt.title("Grafo Bipartito de Organizaciones y Colaboradores")
plt.show()

# Análisis de centralidad para colaboradores
collaborator_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 1}
centrality = nx.degree_centrality(B)
collaborator_centrality = {node: centrality[node] for node in collaborator_nodes}

# Graficar los 10 colaboradores más centrales
top_collaborators = sorted(collaborator_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
top_names, top_scores = zip(*top_collaborators)

plt.figure(figsize=(10, 6))
plt.barh(top_names, top_scores, color="steelblue")
plt.xlabel("Centralidad")
plt.ylabel("Colaborador")
plt.title("Top 10 Colaboradores por Centralidad")
plt.gca().invert_yaxis()
plt.show()

# Detectar comunidades usando el algoritmo de Clauset-Newman-Moore
communities = community.greedy_modularity_communities(B)

# Graficar las comunidades en el grafo
plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(range(len(communities)))
for i, com in enumerate(communities):
    nx.draw_networkx_nodes(B, pos, nodelist=com, node_color=[colors[i]], node_size=100)
nx.draw_networkx_edges(B, pos, edge_color="gray")
nx.draw_networkx_labels(B, pos, font_size=8)
plt.title("Comunidades en el Grafo de Organizaciones y Colaboradores")
plt.show()
