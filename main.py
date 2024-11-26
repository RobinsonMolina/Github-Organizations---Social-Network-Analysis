import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import community as louvain  # Para el algoritmo Louvain
from networkx.algorithms import community as nx_community  # Para el algoritmo Girvan-Newman
from tabulate import tabulate  # Para formatear tablas

# Cargar los datos
df = pd.read_csv('organization.csv')

# Eliminar filas con valores NaN en las columnas 'Organisation' y 'member'
df = df.dropna(subset=['Organisation', 'member'])

# Reducir la muestra para optimizar
df_reducido = df.sample(n=5000, random_state=42).reset_index(drop=True)
df = df_reducido

# Tabla inicial: organizaciones y sus miembros
organization_member_count = df.groupby('Organisation')['member'].nunique().reset_index()
organization_member_count.columns = ['Organización', 'Número de Miembros']
organization_member_count['#'] = range(1, len(organization_member_count) + 1)
print(tabulate(organization_member_count[['#', 'Organización', 'Número de Miembros']], 
               headers='keys', tablefmt='fancy_grid'))

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

if top_collaborators:
    top_names, top_scores = zip(*top_collaborators)
    plt.figure(figsize=(10, 6))
    plt.barh(top_names, top_scores, color="steelblue")
    plt.xlabel("Centralidad")
    plt.ylabel("Colaborador")
    plt.title("Top 10 Colaboradores por Centralidad")
    plt.gca().invert_yaxis()
    plt.show()
else:
    print("No se encontraron suficientes colaboradores para mostrar el top 10.")

# Análisis de centralidad para organizaciones
organisation_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
organisation_centrality = {node: centrality[node] for node in organisation_nodes}

# Graficar las 10 organizaciones más centrales
top_organisation = sorted(organisation_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

if top_organisation:
    top_names, top_scores = zip(*top_organisation)
    plt.figure(figsize=(10, 6))
    plt.barh(top_names, top_scores, color="steelblue")
    plt.xlabel("Centralidad")
    plt.ylabel("Organización")
    plt.title("Top 10 Organizaciones por Centralidad")
    plt.gca().invert_yaxis()
    plt.show()
else:
    print("No se encontraron suficientes organizaciones para mostrar el top 10.")

# Parte 1: Comunidades usando Louvain
partition = louvain.best_partition(B)

# Obtener comunidades de Louvain
communities = {}
for node, community in partition.items():
    if community not in communities:
        communities[community] = []
    communities[community].append(node)

# Mostrar resultados de Louvain
num_communities_louvain = len(communities)
print(f"Número de comunidades detectadas por Louvain: {num_communities_louvain}")

louvain_summary = []
for i, (community, members) in enumerate(communities.items()):
    org_names = [name for name in members if name in organizations]
    rep_org = org_names[0] if org_names else "Sin Organización"
    louvain_summary.append((i + 1, f"Comunidad {i + 1}", rep_org, len(members)))

# Mostrar tabla de Louvain
print(tabulate(louvain_summary, headers=["#", "Comunidad", "Organización Representativa", "Número de Miembros"], tablefmt="fancy_grid"))

# Graficar las comunidades de Louvain
norm = Normalize(vmin=0, vmax=len(communities) - 1)
colors = plt.cm.tab10(norm(range(len(communities))))

plt.figure(figsize=(12, 8))
for i, com in enumerate(communities.values()):
    nx.draw_networkx_nodes(B, pos, nodelist=com, node_color=[colors[i]], node_size=100)
nx.draw_networkx_edges(B, pos, edge_color="gray")
plt.title("Comunidades en el Grafo de Organizaciones y Colaboradores (Louvain)")
plt.show()

# Parte 2: Comunidades usando Girvan-Newman
girvan_newman_communities = nx_community.girvan_newman(B)

# Obtener solo la primera partición (división en dos comunidades)
try:
    first_level_communities = next(girvan_newman_communities)
    first_level_communities = [list(community) for community in first_level_communities]

    # Mostrar resultados de Girvan-Newman
    num_communities_girvan_newman = len(first_level_communities)
    print(f"Número de comunidades detectadas por Girvan-Newman: {num_communities_girvan_newman}")

    girvan_newman_summary = []
    for i, community in enumerate(first_level_communities):
        org_names = [name for name in community if name in organizations]
        rep_org = org_names[0] if org_names else "Sin Organización"
        girvan_newman_summary.append((i + 1, f"Comunidad {i + 1}", rep_org, len(community)))

    # Mostrar tabla de Girvan-Newman
    print(tabulate(girvan_newman_summary, headers=["#", "Comunidad", "Organización Representativa", "Número de Miembros"], tablefmt="fancy_grid"))

    # Graficar las comunidades de Girvan-Newman
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(range(len(first_level_communities)))
    for i, community in enumerate(first_level_communities):
        nx.draw_networkx_nodes(B, pos, nodelist=community, node_color=[colors[i]], node_size=100)
    nx.draw_networkx_edges(B, pos, edge_color="gray")
    plt.title("Comunidades en el Grafo de Organizaciones y Colaboradores (Girvan-Newman)")
    plt.show()
except StopIteration:
    print("Girvan-Newman no pudo dividir más el grafo en comunidades.")
