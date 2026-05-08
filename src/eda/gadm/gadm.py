import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import shape, MultiPolygon
import networkx as nx

# Load the GeoJSON file
import os
file_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "cleaned", "gadm_mys_l1_clean.geojson")
gdf = gpd.read_file(file_path)

# Convert to projected CRS for accurate area calculations
gdf = gdf.to_crs(epsg=3395)  # World Mercator (meters)

# 1. Area Size & Shape Complexity Analysis
print("=== Area Size & Shape Complexity Analysis ===")

# Calculate area (in km²)
gdf['area_km2'] = gdf['geometry'].area / 1_000_000

# Calculate shape complexity metrics

def calculate_shape_metrics(geom):
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    
    area = geom.area
    perimeter = geom.length
    
    # Compactness (circularity): 4π * area / perimeter²
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Fractal dimension: 2 * log(perimeter / 4) / log(area)
    fractal_dim = 2 * np.log(perimeter / 4) / np.log(area) if area > 0 and perimeter > 0 else 0
    
    # Number of vertices
    num_vertices = len(geom.exterior.coords)
    
    return pd.Series({
        'compactness': compactness,
        'fractal_dimension': fractal_dim,
        'num_vertices': num_vertices,
        'perimeter_km': perimeter / 1_000
    })

metrics = gdf['geometry'].apply(calculate_shape_metrics)
gdf = pd.concat([gdf, metrics], axis=1)

# Summary statistics
print("\nArea Statistics (km²):")
print(gdf['area_km2'].describe())

print("\nShape Complexity Metrics:")
print(gdf[['compactness', 'fractal_dimension', 'num_vertices', 'perimeter_km']].describe())

# Plot area distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
gdf['area_km2'].plot.hist(bins=20)
plt.title('Area Distribution (km²)')
plt.xlabel('Area (km²)')

# Plot compactness vs fractal dimension
plt.subplot(1, 2, 2)
plt.scatter(gdf['compactness'], gdf['fractal_dimension'], alpha=0.6)
plt.title('Shape Complexity')
plt.xlabel('Compactness')
plt.ylabel('Fractal Dimension')
plt.tight_layout()
plt.savefig('area_shape_analysis.png')
plt.close()

# 2. Boundary Adjacency Network Graph
print("\n=== Boundary Adjacency Network Graph ===")

# Create spatial adjacency graph
G = nx.Graph()

# Add nodes
for idx, row in gdf.iterrows():
    G.add_node(row['NAME_1'], area=row['area_km2'])

# Find spatial adjacencies
adjacencies = gdf.geometry.apply(lambda g: gdf[gdf.geometry.touches(g)]['NAME_1'].tolist())

# Add edges
for idx, row in gdf.iterrows():
    region = row['NAME_1']
    for neighbor in adjacencies[idx]:
        if not G.has_edge(region, neighbor):
            # Weight by shared boundary length
            neighbor_geom = gdf[gdf['NAME_1'] == neighbor].geometry.iloc[0]
            shared_boundary = row.geometry.intersection(neighbor_geom).length
            G.add_edge(region, neighbor, weight=shared_boundary)

# Network analysis
print("\nNetwork Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
print(f"Network density: {nx.density(G):.3f}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

# Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Add centrality to nodes
for node in G.nodes():
    G.nodes[node]['degree_centrality'] = degree_centrality[node]
    G.nodes[node]['betweenness_centrality'] = betweenness_centrality[node]

# Plot network
plt.figure(figsize=(12, 10))
ax = plt.gca()
pos = nx.spring_layout(G, seed=42)

# Node size based on area
node_sizes = [G.nodes[n]['area'] * 0.1 for n in G.nodes()]

# Node color based on degree centrality
node_colors = [degree_centrality[n] for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

plt.title("Boundary Adjacency Network Graph")
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=ax, label='Degree Centrality')
plt.tight_layout()
plt.savefig('adjacency_network.png')
plt.close()

# Save metrics
metrics_df = gdf[['NAME_1', 'area_km2', 'compactness', 'fractal_dimension', 'num_vertices', 'perimeter_km']]
metrics_df.to_csv('gadm_metrics.csv', index=False)

print("\nAnalysis complete. Output files generated:")
print("- area_shape_analysis.png: Visualization of area and shape metrics")
print("- adjacency_network.png: Visualization of boundary adjacency network")
print("- gadm_metrics.csv: CSV file with all metrics")