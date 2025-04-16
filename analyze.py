import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from pandas import DataFrame as DF
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from pandas import DataFrame
from typing import List

from clusters import Cluster

class ClusterVisualizer:
    @staticmethod
    def plot_cluster_distribution(clusters: List[Cluster]):
        sizes = [len(c.events) for c in clusters]
        labels = [c.name for c in clusters]
        
        plt.figure(figsize=(12, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%',
               colors=[c.color for c in clusters])
        plt.title('Event Cluster Distribution')
        plt.show()
    
    @staticmethod
    def plot_temporal_distribution(clusters: List[Cluster]):
        fig = px.timeline(
            data_frame=DataFrame([{
                "Cluster": c.name,
                "Start": e['start'],
                "End": e['end'],
                "Event": e['name']
            } for c in clusters for e in c.events]),
            x_start="Start",
            x_end="End",
            y="Cluster",
            color="Cluster",
            color_discrete_map={c.name: c.color for c in clusters}
        )
        fig.update_layout(height=600, width=1000)
        fig.show()
    
    @staticmethod
    def generate_wordcloud(cluster: Cluster):
        text = " ".join([e['name'] for e in cluster.events])
        wordcloud = WordCloud(width=800, height=400,
                            background_color='white').generate(text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Word Cloud: {cluster.name}")
        plt.show()
    
    @staticmethod
    def plot_embeddings(clusters: List[Cluster]):
        tsne = TSNE(n_components=2, perplexity=len(clusters) - 1, random_state=42)
#        for c in clusters:
#            print(c.vector)
        embeds = tsne.fit_transform(DF([c.vector for c in clusters]))
        
        plt.figure(figsize=(12, 8))
        for i, c in enumerate(clusters):
            plt.scatter(embeds[i, 0], embeds[i, 1], 
                       color=c.color, s=200, label=c.name)
            plt.text(embeds[i, 0], embeds[i, 1], c.name, 
                    fontsize=9, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.title('Cluster Embedding Space')
        plt.show()
