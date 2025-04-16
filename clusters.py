import logging
import numpy as np
from pandas import DataFrame as DF
from bpemb import BPEmb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import colorsys
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)

@dataclass
class Cluster:
    id: int
    name: str
    color: str
    keywords: List[str]
    events: List[dict]
    vector: np.ndarray

class EventClusterer:
    def __init__(self, language='ru', vs=10000, dim=100):
        self.bpe = BPEmb(lang=language, vs=vs, dim=dim)
        self.clusters: Dict[int, Cluster] = {}
        self._color_palette = []
        
    def _generate_colors(self, n_clusters: int) -> None:
        hsv = [(i/n_clusters, 0.8, 0.7) for i in range(n_clusters)]
        rgb = [colorsys.hsv_to_rgb(*c) for c in hsv]
        self._color_palette = [
            '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
            for r, g, b in rgb
        ]
    
    def _embed_events(self, events: List[dict]) -> np.ndarray:
        return DF([self.bpe.embed(event['name']).mean(axis=0) for event in events])
    
    def _auto_name_cluster(self, events: List[dict]) -> str:
        word_counts = defaultdict(int)
        for event in events:
            for word in event['name'].split():
                word_counts[word.lower()] += 1
        top_words = sorted(word_counts.items(), 
                          key=lambda x: x[1], reverse=True)[:3]
        return " | ".join([w[0] for w in top_words])
    
    def cluster_events(self, events: List[dict], 
                      n_clusters: int = None) -> List[Cluster]:
        # Embedding
        embeddings = self._embed_events(events)
        
        # Determine optimal clusters
        if not n_clusters:
            n_clusters = self._find_optimal_clusters(embeddings)
        logger.info(f"Optimal clusters number: {n_clusters}")

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Create Cluster objects
        self._generate_colors(n_clusters)
        clusters = {}
        for cluster_id in range(n_clusters):
            cluster_events = [e for e, l in zip(events, labels) if l == cluster_id]
            cluster_emb = embeddings[labels == cluster_id].mean(axis=0)
            
            clusters[cluster_id] = Cluster(
                id=cluster_id,
                name=self._auto_name_cluster(cluster_events),
                color=self._color_palette[cluster_id],
                keywords=self._auto_name_cluster(cluster_events).split(" | "),
                events=cluster_events,
                vector=cluster_emb
            )
        
        self.clusters = clusters
        return list(clusters.values())
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, 
                              max_clusters: int = 10) -> int:
        scores = []
        for k in range(2, max_clusters+1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            scores.append(silhouette_score(embeddings, labels))
        return np.argmax(scores) + 2
