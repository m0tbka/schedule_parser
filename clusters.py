import logging
import numpy as np
from pandas import DataFrame as DF
from bpemb import BPEmb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import colorsys
from dataclasses import dataclass
from typing import List, Dict
import re
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
from operator import itemgetter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def preprocess_event_name(text, n_words=None):
    # Инициализация лемматизатора и стоп-слов
    morph = MorphAnalyzer()
    russian_stopwords = stopwords.words('russian') + ['мфти', 'фпми']

    # Приведение к нижнему регистру
    text = text.lower()
    
    # Удаление лишних символов (все, кроме букв, цифр и пробелов)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Токенизация по пробелам
    words = text.split()
    
    # Удаление стоп-слов и лемматизация
    processed_words = []
    for word in words:
        if word not in russian_stopwords and not word.isdigit():
            # Лемматизация
            parsed_word = morph.parse(word)[0]
            lemma = parsed_word.normal_form
            processed_words.append(lemma)
#            processed_words.append(word)
    
    # Сборка обратно в строку
    processed_text = ' '.join(processed_words[:n_words if n_words else len(processed_words)])
    
    return processed_text

@dataclass
class Cluster:
    id: int
    name: str
    color: str
    keywords: List[str]
    events: List[dict]
    vector: DF

class EventClusterer:
    def __init__(self, language='ru', vs=100000, dim=200):
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
    
    def plot_embeddings(self, clusters: List[Cluster]):
        """Визуализирует все события в 2D пространстве с цветами кластеров"""
        # Собираем все эмбеддинги и метки
        all_embeddings = []
        all_labels = []
        event_names = []
    
        for cluster in clusters:
            for event in cluster.events:
                # Используем эмбеддинг названия события
                event_embedding = self.bpe.embed(event['name']).mean(axis=0)
                all_embeddings.append(event_embedding)
                all_labels.append(cluster.id)
                event_names.append(event['name'])
    
        if not all_embeddings:
            logger.warning("No events to visualize")
            return
    
    # Преобразуем в numpy массивы
        embeddings_array = np.array(all_embeddings)
        labels_array = np.array(all_labels)
    
    # Снижение размерности
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
        reduced_embeds = tsne.fit_transform(embeddings_array)
    
    # Создаем график
        plt.figure(figsize=(14, 10))
    
    # Рисуем точки для каждого кластера
        for cluster in clusters:
            mask = labels_array == cluster.id
            plt.scatter(
                reduced_embeds[mask, 0], 
                reduced_embeds[mask, 1],
                color=cluster.color,
                label=f"{cluster.name} ({np.sum(mask)} events)",
                alpha=0.6,
                s=50
            )
    
    # Добавляем подписи для некоторых точек
        for i in range(0, len(event_names), len(event_names)//20):
            plt.text(
                reduced_embeds[i, 0], 
                reduced_embeds[i, 1],
                event_names[i][:15] + "...",
                fontsize=8,
                alpha=0.7
            )
    
        plt.title('Event Embeddings Visualization by Cluster', pad=20)
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(alpha=0.2)
        plt.tight_layout()
    
    # Сохраняем и показываем
#        plt.savefig('event_embeddings.png', dpi=150, bbox_inches='tight')
        plt.show()

    def _embed_events(self, events: List[dict]) -> np.ndarray:
#         vect = TfidfVectorizer(use_idf=False)
#         corpus = ' '.join([' '.join([w for w in e['name'].split()]) for e in events])
         corpus = [e['name'] for e in events]
#         mat = vect.fit_transform(corpus)
#         sr = [word for _, word in sorted(zip(mat.toarray()[0], vect.get_feature_names_out()), reverse=True, key=itemgetter(0, 1))]
         sr = defaultdict(int)
         for event in events:
             for word in event['name'].split():
                 sr[word.lower()] += 1
         logger.info(f"Sorted TF-IDF words:")
         logger.info(sr)
#         corpus = [' '.join(map(lambda x: x[1], sorted([(sr.index(w), w) if w in sr else (10000, w) for w in e.split()[:4]])[:3])) for e in corpus]
         corpus = [' '.join(map(lambda x: x[1], sorted([(sr.get(w, -1), w) for w in e.split()], reverse=True, key=itemgetter(0, 1))[:2])) for e in corpus]
         logger.info(f"New filtered corpus:")
         logger.info(corpus)
#         vect = TfidfVectorizer()
#         return vect.fit_transform(corpus)
         
         return np.array([self.bpe.embed(event).mean(axis=0) 
                       for event in corpus])
    
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

        for e in events:
            e["bname"] = e["name"]
            e["name"] = preprocess_event_name(e["name"])

        # Embedding
        embeddings = self._embed_events(events)
        
        # Determine optimal clusters
        if not n_clusters:
            n_clusters = self._find_optimal_clusters(embeddings)
        logger.info(f"Optimal clusters number: {n_clusters}")

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, tol=1e-8)
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
                              max_clusters: int = 11) -> int:
        scores = []
        for k in range(2, max_clusters+1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            scores.append(silhouette_score(embeddings, labels))
        logger.info(f"Best silhouette_score: {np.max(scores)}")
        return np.argmax(scores) + 2
