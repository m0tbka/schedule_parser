import re
import logging
import logging.config
import argparse
import nltk
from datetime import datetime
from pathlib import Path
from docx import Document
from google_calendar import GoogleCalendarManager

from clusters import EventClusterer, Cluster
from analyze import ClusterVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s::%(funcName)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("log.txt", mode='a', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

class ParserStats:
    def __init__(self):
        self.total_rows = 0
        self.date_rows = 0
        self.event_rows = 0
        self.skipped_rows = 0
        self.error_rows = 0
        self.skipped_entries = []
        self.error_entries = []

    def print_summary(self):
        logger.info("=== Processing Summary ===")
        logger.info(f"Total rows processed:    {self.total_rows}")
        logger.info(f"Date headers detected:   {self.date_rows}")
        logger.info(f"Events successfully parsed: {self.event_rows}")
        logger.info(f"Skipped rows:            {self.skipped_rows}")
        logger.info(f"Rows with errors:        {self.error_rows}")

        if self.skipped_entries:
            logger.info(f"Skipped rows examples: {len(self.skipped_entries)}")
            for idx, entry in enumerate(self.skipped_entries, 1):
                logger.info(f"{idx}. {entry}")

        if self.error_entries:
            logger.info(f"Error examples: {len(self.error_entries)}")
            for idx, entry in enumerate(self.error_entries, 1):
                logger.info(f"{idx}. {entry[0]}")
                logger.info(f"   Error: {entry[1]}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse student camp schedule from DOCX to ICS')
    parser.add_argument('-i', '--input', type=str, default='docs/Программа Студкемпа Яндекса в МФТИ25.docx',
                        help='Input DOCX file path')
    parser.add_argument('-o', '--output', type=str, default='docs/student_camp_schedule.ics',
                        help='Output ICS file path')
    parser.add_argument('-U', '--update', action="store_true", help="Update or not calendar events")
    parser.add_argument('-P', '--plot', action="store_true", help="Plot or not clusters info")
    return parser.parse_args()

def parse_docx_to_events(docx_path, stats):
    doc = Document(docx_path)
    current_date = None
    events = []
    
    logger.info(f"Started processing document: {docx_path}")
    
    for table_idx, table in enumerate(doc.tables, 1):
        logger.info(f"Processing table #{table_idx} ({len(table.rows)} rows)")
        
        for row_idx, row in enumerate(table.rows, 1):
            stats.total_rows += 1
            cells = [cell.text.strip() for cell in row.cells]
            logger.debug(f"Table {table_idx} Row {row_idx}: {cells}")

            # Detect row type
            if is_date_row(cells):
                stats.date_rows += 1
                try:
                    current_date = parse_date_row(cells)
                    logger.info(f"New date detected: {current_date}")
                except Exception as e:
                    stats.error_rows += 1
                    error_msg = f"Error parsing date row {row_idx}: {e}"
                    stats.error_entries.append((cells, str(e)))
                    logger.error(error_msg)
            elif is_event_row(cells):
                try:
                    event = parse_event_row(cells, current_date)
                    events.append(event)
                    stats.event_rows += 1
                    logger.info(f"Added event: {event['name']} ({event['start'].time()}-{event['end'].time()})")
                except Exception as e:
                    stats.error_rows += 1
                    error_msg = f"Error parsing event row {row_idx}: {e}"
                    stats.error_entries.append((cells, str(e)))
                    logger.error(error_msg)
            else:
                stats.skipped_rows += 1
                stats.skipped_entries.append(cells)
                logger.warning(f"Skipping unrecognized row {row_idx}: {cells}")

    logger.info(f"Successfully processed {stats.event_rows} events")
    return events

def is_date_row(cells):
    return (
        len(cells) >= 3 and 
        cells[0] == cells[1] == cells[2] and 
        re.match(r"[А-Яа-я]+, \d{1,2} [А-Яа-я]+ \d{4} года", cells[0])
    )

def parse_date_row(cells):
    date_str = cells[0].replace("**", "").strip().split(' ')
    return datetime.strptime(date_str[1] + " 4 2025", "%d %m %Y")

def is_event_row(cells):
    return (
        len(cells) >= 3 and 
        re.match(r"\d{1,2}:\d{2}[–\-]\d{1,2}:\d{2}", cells[0].replace(' ', ''))
    )

def parse_event_row(cells, current_date):
    if not current_date:
        raise ValueError("No date context available")

    time_str = re.sub(r"[^0-9:–-]", "", cells[0])
    start_time, end_time = re.split(r"[–-]", time_str)
    
    event_name = cells[2].split("(")[0].strip()
    location = cells[3] if len(cells) > 3 else ""
    
    lecturer_match = re.search(r"\((.*?)\)", cells[2])
    description = f"Lecturer: {lecturer_match.group(1)}" if lecturer_match else ""
    
    start = datetime.combine(
        current_date.date(),
        datetime.strptime(start_time, "%H:%M").time()
    )
    end = datetime.combine(
        current_date.date(),
        datetime.strptime(end_time, "%H:%M").time()
    )
    
    return {
        "start": start,
        "end": end,
        "name": event_name,
        "location": location,
        "description": description
    }

def analyze_clusters(events, plot=False):
    # Кластеризация
    clusterer = EventClusterer()
    clusters = clusterer.cluster_events(events)
    
    # Анализ
    logger.info(f"Cluster Summary: {len(clusters)}")
    for cluster in clusters:
        logger.info(f"• {cluster.name} ({len(cluster.events)} events): {cluster.color}")
        for e in cluster.events:
            logger.info(f"\t- {e['name']}")
    
    # Визуализация
    if not plot:
        return clusters

    logger.info("Plotting cluster distribution")
    ClusterVisualizer.plot_cluster_distribution(clusters)
#    logger.info("Plotting temporal distribution")
#    ClusterVisualizer.plot_temporal_distribution(clusters)
    logger.info("Plotting embeddings")
    clusterer.plot_embeddings(clusters)
#    ClusterVisualizer.plot_embeddings(clusters)
    # Пример облака слов для первого кластера
    if clusters:
        logger.info("Plotting wordcloud")
        ClusterVisualizer.generate_wordcloud(clusters[0])
    
    return clusters


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(f"Args: {args}")
    stats = ParserStats()
    
    Path(args.input).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        events = parse_docx_to_events(args.input, stats)
        nltk.download('stopwords')
        clusters = analyze_clusters(events, args.plot)

        if args.update:
            calendar_manager = GoogleCalendarManager()
            calendar_manager.create_events(clusters)
    except FileNotFoundError:
        logger.exception(f"File not found: {args.input}")
    except Exception as e:
        logger.exception(f"Critical error: {e}")
    finally:
        stats.print_summary()
