import re
import logging
import argparse
from datetime import datetime
from pathlib import Path
from docx import Document
from icalendar import Calendar, Event

from clusters import EventClusterer, Cluster
from analyze import ClusterVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
        logger.info("\n=== Processing Summary ===")
        logger.info(f"Total rows processed:    {self.total_rows}")
        logger.info(f"Date headers detected:   {self.date_rows}")
        logger.info(f"Events successfully parsed: {self.event_rows}")
        logger.info(f"Skipped rows:            {self.skipped_rows}")
        logger.info(f"Rows with errors:        {self.error_rows}")

        if self.skipped_entries:
            logger.info("\nSkipped rows examples:")
            for idx, entry in enumerate(self.skipped_entries, 1):
                logger.info(f"{idx}. {entry}")

        if self.error_entries:
            logger.info("\nError examples:")
            for idx, entry in enumerate(self.error_entries, 1):
                logger.info(f"{idx}. {entry[0]}")
                logger.info(f"   Error: {entry[1]}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse student camp schedule from DOCX to ICS')
    parser.add_argument('-i', '--input', type=str, default='docs/Программа Студкемпа Яндекса в МФТИ25.docx',
                        help='Input DOCX file path')
    parser.add_argument('-o', '--output', type=str, default='docs/student_camp_schedule.ics',
                        help='Output ICS file path')
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

def analyze_clusters(events):
    # Кластеризация
    clusterer = EventClusterer()
    clusters = clusterer.cluster_events(events)
    
    # Анализ
    logger.info("\nCluster Summary:")
    for cluster in clusters:
        logger.info(f"• {cluster.name} ({len(cluster.events)} events): {cluster.color}")
    
    # Визуализация
    logger.info("Plotting cluster distribution")
    ClusterVisualizer.plot_cluster_distribution(clusters)
#    logger.info("Plotting temporal distribution")
#    ClusterVisualizer.plot_temporal_distribution(clusters)
#    logger.info("Plotting embeddings")
#    ClusterVisualizer.plot_embeddings(clusters)
    
    # Пример облака слов для первого кластера
    if clusters:
        logger.info("Plotting wordcloud")
        ClusterVisualizer.generate_wordcloud(clusters[0])
    
    return clusters

def create_ics(events, output_path, clusters):
    try:
        cal = Calendar()
        cal.add('prodid', '-//Student Camp Calendar//mxm.dk//')
        cal.add('version', '2.0')

        # Создаем календарь с цветами
        for cluster in clusters:
            for event in cluster.events:
                ical_event = Event()
                ical_event.add('summary', event["name"])
                ical_event.add('dtstart', event["start"])
                ical_event.add('dtend', event["end"])
                ical_event.add('location', event["location"])
                ical_event.add('description', event["description"])
                ical_event.add('x-wr-calname', cluster.name)
                ical_event.add('color', cluster.color)
                cal.add_component(ical_event)

        with open(output_path, 'wb') as f:
            f.write(cal.to_ical())
        logger.info(f"File {output_path} successfully created")

    except Exception as e:
        logger.error(f"Error creating ICS file: {e}")
        raise

if __name__ == "__main__":
    args = parse_arguments()
    stats = ParserStats()
    
    Path(args.input).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        events = parse_docx_to_events(args.input, stats)
        clusters = analyze_clusters(events)
        create_ics(events, args.output, clusters)
    except FileNotFoundError:
        logger.error(f"File not found: {args.input}")
    except Exception as e:
        logger.error(f"Critical error: {e}")
    finally:
        stats.print_summary()
