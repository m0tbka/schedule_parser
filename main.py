import re
import locale
import logging
import argparse
from datetime import datetime
from pathlib import Path
from docx import Document
from icalendar import Calendar, Event

locale.setlocale(locale.LC_TIME, 'ru_RU.UTF-8')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse student camp schedule from DOCX to ICS')
    parser.add_argument('-i', '--input', type=str, default='docs/Программа Студкемпа Яндекса в МФТИ25.docx',
                        help='Input DOCX file path')
    parser.add_argument('-o', '--output', type=str, default='docs/student_camp_schedule.ics',
                        help='Output ICS file path')
    return parser.parse_args()

def parse_docx_to_events(docx_path):
    doc = Document(docx_path)
    current_date = None
    events = []
    
    logger.info(f"Started processing document: {docx_path}")
    
    for table_idx, table in enumerate(doc.tables, 1):
        logger.info(f"Processing table #{table_idx} ({len(table.rows)} rows)")
        
        for row_idx, row in enumerate(table.rows, 1):
            cells = [cell.text.strip() for cell in row.cells]
            logger.debug(f"Table {table_idx} Row {row_idx}: {cells}")

            # Detect row type
            if is_date_row(cells):
                try:
                    current_date = parse_date_row(cells)
                    logger.info(f"New date detected: {current_date}")
                except Exception as e:
                    logger.error(f"Error parsing date row {row_idx}: {e}")
            elif is_event_row(cells):
                try:
                    event = parse_event_row(cells, current_date)
                    events.append(event)
                    logger.info(f"Added event: {event['name']} ({event['start'].time()}-{event['end'].time()})")
                except Exception as e:
                    logger.error(f"Error parsing event row {row_idx}: {e}")
            else:
                logger.warning(f"Skipping unrecognized row {row_idx}: {cells}")

    logger.info(f"Successfully processed {len(events)} events")
    return events

def is_date_row(cells):
    """Check if row contains date information"""
    return (
        len(cells) >= 3 and 
        cells[0] == cells[1] == cells[2] and 
        re.match(r"[А-Яа-я]+, \d{1,2} [А-Яа-я]+ \d{4} года", cells[0])
    )

def parse_date_row(cells):
    """Extract date from date row"""
    date_str = cells[0].replace("**", "").strip().split(' ')
    return datetime.strptime(date_str[1] + " 4 2025", "%d %m %Y")

def is_event_row(cells):
    """Check if row contains event information"""
    return (
        len(cells) >= 3 and 
        re.match(r"\d{1,2}:\d{2}[–\-]\d{1,2}:\d{2}", cells[0].replace(' ', ''))
    )

def parse_event_row(cells, current_date):
    """Parse event row into structured data"""
    # Clean time string
    time_str = re.sub(r"[^0-9:–-]", "", cells[0]).replace(' ', '')
    start_time, end_time = re.split(r"[–-]", time_str)
    
    # Parse event name and location
    event_name = cells[2].split("(")[0].strip()
    location = cells[3] if len(cells) > 3 else ""
    
    # Extract lecturer from parentheses
    lecturer_match = re.search(r"\((.*?)\)", cells[2])
    description = f"Lecturer: {lecturer_match.group(1)}" if lecturer_match else ""
    
    # Combine with current date
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

def create_ics(events, output_path):
    try:
        cal = Calendar()
        cal.add('prodid', '-//Student Camp Calendar//mxm.dk//')
        cal.add('version', '2.0')

        for event in events:
            ical_event = Event()
            ical_event.add('summary', event["name"])
            ical_event.add('dtstart', event["start"])
            ical_event.add('dtend', event["end"])
            ical_event.add('location', event["location"])
            ical_event.add('description', event["description"])
            cal.add_component(ical_event)

        with open(output_path, 'wb') as f:
            f.write(cal.to_ical())
        logger.info(f"File {output_path} successfully created")

    except Exception as e:
        logger.error(f"Error creating ICS file: {e}")
        raise

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create directories if missing
    Path(args.input).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    try:
        events = parse_docx_to_events(args.input)
        create_ics(events, args.output)
    except FileNotFoundError:
        logger.error(f"File not found: {args.input}")
    except Exception as e:
        logger.error(f"Critical error: {e}")
