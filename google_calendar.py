import os
import logging
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datetime import datetime

# Настройки OAuth 2.0
SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.json'

logger = logging.getLogger(__name__)

class GoogleCalendarManager:
    def __init__(self, calendar_name="Student Camp Schedule"):
        self.service = self._authenticate()
        self.calendar_id = self._setup_calendar(calendar_name)
        self.color_map = {
            # Базовые цвета Google Calendar
            1: "Lavender",    # Лавандовый
            2: "Sage",        # Шалфейный
            3: "Grape",       # Виноградный
            4: "Flamingo",    # Фламинго
            5: "Banana",      # Банановый
            6: "Tangerine",   # Мандариновый
            7: "Peacock",     # Павлиний
            8: "Graphite",    # Графитовый
            9: "Blueberry",   # Черничный
            10: "Basil",      # Базилик
            11: "Tomato"      # Томатный
        }
        self.cluster_colors = {}

    def _authenticate(self):
        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
        
        return build('calendar', 'v3', credentials=creds)

    def _setup_calendar(self, name):
        # Поиск существующего календаря
        calendars = self.service.calendarList().list().execute()
        for cal in calendars.get('items', []):
            if cal['summary'] == name:
                return cal['id']
        
        # Создание нового календаря
        calendar = {
            'summary': name,
            'timeZone': 'Europe/Moscow'
        }
        created = self.service.calendars().insert(body=calendar).execute()
        return created['id']

    def _map_cluster_to_color(self, cluster_id):
        if cluster_id not in self.cluster_colors:
            color_id = (cluster_id % 11) + 1
            self.cluster_colors[cluster_id] = self.color_map[color_id]
        return self.cluster_colors[cluster_id]

    def create_events(self, clusters):
        for cluster in clusters:
#            color = self._map_cluster_to_color(cluster.id)
            for event in cluster.events:
                self._create_event(event, cluster.id)

    def _create_event(self, event_data, color):
        event = {
            'summary': event_data['name'],
            'location': event_data.get('location', ''),
            'description': event_data.get('description', ''),
            'start': {
                'dateTime': event_data['start'].isoformat(),
                'timeZone': 'Europe/Moscow',
            },
            'end': {
                'dateTime': event_data['end'].isoformat(),
                'timeZone': 'Europe/Moscow',
            },
            'colorId': color,
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'popup', 'minutes': 30}
                ]
            }
        }

        try:
            self.service.events().insert(
                calendarId=self.calendar_id,
                body=event
            ).execute()
            logger.info(f"Event created: {event_data['name']}")
        except Exception as e:
            logger.error(f"Error creating event: {str(e)}")
