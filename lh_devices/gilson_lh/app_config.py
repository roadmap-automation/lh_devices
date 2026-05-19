from pathlib import Path

LOG_PATH = Path(__file__).parent.parent.parent / 'logs'
PERSISTENT_PATH = Path(__file__).parent.parent.parent / 'persistent_state' / 'gilson_lh'
NOTIFICATION_SETTINGS = PERSISTENT_PATH / 'notification_settings.json'
LAYOUT_LOG = PERSISTENT_PATH / 'layout.json'
HISTORY_LOG = PERSISTENT_PATH / 'lh_jobs.sqlite'
RESERVATION_DB = PERSISTENT_PATH / 'well_reservations.sqlite'

class Config:

    log_path: str = LOG_PATH
    persistent_path: Path = PERSISTENT_PATH
    layout_path: Path = LAYOUT_LOG
    history_path: Path = HISTORY_LOG
    reservation_path: Path = RESERVATION_DB
    notify_path: Path = NOTIFICATION_SETTINGS
    gilson_lh_url: str = 'http://localhost:5001'

config = Config()
