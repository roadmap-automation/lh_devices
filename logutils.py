import json
import logging
from typing import Dict, Any

# ======== Logging for collecting metadata from method classes ========
# adapted from https://github.com/madzak/python-json-logger/blob/master/src/pythonjsonlogger/jsonlogger.py
class JsonFormatter(logging.Formatter):

    def __init__(self, *args, **kwargs):
        super().__init__(fmt = '%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                         *args, **kwargs)
        self.default_msec_format = '%s.%03d'

    def format(self, record: logging.LogRecord):
        """Formats a log record and serializes to json"""

        record.asctime = self.formatTime(record)

        log_record: Dict[str, Any] = dict(time=record.asctime,
                                          level=record.levelname,
                                          message=record.msg)

        return json.dumps(log_record)

# https://stackoverflow.com/questions/37944111/python-rolling-log-to-a-variable
class MethodLogHandler(logging.Handler):

    def __init__(self, log_queue: list):
        logging.Handler.__init__(self)
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.append(self.format(record))

    def pop(self):
        rval = [v for v in self.log_queue]
        self.log_queue = []
        return rval
    
class Loggable:
    """Attaches a unique logger"""
    def __init__(self):
        logger = logging.getLogger(str(id(self)))
        logger.setLevel(logging.INFO)
        self.logger = logger
