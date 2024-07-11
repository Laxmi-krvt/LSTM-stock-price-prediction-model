from enum import Enum


class Column(Enum):
    OPEN = 'Open'
    CLOSE = 'Close'
    DATE = 'Date'
    HIGH = 'High'
    LOW = 'Low'
    VOLUME = 'Volume'