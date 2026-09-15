import re
from zookeeper_sft.models import LogEntry


def parse_line(line: str, pattern: str) -> LogEntry:
    """
    Parse a raw log string into structure LogEntry class using
    regex to extract components necessary
    """

    regex = re.compile(pattern)
    match = regex.match(line)
    if match:
        return LogEntry(
            timestamp=match.group(1),
            level=match.group(2),
            thread_component=match.group(3),
            message=match.group(4),
            raw=line,
        )
    return LogEntry(
        timestamp="",
        level="UNKNOWN",
        thread_component="general",
        message=line,
        raw=line,
    )
