# from https://github.com/mthrok/tenhou-log-utils/blob/master/tenhou_log_utils/parser.py
from urllib2 import unquote as _unquote
import xml.etree.ElementTree as ET

def ensure_unicode(string):
    """Convert string into unicode."""
    if not isinstance(string, unicode):
        return string.decode('utf-8')
    return string


def ensure_str(string):
    """Convert string into str (bytes) object."""
    if not isinstance(string, str):
        return string.encode('utf-8')
    return string

def unquote(string):
    unquoted = _unquote(ensure_str(string))
    if isinstance(string, unicode):
        return unquoted.decode('utf-8')
    return unquoted

def load_mjlog(xml_string):
    return ET.ElementTree(ET.fromstring(xml_string)).getroot()
