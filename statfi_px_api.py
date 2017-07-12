"""
This module contains tools to load information about PC Axis files in
Statistics Finland's databases and download the files from the databases
using the open data API:
http://www.stat.fi/org/lainsaadanto/avoin_data_en.html

For license see LICENSE document
"""

import os, csv, datetime, urllib.request, urllib.parse, urllib.error, zlib, time

class PxInfo(object):
    """
    A simple object representation of PX information in 
    Statistics Finland's open data API:
    """

    _timeformat = '%Y-%m-%d %H:%M' #Just a cache place for dateformat

    def __init__(self, path, size, created, updated, variables,
tablesize, type, language, title, *args):
        self.path = path.strip()
        self.size = size.strip()
        self.created = created.strip()
        self.updated = updated.strip()
        self.variables = variables.strip()
        self.tablesize = tablesize.strip()
        self.type = type.strip()
        self.language = language.strip()
        self.title = title.strip()

    def __str__(self):
        return 'PX file %s: %s' % (self.path, self.title)

    def __repr__(self):
        return str(self)

    @property
    def created_dt(self):
        return datetime.datetime.strptime(self.created, self._timeformat)

    @property
    def updated_dt(self):
        return datetime.datetime.strptime(self.updated, self._timeformat)

def create_px(url="http://pxweb2.stat.fi/database/StatFin/StatFin_rap.csv"):
    """
    Creates a list of Px-objects from a given url. Url should point to a CSV file.
    Url's default value points to Statfin databases contents CSV.
    """
    response = urllib.request.urlopen(url)
    lines = iter(response.read().decode('iso-8859-1').splitlines())
    next(lines) # Skip headers
    return [PxInfo(*i) for i in csv.reader(lines, delimiter=";")]

def fetch_px_zipped(px_objs, target_dir=".", sleep=1):
    fetch_px(px_objs, target_dir=target_dir, compressed=True, sleep=sleep)

def fetch_px(px_objs, target_dir=".", compressed=False, sleep=1):
    """
    Fetch PC Axis files for given list of Px objects
    Save the files to target directory

    WARNING: Statfin database contains over 2500 PX files with many gigabytes of data.
    """

    for px_obj in px_objs:
        url_parts = urllib.parse.urlparse(px_obj.path)
        target_path = os.path.join(target_dir, url_parts.path[1:]) # url_parts.path starts with '/'
        target_path = os.path.abspath(target_path)

        if os.path.exists(target_path):
            if is_latest(px_obj.path, target_path):
                time.sleep(1)
                continue

        try:
            request = urllib.request.Request(px_obj.path)
            if compressed:
                request.add_header('Accept-encoding', 'gzip')
            response = urllib.request.urlopen(request)
        except urllib.error.HTTPError as e:
            continue

        makedirs(target_path)
        try:
            with open(target_path, 'wb') as f:
                data = response.read()
                if compressed:
                    data = zlib.decompress(data, zlib.MAX_WBITS|16)
                f.write(data)
        except IOError as e:
            break

        time.sleep(sleep)

def is_latest(url, file_path):
    """
    Check that network resource is newer than file resource
    """
    try:
        response = urllib.request.urlopen(urllib.request.Request(
            url,
            method='HEAD'
        ))
        file_mtime_dt = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        url_modified_dt = datetime.datetime.strptime(
            response.getheader('last-modified'),
            '%a, %d %b %Y %H:%M:%S GMT'
        )
        return url_modified_dt < file_mtime_dt
    except urllib.error.HTTPError as e:
        return True

def makedirs(px_path):
    try:
        os.makedirs(os.path.dirname(px_path))
    except OSError as e:
        pass
