# -*- coding: utf-8 -*-

# Copyright (c) 2012,2013 Statistics Finland
# Modifications 2017-2018 by Leo Leppänen
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This module contains a Px class which parses the PX file structure including meta

In addition there is a conversion functionality to create a Pandas DataFrame object with MultiIndex
(multidimensional table) from PX data
"""

import re
from collections import OrderedDict
from datetime import datetime
from functools import reduce
from itertools import cycle, zip_longest
from operator import mul

import pandas as pd
from lark import Lark, Token, Transformer

PYTHONIFY_PATTERN = re.compile(r'[\W]+')


px_parser = Lark(r"""
    start: assignment+

    assignment: (meta_assignment|data_assignment) ";" NEWLINE

    meta_assignment: key "=" [NEWLINE] [value ("," [NEWLINE] value)*]
    data_assignment: "DATA=" NEWLINE data

    key: /[A-Z0-9-]+/ ["[" language "]"] [specifier]

    value: (multiline|time_specifier|NUMBER|BOOLEAN)
    specifier: "(" ESCAPED_STRING ("," ESCAPED_STRING)* ")"
    language: /[a-z]{2}/

    data: dataline+
    dataline: /./+ NEWLINE

    BOOLEAN: ("YES"|"NO")
    multiline: line (";" NEWLINE line)*
    line: ESCAPED_STRING (NEWLINE ESCAPED_STRING)*
    time_specifier: "TLIST(" period_dtype ")" ("," ESCAPED_STRING)*
    period_dtype: ("A1"|"H1"|"Q1"|"M1"|"W1")

    %import common.ESCAPED_STRING
    %import common.NUMBER
    %import common.NEWLINE
""", start='start')


class PXTransformer(Transformer):
    def value(self, tokens):
        assert len(tokens) == 1
        t = tokens[0]
        if isinstance(t, Token):
            if t.type == 'BOOLEAN':
                t = dict(YES=True, NO=False)[t.value]
            elif t.type == 'NUMBER':
                t = int(t.value)
        return t

    def multiline(self, tokens):
        return '\n'.join(tokens)

    def line(self, tokens):
        return ''.join([t[1:-1].replace('#', '\n') for t in tokens if t.type == 'ESCAPED_STRING'])

    def specifier(self, tokens):
        tokens = [t.value[1:-1] for t in tokens]
        return dict(specifier=tokens)

    def key(self, tokens):
        assert len(tokens) >= 1
        key = tokens.pop(0)
        d = {'key': key.value.lower().replace('-', '_')}
        for t in tokens:
            assert isinstance(t, dict)
            d.update(t)
        return d

    def language(self, tokens):
        assert len(tokens) == 1
        return dict(language=tokens[0].value)

    def meta_assignment(self, tokens):
        key = tokens.pop(0)
        values = [t for t in tokens if getattr(t, 'type', None) != 'NEWLINE']
        if len(values) == 1:
            values = values[0]
        key['value'] = values
        return key

    def assignment(self, tokens):
        tokens = [t for t in tokens if isinstance(t, dict)]
        assert len(tokens) == 1
        return tokens[0]

    def start(self, tokens):
        d = OrderedDict()
        for t in tokens:
            key = t.pop('key')
            value = t.pop('value')
            if 'language' in t:
                # TODO: proper support for languages
                continue
            if 'specifier' in t:
                specifiers = t.pop('specifier')
                if len(specifiers) > 1:
                    # TODO: proper support for several specifiers
                    continue

                if key in d and not isinstance(d[key], OrderedDict):
                    d[key] = OrderedDict([('', d[key])])

                vals = d.setdefault(key, OrderedDict())

                for specifier in specifiers:
                    assert specifier not in vals
                    vals[specifier] = value
            else:
                assert key not in d
                d[key] = value
            assert not t

        return d


def grouper(n, iterable, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    Lifted from itertools module's examples
    """
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


class PxFile:
    def _generate_indices(self):
        """
        Pandas has a concept of MultiIndex for hierarchical or multidimensional tables
        PC Axis files have list of column and row variables (can be thought of as column
        and row headings for the purposes of this documentation)

        Lowest level (last in the list) variable is repeated for exactly one
        column or row each till all columns/rows have a variable

        Going up the convention states that upper level variable groups lower level variable.

        Since Pandas MultiIndex excepts certain format for its variable structure:

        first level : [val1, val1, val1, val1, val2, val2, val2, val2]
        second level: [valx, valx, valz, valz, valx, valx, valz, valz]
        third level : [vala, valb, vala, valb, vala, valb, vala, valb] the lowest level

        This is one algorithm for generating repeating variable values from PX table structure
        First level/dimension:
            repeat = cols or rows / number of level's values
        Second level:
            repeat = first iterations repeat/ number of second level's values
        And so on

        Example:
        cols = 12
        first level values = 2
        second level values = 3
        third level values = 3
        12/2 = 6
        6 / 2 = 3
        3 / 3 = 1
        """
        col_index = []
        rep_index = self.n_cols
        for n, field in enumerate(self.meta['heading']):
            field_values = [x.strip() for x in self.meta['values'][field]]
            repeats = rep_index / len(field_values)
            rep_index = repeats

            col_index.append(list())
            index = 0
            values = cycle(field_values)
            value = next(values)
            for i, rep in enumerate(range(self.n_cols)):
                if index == repeats:
                    index = 0
                    value = next(values)
                index += 1
                col_index[n].append(value)

        row_index = []
        rep_index = self.n_rows
        for n, field in enumerate(self.meta['stub']):
            field_values = [x.strip() for x in self.meta['values'][field]]
            repeats = rep_index / len(field_values)
            rep_index = repeats

            row_index.append(list())
            index = 0
            values = cycle(field_values)
            value = next(values)
            for i, rep in enumerate(range(self.n_rows)):
                if index == repeats:
                    index = 0
                    value = next(values)
                index += 1
                row_index[n].append(value)
        return col_index, row_index

    def to_df(self, melt=False, dropna=False, drop_sums=True):
        """
        Build a Pandas DataFrame from Px rows and columns
        """
        cols, rows = self._generate_indices()
        col_index = pd.MultiIndex.from_arrays(cols, names=self.meta['heading'])
        if len(self.meta['heading']) == 1:
            # Convert to flat index
            col_index = col_index.get_level_values(0)

        row_index = pd.MultiIndex.from_arrays(rows, names=self.meta['stub'])
        if len(self.meta['stub']) == 1:
            # Convert to flat index
            row_index = row_index.get_level_values(0)
        df = pd.DataFrame(self.data, index=row_index, columns=col_index)

        elimination = self.meta.get('elimination')
        if drop_sums and elimination:
            index = df.index.copy()
            for level, name in enumerate(df.index.names):
                if name not in elimination:
                    continue
                if isinstance(index, pd.MultiIndex):
                    index = index.drop(elimination[name], level=level)
                else:
                    index = index.drop(elimination[name])
            if isinstance(index, pd.MultiIndex):
                index = index.remove_unused_levels()
            df = df.reindex(index)

            for level, name in enumerate(df.columns.names):
                if name not in elimination:
                    continue
                df = df.drop(columns=elimination[name], level=level)

        if melt:
            df = pd.melt(df.reset_index(), id_vars=self.meta['stub'])
            variable_types = self.meta.get('variable_type', {})
            for row_name in df.columns:
                if row_name in self.meta['stub']:
                    dtype = 'category'
                elif variable_types.get(row_name, '') == 'Classificatory':
                    dtype = 'category'
                elif self.meta.get('contvariable', '') == row_name:
                    dtype = 'category'
                else:
                    continue
                df[row_name] = df[row_name].astype(dtype)
            if dropna:
                df.value = pd.to_numeric(df.value, errors='coerce')
                df = df.dropna()

        return df

    def __init__(self, meta, data):
        self.meta = meta

        # Number of rows and cols is multiplication of number of variables for
        # both directions
        self.n_cols = reduce(mul, [len(meta['values'][key]) for key in meta['heading']], 1)
        self.n_rows = reduce(mul, [len(meta['values'][key]) for key in meta['stub']], 1)

        self.data = list(grouper(self.n_cols, data))


def convert_cell(c):
    if c.startswith('"'):
        # Strip quotation marks
        return c[1:-1]
    else:
        try:
            return int(c)
        except ValueError:
            return float(c)


class PxParser:
    """
    PC Axis document structure as a object interface

    Creates dynamically fields containing everything from PC Axis file's
    metadata part (excluding multilingual fields for the moment)
    #FIXME multilingual fields
    """

    _timeformat = '%Y%m%d %H:%M'

    def parse(self, content):
        """
        Parses metadata keywords from px_doc and inserts those into self object
        Returns the data part
        """
        if isinstance(content, bytes):
            content = content.decode(self.encoding)

        meta_content, data = content.split("DATA=")

        tree = px_parser.parse(meta_content)
        meta = PXTransformer().transform(tree)

        if 'timeval' in meta:
            del meta['timeval']

        # Parse timestamps
        for key in ('creation_date', 'last_updated'):
            if key not in meta:
                continue
            if isinstance(meta[key], OrderedDict):
                # FIXME
                timestamp = list(meta[key].values())[0]
            else:
                timestamp = meta[key]
            meta[key] = datetime.strptime(timestamp, self._timeformat)

        if 'stub' not in meta:
            # If stub is not specified, we try to find the most suitable
            # 'heading' variable to act as the row index. If all the values
            # are integers, it's probably a year column, so we choose that.
            # If not, we pick the one that has the most values.
            heading = None
            for key in meta['heading']:
                for val in meta['values'][key]:
                    try:
                        int(val)
                    except ValueError:
                        break
                else:
                    heading = key
                    break
            else:
                lengths = [(x[0], len(x[1])) for x in meta['values'].items()]
                longest = max(lengths, key=lambda x: x[1])
                heading = longest[0]

            meta['stub'] = [heading]
            meta['heading'].remove(heading)

        # Make sure these are lists
        for key in ('heading', 'stub'):
            val = meta[key]
            if not isinstance(val, list):
                meta[key] = [val]

        if 'values' in meta:
            for field, val in meta['values'].items():
                if not isinstance(val, list):
                    meta['values'][field] = [val]

        lines = [l.strip() for l in data.strip().rstrip(';').splitlines()]
        cells = [convert_cell(c) for line in lines for c in line.split()]

        return PxFile(meta, cells)

    def __init__(self, language=None, encoding='ISO-8859-1'):
        self.language = language
        self.encoding = encoding
