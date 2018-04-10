# -*- coding: utf-8 -*-

# Copyright (c) 2012,2013 Statistics Finland
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This module contains a Px class which parses the PX file structure including meta

In addition there is a conversion functionality to create a Pandas DataFrame object with MultiIndex
(multidimensional table) from PX data

Note: Python 2.7 support required
"""

from collections import OrderedDict as OD
from itertools import zip_longest, cycle
from operator import mul
import pandas as pd
from functools import reduce
import re

PYTHONIFY_PATTERN = re.compile('[\W]+')

class Px(object):
    """
    PC Axis document structure as a object interface

    Creates dynamically fields containing everything from PC Axis file's metadata part
    (excluding multilingual fields for the moment #FIXME multilingual fields)
    """
    
    _timeformat = '%Y-%m-%d %H:%M'
    _subfield_re = re.compile(r'^(.*?)\("(.*?)"\)=')
    _items_re = re.compile(r'"(.*?)"')

    def _get_subfield_name(self, field):
        m = self._subfield_re.search(field)
        if m:
            return m.groups()

    def _clean_value(self, value):
        items = self._items_re.findall(value)
        if len(items) == 1:
            return items.pop()
        else:
            return items

    def _get_subfield(self, m, line):
        field, subkey = m.groups()
        value = line[m.end():]
        return field.lower(), subkey, self._clean_value(value)

    def _split_px(self, px_doc, language=None):
        """
        Parses metadata keywords from px_doc and inserts those into self object
        Returns the data part
        """
        meta, data = open(px_doc, encoding='ISO-8859-1').read().split("DATA=")
        nmeta = {}
        for line in meta.strip().split(';\n'):
            line=line.strip()
            if line:
                m = self._subfield_re.match(line)
                if m:
                    field, subkey, value = self._get_subfield(m, line)
                    if language: 
                        if not '[{}]'.format(language) in field:
                            continue
                        field = field[:-2-len(language)]
                        
                    if hasattr(self, field):
                        getattr(self, field)[subkey] = value
                    else:
                        setattr(self, field, OD(
                            [(subkey, value)]
                            ))
                else: 
                    field, value = line.split('=', 1)
                    if language: 
                        if not '[{}]'.format(language) in field:
                            continue
                        field = field[:-2-len(language)]
                    if not field.startswith('NOTE'):
                        setattr(self, field.strip().lower(), self._clean_value(value))
                        #TODO: NOTE keywords can be standalone or have subfields...
        return data.strip()[:-1]
   
    def __init__(self, px_doc, language=None):
        self._data = self._split_px(px_doc, language=language)

        if type(self.stub) != type(list()):
            self.stub = [self.stub]

        if type(self.heading) != type(list()):
            self.heading = [self.heading]

        for key, val in list(self.values.items()):
            if type(val) != type(list()):
                self.values[key] = [val]

        #
        # Number of rows and cols is multiplication of number of variables for both directions
        #
        self.cols = reduce(mul, [len(self.values.get(i)) for i in self.heading], 1)
        self.rows = reduce(mul, [len(self.values.get(i)) for i in self.stub], 1)
   
    @property
    def created_dt(self):
        return datetime.datetime.strptime(self.created, self._timeformat)
   
    @property
    def updated_dt(self):
        return datetime.datetime.strptime(self.updated, self._timeformat)

    @property
    def data(self):
        return list(grouper(self.cols, self._data.split()))

    def pd_dataframe(self):
        """
        Shortcut function to return Pandas DataFrame build from PX file's structure
        """
        return build_dataframe(self)


def grouper(n, iterable, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    Lifted from itertools module's examples
    """
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def generate_indices(px):
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
    rep_index = px.cols
    for n, field in enumerate(px.heading):
        field_values = px.values.get(field)
        repeats = rep_index / len(field_values)
        rep_index = repeats

        col_index.append(list())
        index = 0
        values = cycle(field_values)
        value = next(values)
        for i, rep in enumerate(range(px.cols)):
            if index == repeats:
                index = 0
                value = next(values)
            index += 1
            col_index[n].append(value)
    row_index = []
    rep_index = px.rows
    for n, field in enumerate(px.stub):
        field_values = px.values.get(field)
        repeats = rep_index / len(field_values)
        rep_index = repeats

        row_index.append(list())
        index = 0
        values = cycle(field_values)
        value = next(values)
        for i, rep in enumerate(range(px.rows)):
            if index == repeats:
                index = 0
                value = next(values)
            index += 1
            row_index[n].append(value)
    return col_index, row_index


def build_dataframe(px):
    """
    Build a Pandas DataFrame from Px rows and columns
    """
    cols, rows = generate_indices(px)
    col_index = pd.MultiIndex.from_arrays(cols)
    row_index = pd.MultiIndex.from_arrays(rows)
    return pd.DataFrame(px.data, index=row_index, columns=col_index)


def pythonify_column_names(df):
    """
    Pythonifies column names to *approximately* valid python identifiers. Column names
    starting with numbers will still be invalid python identifiers.

    spaces are replaced by underscores, 'ä', 'ö' and 'å' are mapped to 'a', 'o' and 'a'
    respectively. Percentage signs are replaced with the string 'perc'.

    All other non-alphanumerical characters are removed. Alphabet-characters are lowercased.

    Examples:
    'Voting turnout' -> 'voting_turnout'
    'Males %' -> 'males_perc'
    'Ellis Example / GOP' -> 'ellis_example__gop'
    """
    if isinstance(df, pd.Series):
        return df
    cols = [col for col in df]
    new_cols = _prepare_names_for_hdf5(cols)
    df.columns = new_cols
    return df


def _prepare_names_for_hdf5(names):
    new_names = []
    for name in names:
        if isinstance(name, list):
            new_names.append(prepare_names_for_hdf5(name))
        else:
            if isinstance(name, str):
                new_name = name.lower().replace(' ', '_').replace('ä', 'a').replace('ö', 'o').replace('å','a').replace('%', 'perc')
                new_name = PYTHONIFY_PATTERN.sub('', new_name)
                new_names.append(new_name)
            else:
                new_names.append(name)
    return new_names


def flatten(df, stacked_cols=None, unstacked_indices=None):
    """
    Flattens a pandas.DataFrame with MultiIndex row and/or column indices to a 2D pandas.DataFrame with either
    Index or RangeIndex row and column indices. If input is an instance of pandas.Series or an already flat 
    pandas.DataFrames, it is returned as-is.

    When figuring out indices for stacked_cols and unstacked_indices, note that stacked_cols are stacked before
    unstacked_indices are unstacked.

    :param df: DataFrame (or Series to Flatten)
    :param stacked_cols: Any column levels that should be extracted into column(s), rather than flattened into parts of
        the single-level column index. None indicates no stacking. Equivalent to calling DataFrame.stack(level=stacked_cols)
    :param unstacked_indices: Any row-indices that should be extracted into column indices. None indicates no unstacking.
        Equivalent to calling DataFrame.unstack(level=unstacked_indices).
    :return: the flattened DataFrame.
    """

    if isinstance(df, pd.Series):
        # It's Series and by definition already flat, return as is
        print(df.index.nlevels, df.columns.nlevels)
        return df

    if df.index.nlevels == 1 and df.columns.nlevels == 1:
        # This DataFrame is already flat, return as is
        print(df.index.nlevels, df.columns.nlevels)
        return df

    # If column is multi-indexed, extract the values of one column-index level to separate column.
    #
    # I.e. transform
    # |------------------------------------------------------------------------------------------------|
    # |                     total              total           ... Center Party     Center Party    ...|
    # |                     candidate_votes    candidate_votes ... candidate_votes  candidate_votes ...|
    # |                     all                male            ... all              male            ...|
    # | Koko maa    2015    2968459            1704054         ... 626218           408842          ...|
    # | Koko maa    2011    2939571            1709391         ... 463266           295650          ...|
    # | ...         ...     ...                ...             ... ...              ...             ...|
    # |------------------------------------------------------------------------------------------------|
    # to
    # |-----------------------------------------------------------------------------------------------------------------|
    # |                                     candidate_votes    candidate_votes  ... candidate_votes  candidate_votes ...|
    # |                                     all                male             ... all              male            ...|
    # | Koko maa    2015    total           2968459            1704054          ... 626218           408842          ...|
    # | Koko maa    2011    Center Party    2939571            1709391          ... 463266           295650          ...|
    # | ...         ...     ...             ...                ...              ... ...              ...             ...|
    # |-----------------------------------------------------------------------------------------------------------------|
    #
    if df.columns.nlevels > 1 and stacked_cols is not None:
        df = df.stack(level=stacked_cols)

    # Generates a column-index out of a row-index.
    #
    # I.e. transform
    #
    # one  a   1.0
    #      b   2.0
    # two  a   3.0
    #      b   4.0
    #
    # to
    #
    #        a    b
    # one    1.0  2.0
    # two    3.0  4.0
    #
    if df.index.nlevels > 1 and unstacked_indices is not None:
        df = df.unstack(level=unstacked_indices)

    # If the DataFrame still has multi-index for columns, flatten the column index.
    #
    # I.e. transform
    # |------------------------------------------------------------------------------------------------------------------|
    # |                                     candidate_votes    candidate_votes  ... candidate_votes   candidate_votes ...|
    # |                                     all                male             ... all               male            ...|
    # | Koko maa    2015    total           2968459            1704054          ... 626218            408842          ...|
    # | Koko maa    2011    Center Party    2939571            1709391          ... 463266            295650          ...|
    # | ...         ...     ...             ...                ...              ... ...               ...             ...|
    # |------------------------------------------------------------------------------------------------------------------|
    # to
    # |-----------------------------------------------------------------------------------------------------------------------------------------|
    # |                                     candidate_votes_all    candidate_votes_male    ... candidate_votes_all   candidate_votes_male    ...|
    # | Koko maa    2015    total           2968459                1704054                 ... 626218                408842                  ...|
    # | Koko maa    2011    Center Party    2939571                1709391                 ... 463266                295650                  ...|
    # | ...         ...     ...             ...                    ...                     ... ...                   ...                     ...|
    # |-----------------------------------------------------------------------------------------------------------------------------------------|
    if isinstance(df, pd.DataFrame) and df.columns.nlevels > 1:
        flat_col_names = ['_'.join(col_tuple) for col_tuple in df.columns]
        flat_idx = pd.Index(flat_col_names)
        df.columns = flat_idx

    # By this point, Pandas can figure out that the table no longer needs to be multi-indexed for
    # columns OR rows. Calling reset_index() reflects this to the DataFrame.
    # To be more explicit: at this point col-index has only one level can can be transformed to a
    # simple Index. reset_index() also -- far whatever reason I don't fully understand -- transforms
    # the ROW INDEX to either a RangeIndex or Index from a MultiIndex. Thus, at the end of this
    # operation the whole DataFrame is flat.
    df = df.reset_index()

    return df