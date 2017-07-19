
# PC Axis PX file format reader for Python

# WIP

Now requires Python 3, but has not been tested beyond a simple Download -> PX file on disk -> Pandas Dataframe workflow on Python 3.6 and Pandas 0.20.2. Python 2 is not supported at the moment (PX file -> Pandas DataFrame works on commit 7669e5ccb5b8e0825d1044d5fe908b162904a523, current status in unknown).

## Usage

Requisite packages:

```
pip install pandas
```

To fetch list of available PX files
```python
import statsfi_px_api
px_info = statsfi_px_api.list_available_px(url="http://pxweb2.stat.fi/database/StatFin/StatFin_rap.csv")
```

To download PX Files to disk (default to current directory)
```python
statsfi_px_api.download_px(px_info, target_dir=".", compressed=False, sleep=1)
```
**NB:** Setting `compressed=True` is broken at 2017-07-12 due to stats.fi API not behaving as documented.

To read files into Pandas DataFrames
```python
import px_reader
px_obj = px_reader.Px('a_px_file_on_filesystem.px')
pandas_dataframe = px_obj.pd_dataframe()
```

## Features

Notable feature is conversion to a [Pandas][pandas] DataFrame using MultiIndex, which supports multidimensional table object. Pandas calls this [hierarchical indexing][pandas indexing]. Pandas has an [extensive feature list][pandas features]. Thus you can use PC Axis files for data analysis, visualization and export to other data formats.

Installing scientific Python toolset can be a daunting task. One option is the [Anaconda distribution][anaconda]. Otherwise Pandas installation may work with just `pip install pandas`. This code is unsupported, but please create an issue if you run into problems.

[anaconda]: http://continuum.io/downloads.html
[pandas]: http://pandas.pydata.org/
[pandas features]: http://pandas.pydata.org/#library-highlights
[pandas indexing]: http://pandas.pydata.org/pandas-docs/stable/indexing.html#hierarchical-indexing-multiindex

License
-------

This repository is a fork from https://github.com/jussiarpalahti/opendata which in turn is forked from https://github.com/statfi/opendata 

Below is its license which applies here also

All code here is under the BSD license unless otherwise stated. Otherwise Mozilla public license (MPL) is used since it supports both open and proprietary development alike.
