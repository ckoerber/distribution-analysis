# Distribution Analysis

## Content
1. [Description](#Description)
2. [Contains](#Contains)
3. [Getting started](#Getting-started)
    1. [Perquisites](#Perquisites)
4. [Usage](#Usage)
5. [Authors](#Authors)
6. [License](#License)


## <a name="Description"></a>Description
Python module for analyzing and visualizing statistical distributions.

## <a name="Contains"></a>Contains
* `plotDist.py` the visualization module
* `utilities.py` the supporting data manipulation module

## <a name="Getting-started"></a>Getting started
The python library can be directly imported
```python
import distalysis.plotDist as pD
import distalysis.utilities as ut
```
You can also pip install the package by running the following command in the source directory
```bash
pip install .
```
Add flags like `-e` (symlink directory) or `--user` for local installation if you like.


### <a name="Perquisites"></a>Perquisites
Pip install the following Python modules:
* `numpy`
* `scipy`
* `statsmodels`
* `gvar`
* `matplotlib`
* `seaborn`
* `pandas`


## <a name="Usage"></a>Usage
See [`ExampleUsage.ipynb`](ExampleUsage.ipynb)


## <a name="Authors"></a>Authors
* **Christopher Körber**

## <a name="License"></a>License
MIT License

Copyright (c) 2018 Christopher Körber

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
