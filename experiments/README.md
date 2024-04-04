# CUML Experiments
Since this is my first time working with CUML this folder has some experiments from the official documentation. It serves both as a quick point of reference, as well as a short intor into using it.

The main goal of CUML is to provide a GPU accelerated scikit-learn like api.

|File | Description|
|---|---|
|[cuml_benchmarks.ipynb](cuml_benchmarks.ipynb)| A benchmark notebook from cuml to show the speedup it provides
|[random_forest_example.py](random_forest_example.py)| A simple example featuring a random forest implementation in cuml

## Notes on making datasets
cuml excpects data to be either a cuDF DataFrame, Series or a cuda_array_interface compliant array.
As such we cannot simply pass our prior Panadas df into a cuml model, we must first transform them to a compliant datatype.

Thankfully cuml has a builtin for this:
~~~python
import cudf
import pandas as pd

data = [[0, 1], [1, 2], [3, 4]]
pdf = pd.DataFrame(data, columns=['a', 'b'], dtype=int)
pdf
#    a  b
# 0  0  1
# 1  1  2
# 2  3  4
gdf = cudf.from_pandas(pdf)
gdf
#    a  b
# 0  0  1
# 1  1  2
# 2  3  4
~~~