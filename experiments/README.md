# CUML Experiments
Since this is my first time working with CUML this folder has some experiments from the official documentation. It serves both as a quick point of reference, as well as a short intro to using it.

While marketing researchers often recommend to use GPUs for NN, CNN, LSTM or similar architectures built using Keras, Tensorflow, or PyTorch they omit that "traditional" models like RF, SVM and NB can also leverage GPUs.
And while new libraries often come with a learning curve, the main goal of CUML is to provide a GPU accelerated scikit-learn like api. In that sense researchers already familiar with Sklearn that face large datasets can leverage their existing knowledge with cuml.

## Notes on expected speed-ups
While cuml usually beats out Sklearn in 1-to-1 direct comparisons, there are cases where the CPU version comes out on top.
1. **Small datasets**: The overhead of cuml, and the time needed to copy the data onto the GPU means, that very small datasets can usually run faster on the CPU. --> **IO-bound bottleneck**
2. **Highly parallel problems**: Tasks like hyperparameter optimization can be parallelized on the CPU. E.g. each core tackles a single fold, with 16 cores 16 folds are evaluated at the same time. While the GPU might be faster than a single core, so long as it is not 16 times faster than the CPU, the CPU will be faster.
This becomes even more appararent when the folds become so small, that the CPU beats out the GPU per-fold too.

So if you have a very large dataset, where your CPU cores are unlikely to beat the GPU, you should use the GPU instead. Similarly, some HPO configurations can take SIGNIFICANTLY longer than others (dart > gbtree > gblinear), in these cases it might make sense to split the training according to long running HPOs. E.g. our SVM model took 13682s for the Automotive dataset. Each fold took between 60s and 600s, the GPU took about: 

In our utils we provide a few scripts that provide summaries of the training times which can be used to evaluate which approach is better in that specific case. See [here](../utils/README.md) for an overview or [here](../utils/summarize_results.py) for the script.

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