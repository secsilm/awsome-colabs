# Awsome Colabs

这里是一些个人认为比较好的 Colab Notebooks 的集合，所有 notebook 的最开始都会有一个指向 colab 的链接，如下：

![colab](https://i.imgur.com/50rLOBn.png)

如果你还不了解 Colab Notebook，那么简单来说就是云端的 Jupyter Notebook，并且 TensorFlow 等常用库已经配置好。详情可参见 [Colaboratory 简介 - Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) 和 [Colab: An easy way to learn and use TensorFlow – TensorFlow – Medium](https://medium.com/tensorflow/colab-an-easy-way-to-learn-and-use-tensorflow-d74d1686e309)。

## 目录：

### Estimators

这里主要是程序中使用 [`tf.estimator`](https://www.tensorflow.org/api_docs/python/tf/estimator) API 进行训练的 Colab Notebooks。

- [`linear.ipynb`](estimators/linear.ipynb)：这是使用 `tf.estimator` 中预构建的 [`LinearClassifier`](https://www.tensorflow.org/api_docs/python/tf/estimator/LinearClassifier) 和 [`tf.feature_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column) 来做线性二分类（收入是否超过 50K）， 数据集使用的 1994 年到 1995 年是美国人口普查数据 [U.S Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/Census+Income)。
- [`nlp_estimators.ipynb`](estimators/nlp_estimators.ipynb)：这个 notebook 是关于使用 `tf.estimator` 进行文本分类（评论是积极还是消极）的例子，既使用了预构建的 estimator，也使用了自定义的。数据集使用的是 [IMDB 电影评论数据](http://ai.stanford.edu/~amaas/data/sentiment/)，模型有线性分类、CNN 和 LSTM，而且分别使用了词袋模型、随机初始化的词向量和预训练的词向量（GloVe）来进行词嵌入。我对[原博文](http://ruder.io/text-classification-tensorflow-estimators/)进行了翻译，见 [使用 TensorFlow Estimators 进行文本分类](https://alanlee.fun/2018/07/18/text-classification-with-tensorflow-estimator/#Building-a-baseline)，更多的翻译可见我的 GitHub repo [secsilm/awesome-posts](https://github.com/secsilm/awesome-posts)。
