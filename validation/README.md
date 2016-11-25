Cross Validation _Notes_
========================

A simple and powerful technique to test the accuracy of your algorithm and avoid
overfitting.

_Learning the parameters of a prediction function and testing it on the same data
is a methodological mistake: a model that would just repeat the labels of the
samples that it has just seen would have a perfect score but would fail to
predict anything useful on yet-unseen data. This situation is called overfitting._

_To avoid it, it is common practice when performing a (supervised) machine learning
experiment to hold out part of the available data as a **test set**._ - [Sklearn Documentation](http://scikit-learn.org/stable/modules/cross_validation.html).

In cross-validating our algorithm we usually split the dataset in two parts:

  * Training Set - **80%** of the whole dataset.
  * Test Set - **20%** of the whole dataset.

After the split, our dataset is ready for Training, Feature Transformation and Prediction.

In particular, given a set of _train_feature_ and _test_feature_ you may want to
follow the steps below to applying PCA on your features and run your algorithm.

```python
#### Training Phase
PCA.fit(train_features)         # Run PCA on your training features
PCA.transform(train_features)   # Run PCA transformation on the training features
SVC.fit(train_features)         # Fit you classifier on the new features

#### Testing Phase
PCA.transform(test_feature)     # DO NOT PCA.transform(train_features)
                                # it would lead to inconsistence results
SVC.predict(test_features)

#### Here you can compute the accuracy of your algorithm
```

K-Fold Cross Validation
=======================

Given a dataset, split it into _k_ chunks of the same size.
_For each chunk_, run your machine learning algorithm using the chunk as test set
and the other _k-1_ chunks as training set. Get the score of the single experiment.
The resulting score of the algorithm is the average of the _k-1_ scores.

_K-Fold_ technique operates on your dataset as a butcher, it firstly chop your dataset
into small _pieces_ of the same size and latter it will randomly select all but one
piece as training set and the one left as test set. It will then repeat the selection
for all the pieces. At the end of the operation, the _K-Fold_ accuracy is given
by averaging the values of the single runs.

GridSearchVC in Sklearn
=======================

_GridSearchCV_ is a way of systematically working through multiple combinations
of parameter tunes, cross-validating as it goes to determine which tune gives
the best performance of your algorithm. The beauty is that it can work through
many combinations in only a couple extra lines of code.

Let see the GridSearchVC in action.

    #### 1. Define the algorithm to use in the classifier.
    svr = svm.SVC()

    #### 2. Define the parameters to be tuned for the algorithm.
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

    #### 3. Create the classifier which generates the grid of parameters.
    clf = grid_search.GridSearchCV(svr, parameters)

    #### 4. The fit function tries all the parameter combinations and returns
    ####    a fitted classifier that is automatically tuned to the optimal parameter
    ####    combination.
    clf.fit(iris.data, iris.target)

    #### 5. Here is the magic result - the best parameters combination.
    clf.best_params_

Focusing a bit more on the second line of code, the _parameters_ dictionary will
be used by _GridSearchVC_ to generate a 2 x 2 = 4 classifiers with all the
combinations of the parameters received in the dictionary.
For the sake of completeness here below is listed the set of combinations:

  * (rbf, 1)
  * (rbf, 10)
  * (linear, 1)
  * (linear, 10)
