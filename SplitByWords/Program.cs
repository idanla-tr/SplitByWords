Console.Write("Enter number of words per line: ");
int wordsPerLine = int.Parse(Console.ReadLine());

string source = @"Algorithms in the Machine Learning Toolkit
The Splunk Machine Learning Toolkit (MLTK) supports all of the algorithms listed here. Details for each algorithm are grouped by algorithm type including Anomaly Detection, Classifiers, Clustering Algorithms, Cross-validation, Feature Extraction, Preprocessing, Regressors, Time Series Analysis, and Utility Algorithms. You can find more examples for these algorithms on the scikit-learn website.

The MLTK supported algorithms use the fit and apply commands. For information on the steps taken by these commands, see Understanding the fit and apply commands.

For information on using the score command, see Scoring metrics in the Machine Learning Toolkit.

ML-SPL Quick Reference Guide
Download the Machine Learning Toolkit Quick Reference Guide for a handy cheat sheet of current ML-SPL commands and machine learning algorithms available in the Splunk Machine Learning Toolkit. This document is also offered in Japanese.

ML-SPL Performance App
Download the ML-SPL Performance App for the Machine Learning Toolkit to use performance results for guidance and benchmarking purposes in your own environment.

Extend the algorithms you can use for your models
The algorithms listed here and in the ML-SPL Quick Reference Guide are available natively in the Splunk Machine Learning Toolkit. You can also base your algorithm on over 300 open source Python algorithms from scikit-learn, pandas, statsmodel, numpy and scipy libraries available through the Python for Scientific Computing add-on in Splunkbase.

For information on how to import an algorithm from the Python for Scientific Computing add-on into the Splunk Machine Learning Toolkit, see the ML-SPL API Guide.

Add algorithms through GitHub
On-prem customers looking for solutions that fall outside of the 30 native algorithms can use GitHub to add more algorithms. Join the Splunk Community for MLTK on GitHub. to also learn about new machine learning algorithms, solve custom uses cases through sharing and reusing algorithms, and help fellow users of the MLTK.

Splunk Cloud Platform customers can also use GitHub to add more algorithms via an app. The Splunk GitHub for Machine learning app provides access to custom algorithms and is based on the Machine Learning Toolkit open source repo. Splunk Cloud Platform customers need to create a support ticket to have this app installed.

Anomaly Detection
Anomaly detection algorithms detect anomalies and outliers in numerical or categorical fields.

DensityFunction
The DensityFunction algorithm provides a consistent and streamlined workflow to create and store density functions and utilize them for anomaly detection. DensityFunction allows for grouping of the data using the by clause, where for each group a separate density function is fitted and stored. This algorithm supports incremental fit.

The DensityFunction algorithm supports the following continuous probability density functions: Normal, Exponential, Gaussian Kernel Density Estimation (Gaussian KDE), and Beta distribution.

Using the DensityFunction algorithm requires running version 1.4 or higher of the Python for Scientific Computing add-on.

The accuracy of the anomaly detection for DensityFunction depends on the quality and the size of the training dataset, how accurately the fitted distribution models the underlying process that generates the data, and the value chosen for the threshold parameter.

Follow these guidelines to make your models perform more accurately:

Aim for fitted distributions to have a cardinality (training dataset size) of at least 50. If you cannot collect more training data, create fewer groups of data using the by clause, giving you more data points per group.
The threshold parameter has a default value, but ideally the value for threshold, lower_threshold, or upper_threshold are chosen based on experimentation as guided by domain knowledge.
Continue tuning the threshold parameter until you are satisfied with the results.
Inspect the model using the summary command.
The values reported for the mean and standard deviation are either the statistics of the fitted distribution, or of the data, depending on the type of the distribution.
In the case of parametric distributions (Normal, Beta, and Exponential) the mean and standard deviation are calculated from the fitted distribution. When the parametric distribution is not a good fit for the data, the reported mean and std might not be close to that of data.
In the case of non-parametric distributions (Gaussian KDE) the mean and standard deviation are calculated from the data passed in during fit.
If the distribution of the data changes through time, re-train your models frequently.
Parameters

The partial_fit parameter controls whether an existing model should be incrementally updated on not. This allows you to update an existing model using only new data without having to retrain it on the full training data set.
The partial_fit parameter default is False.
If partial_fit is not specified, the model specified is created and replaces the pre-trained model if one exists.
Using partial_fit=True on an existing model ignores the newly supplied parameters. The parameters supplied at model creation are used instead.
Valid values for the dist parameter include: norm (normal distribution), expon (exponential distribution), gaussian_kde (Gaussian KDE distribution), beta (beta distribution), and auto (automatic selection).
The dist parameter default is auto.
When set to auto, norm (normal distribution), expon (exponential distribution), gaussian_kde (Gaussian KDE distribution) , and beta (beta distribution) all run, with the best results returned.
Use the exclude_dist parameter to exclude a minimum of 1 and a maximum of 3 of the available dist parameter values (norm, expon, gaussian_kde, beta).
The exclude_dist parameter is only available when the dist parameter is auto.
DensityFunction will run using any non-excluded dist parameter values.
Use a comma to note multiple excluded dist parameter values. For example, exclude_dist=""beta,expon""
Attempts to use the exclude_dist parameter on more than 3 dist parameter values, or on a dist parameter other than auto will result in an error message.
Beta distribution was added in version 5.2.0 of the Machine Learning Toolkit
If the data distribution takes a U shape, outlier detection will not be accurate.
The metric parameter calculates the distance between the sampled dataset from the density function and the training dataset.
Valid metrics for the metric parameter include: kolmogorov_smirnov and wasserstein.
The metric parameter default is wasserstein.
The sample parameter can be used during fit or apply stages.
The sample parameter default is False.
If the sample parameter is set to True during the fit stage, the size of the samples will be equal to the training dataset.
If the sample parameter is set to True during the apply stage, the size of the samples will be equal to the testing dataset.
If the sample parameter is set to True:
Samples are taken from the fitted density function.
Results output in a new column called SampledValue.
Sampled values only come from the inlier region of the distribution.
The full_sample parameter can be used during fit or apply stages.
The full_sample parameter default is False.
If the full_sample parameter is set to True during the fit stage, the size of the samples will be equal to the training dataset.
If the full_sample parameter is set to True during the apply stage, the size of the samples will be equal to the testing dataset.
If the full_sample parameter is set to True:
Samples are taken from the fitted density function.
Results output in a new column called FullSampledValue.
Sampled values come from the whole distribution (both inlier and outlier regions).
Use the summary command to inspect the model.
The values reported for the mean and standard deviation are either the statistics of the fitted distribution, or of the data, depending on the type of the distribution.
In the case of parametric distributions (Normal, Beta, and Exponential) the mean and standard deviation are calculated from the fitted distribution. When the parametric distribution is not a good fit for the data, the reported mean and std might not be close to that of data.
In the case of non-parametric distributions (Gaussian KDE) the mean and standard deviation are calculated from the data passed in during fit.
Version 4.4.0 of the MLTK and above support min and max values in summary.
The min value is the minimum value of the dataset on which the density function is fitted.
The max value is the maximum value of the dataset on which the density function is fitted.
The cardinality value generated by the summary command represents the number of data points used when fitting the selected density function.
The distance value generated by the summary command represents the metric type used when calculating the distance as well as the distance between the sampled data points from the density function and the training dataset.
The mean value generated by the summary command is the mean of the density function.
The value for std generated by the summary command represents the standard deviation of the density function.
A value under other represents any parameters other than mean and std as applicable. In the case of Gaussian KDE, other could show parameter size or bandwidth.
The type field generated by the summary command shows both the chosen density function as well as if the dist parameter is set to auto.
The show_density parameter default is False. If the parameter is set to True, the density of each data point will be provided as output in a new field called ProbabilityDensity.
The output for ProbabilityDensity is the probability density of the data point according to the fitted probability density. This output is provided when the show_density parameter is set to True.
The fit command will fit a probability density function over the data, optionally store the resulting distribution's parameters in a model file, and output the outlier in a new field called IsOutlier.
The output for IsOutlier is a list of labels. Number 1 represents outliers, and 0 represents inliers, assigned to each data point. Outliers are detected based on the values set for the threshold parameter. Inspect the IsOutlier results column to see how well the outlier detection is performing.
The parameters threshold, lower_threshold, and upper_threshold control the outlier detection process.
The threshold parameter is the center of the outlier detection process. It represents the percentage of the area under the density function and has a value between 0.000000001 (refers to ~0%) and 1 (refers to 100%). The threshold parameter guides the DensityFunction algorithm to mark outlier areas on the fitted distribution. For example, if threshold=0.01, then 1% of the fitted density function will be set as the outlier area.
The threshold parameter default value is 0.01.
The threshold, lower_threshold, and upper_threshold parameters can take multiple values.
Multiple values must be in quotation marks and separated by commas.
In cases of multiple values for threshold, the default maximum is 5. Users with access permissions can change this default maximum under the Settings tab.
In cases of multiple values, you are limited to one type of threshold (threshold,lower_threshold, or upper_threshold).
The output for BoundaryRanges is the boundary ranges of outliers on the density function which are set according to the values of the threshold parameter.
Each boundary region has three values: boundary opening point, boundary closing point, and percentage of boundary region.
The boundary region syntax follows the convention of a multi-value field where each boundary region appears in a new line:
first_boundary_region
second_boundary_region
n_th_boundary_region
When multiple thresholds are provided, Boundary Ranges for each threshold appears in a different column separated with the suffix of _th=and the threshold values:
BoundaryRanges_th=threshold_val_1
first_boundary_region_of_th1
second_boundary_region_of_th1
n_th_boundary_region_of_th1
BoundaryRanges_th=threshold_val_2
first_boundary_region_of_th2
second_boundary_region_of_th2
n_th_boundary_region_of_th2
In cases of a single boundary region, the value for the percentage of boundary region is equal to the threshold parameter value.
In some distributions (for example Gaussian KDE), the sum of outlier areas might not add up to the exact value of threshold parameter value, but will be a close approximation.
BoundaryRanges is calculated as an approximation and will be empty in the following two cases:
Where the density function has a sharp peak from low standard deviation.
When there are a low number of data points.
Data points that are exactly at the boundary opening or closing point are assigned as inliers. An opening or closing point is determined by the density function in use.
Normal density function has left and right boundary regions. Data points on the left of the left boundary closing point, and data points on the right of the right boundary opening point are assigned as outliers.
Exponential density function has one boundary region. Data points on the right of the right boundary opening point are assigned as outliers.
Beta density function has one boundary region. Data points on the left of the left boundary closing point are assigned as outliers.
Gaussian KDE density function can have one or more boundary regions, depending on the number of peaks and dips within the density function. Data points in these boundary regions are assigned as outliers. In cases of boundary regions to the left or right, guidelines from Normal density function apply. As the shape for Gaussian KDE density function can differ from dataset to dataset, you do not consistently observe left and right boundary regions.
The random_state parameter is the seed of the pseudo random number generator to use when creating the model. This parameter is optional but the value must be an integer.
The random_state parameter is available in MLTK version 5.0.0 or higher. This parameter is not supported in version 4.5.0 of the MLTK.

Syntax

| fit DensityFunction <field> [by ""<field1>[,<field2>,....<field5>]""] [into <model name>] [dist=<str>] [show_density=true|false] [sample=true|false][full_sample=true|false][threshold=<float>|lower_threshold=<float>|upper_threshold=<float>] [metric=<str>] [random_state=<int>] [partial_fit=<true|false>]
You can apply the saved model to new data with the apply command, with the option to update the parameters for threshold, lower_threshold, upper_threshold, and show_density. Parameters for dist and metric cannot be applied at this stage, and any new values provided will be ignored.

apply <model name> [threshold=<float>|lower_threshold=<float>|upper_threshold=<float>] [show_density=true|false][sample=true|false][full_sample=true|false]
You can inspect the model learned by DensityFunction with the summary command. Version 4.4.0 of the MLTK or above supports min and max values in the summary command.

| summary <model name>
Syntax constraints

Fields within the by clause must be given in quotation marks.
The maximum number of fields within the by clause is 5.
The total number of groups calculated with the by clause can not exceed 1024. In an example clause of by ""DayOfWeek,HourOfDay"" there are two fields: one for DayOfWeek and one for HourOfDay. As there are seven days in a week, there are seven groups for DayOfWeek. As there are twenty-four hours in a day, there are twenty-four groups for HourOfDay. Meaning the total number of groups calculated with the by clause is 7*24= 168.
The limited number of groups prevents model files from growing too large. You can increase the limit by changing the value of max_groups in the DensityFunction settings. Larger limits mean larger model files and longer load times when running apply.
Decrease max_kde_parameter_size to allow for the increase of max_groups. This change keeps model sizes small while allowing for increased groups.
Field names used within the by clause that match any one of the reserved summary field names, produces an error. You must rename your field(s) used within the by clause to fix the error. Reserved summary field names include: type, min, max, mean, std, cardinality, distance, and other.
The parameters threshold, lower_threshold, and upper_threshold must be within the range of 0.00000001 to 1.
If the parameters of lower_threshold and upper_threshold are both provided, the summation of these parameters must be less than 1 (100%).
The threshold and lower_threshold / upper_threshold parameters can not be specified together.
The threshold, lower_threshold, and upper_threshold parameters can take multiple values but in these cases you are limited to one type of threshold (threshold,lower_threshold, or upper_threshold).
Exponential density function only supports threshold and upper_threshold.
Exponential density function supports using lower_threshold but results in empty Boundary regions and 0 outliers.
Normal density function supports either threshold or lower_threshold / upper_threshold.
Gaussian KDE density function supports either threshold or lower_threshold / upper_threshold.
The parameters lower_threshold and upper_threshold can be used with any density function including auto.
Exponential density function supports using lower_threshold but results in empty Boundary regions and 0 outliers.
If you use the summary command to inspect a model created in version 4.3.0 of the MLTK or earlier (prior to the support of min and max), approximate values for min and max are used.
Examples

The following example shows DensityFunction on a dataset with the fit command.

| inputlookup call_center.csv
| eval _time=strptime(_time, ""%Y-%m-%dT%H:%M:%S"")
| bin _time span=15m
| eval HourOfDay=strftime(_time, ""%H"")
| eval BucketMinuteOfHour=strftime(_time, ""%M"")
| eval DayOfWeek=strftime(_time, ""%A"")
| stats max(count) as Actual by HourOfDay,BucketMinuteOfHour,DayOfWeek,source,_time
| fit DensityFunction Actual by ""HourOfDay,BucketMinuteOfHour,DayOfWeek"" into mymodel
This image of the MLTK shows the Statistics tab with many results listed. The fit command is included in the SPL written in the search string. Both numeric and categorical values are listed under columns including hour of day, day of week, source, and time.

The following example shows DensityFunction on a dataset with the apply command.

| inputlookup call_center.csv
| eval _time=strptime(_time, ""%Y-%m-%dT%H:%M:%S"")
| bin _time span=15m
| eval HourOfDay=strftime(_time, ""%H"")
| eval BucketMinuteOfHour=strftime(_time, ""%M"")
| eval DayOfWeek=strftime(_time, ""%A"")
| stats max(count) as Actual by HourOfDay,BucketMinuteOfHour,DayOfWeek,source,_time
| apply mymodel show_density=True sample=True
This image of the toolkit shows the Statistics tab with many results listed. The apply command as well as the sample command are included in the SPL written in the search string. Both numeric and categorical values are listed under columns including hour of day, day of week, source, time, and sampled value.

The following example shows DensityFunction on a dataset with the summary command. This example includes min and max values, which are supported in version 4.4.0 and above of the MLTK.

| summary mymodel
This image of the toolkit shows the Statistics tab with many results listed. The summary command is included in the SPL written in the search string. Both numeric and categorical values are listed under columns including bucket minute of hour, cardinality, mean, std, type, min, and max.

The following example shows BoundaryRages on a test set. In this example the threshold is set to 30% (0.3). The first row has a left boundary range which starts at -Infinity and goes up to the number 44.6912. The area of the left boundary range is 15% of the total area under the density function. It has also a right boundary range which starts at a number 518.3088 and goes up to Infinity. Again, the area of the right boundary range is the same as the left boundary range with 15% of the total area under the density function. The areas of right and left boundary ranges add up to the threshold value of 30%. The third row has only one boundary range which starts at number 300.0943 and goes up to Infinity. The area of the boundary range is 30% of the area under the density function.

| inputlookup call_center.csv
| eval _time=strptime(_time, ""%Y-%m-%dT%H:%M:%S"")
| bin _time span=15m
| eval HourOfDay=strftime(_time, ""%H"")
| eval BucketMinuteOfHour=strftime(_time, ""%M"")
| eval DayOfWeek=strftime(_time, ""%A"")
| stats max(count) as Actual by HourOfDay, BucketMinuteOfHour, DayOfWeek, source, _time
| fit DensityFunction Actual by ""HourOfDay, BucketMinuteOfHour, DayOfWeek"" threshold=0.3 into mymodel
This image of the toolkit shows the Statistics tab with results listed in columns that include hour of day, source, time, Actual, Boundary Ranges, and Is Outlier.

LocalOutlierFactor
The LocalOutlierFactor algorithm uses the scikit-learn Local Outlier Factor (LOF) to measure the local deviation of density of a given sample with respect to its neighbors. LocalOutlierFactor performs one-shot learning and is limited to fitting on training data and returning outliers. LocalOutlierFactor is an unsupervised outlier detection method. The anomaly score depends on how isolated the object is with respect to its neighbors.

For descriptions of the n_neighbors, leaf_size and other parameters, see the sci-kit learn documentation: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html

Using the LocalOutlierFactor algorithm requires running version 1.3 or above of the Python for Scientific Computing add-on.

Parameters

The anomaly_score parameter default is True. Disable this default by adding the False keyword to the command.
The n_neighbors parameter default is 20
The leaf_size parameter default is 30
The p parameter is limited to p >=1
The contamination parameter must be within the range of 0.0 (not included) to 0.5 (included)
The contamination parameter default is 0.1
Options for the algorithm parameter include: brute, kd_tree, ball_tree, and auto. The default is auto.
The brute, kd_tree, ball_tree, and auto algorithm options have respective valid metrics. The default metric for each is minkowski.
Valid metrics for brute include: cityblock, euclidean, l1, l2, manhattan, chebyshev, minkowski, braycurtis, canberra, dice, hamming, jaccard, kulsinski, matching, rogerstanimoto, russellrao, sokalmichener, sokalsneath, cosine, correlation, sqeuclidean, and yule.
Valid metrics for kd_tree include: cityblock, euclidean, l1, l2, manhattan, chebyshev, and minkowski.
Valid metrics for ball_tree include: cityblock, euclidean, l1, l2, manhattan, chebyshev, minkowski, braycurtis, canberra, dice, hamming, jaccard, kulsinski, matching, rogerstanimoto, russellrao, sokalmichener, and sokalsneath.
The output for LocalOutlierFactor is a list of labels titled is_outlier, assigned 1 for outliers, and -1 for inliers
Syntax

fit LocalOutlierFactor <fields>
[n_neighbors=<int>] [leaf_size=<int>] [p=<int>]
[contamination=<float>]
[metric=<str>] [algorithm=<str>] [anomaly_score=<true|false>] 
Syntax constraints

You cannot save LocalOutlierFactor models using the into keyword. This algorithm does not support saving models and you cannot apply a saved model to new data.
LOF does not include the predict method.
Example

The following example uses LocalOutlierFactor on a test set.

| inputlookup iris.csv | fit LocalOutlierFactor petal_length petal_width n_neighbors=10 algorithm=kd_tree metric=minkowski p=1 contamination=0.14 leaf_size=10
OneClassSVM
The OneClassSVM algorithm uses the scikit-learn OneClassSVM to fit a model from a set of features or fields for detecting anomalies and outliers, where features are expected to contain numerical values. OneClassSVM is an unsupervised outlier detection method.

For further information, see the sci-kit learn documentation: http://scikit-learn.org/stable/modules/svm.html#kernel-functions

Parameters

The kernel parameter specifies the kernel type for using in the algorithm, where the default value is kernel is rbf.
Kernel types include: linear, rbf, poly, and sigmoid.
You can specify the upper bound on the fraction of training error as well as the lower bound of the fraction of support vectors using the nu parameter, where the default value is 0.5.
The degree parameter is ignored by all kernels except the polynomial kernel, where the default value is 3.
gamma is the kernel co-efficient that specifies how much influence a single data instance has, where the default value is 1/ number of features.
The independent term of coef0 in the kernel function is only significant if you have polynomial or sigmoid function.
The term tol is the tolerance for stopping criteria.
The shrinking parameter determines whether to use the shrinking heuristic.
Syntax

fit OneClassSVM <fields> [into <model name>]
[kernel=<str>] [nu=<float>] [coef0=<float>]
[gamma=<float>] [tol=<float>] [degree=<int>] [shrinking=<true|false>]
You can save OneClassSVM models using the into keyword.
You can apply the saved model later to new data with the apply command.
Syntax constraints

After running the fit or apply command, a new field named isNormal is generated. This field defines whether a particular record (row) is normal (isNormal=1) or anomalous (isNormal=-1).
You cannot inspect the model learned by OneClassSVM with the summary command.
Example

The following example uses OneClassSVM on a test set.

... | fit OneClassSVM * kernel=""poly"" nu=0.5 coef0=0.5 gamma=0.5 tol=1 degree=3 shrinking=f into
TESTMODEL_OneClassSVM
Classifiers
Classifier algorithms predict the value of a categorical field.

The kfold cross-validation command can be used with all Classifier algorithms. For details, see K-fold cross-validation.

AutoPrediction
AutoPrediction automatically determines the data type as categorical or numeric. AutoPrediction then invokes the RandomForestClassifier algorithm to carry out the prediction. For further details, see RandomForestClassifier. AutoPrediction also executes the data split for training and testing during the fit process, eliminating the need for a separate command or macro. AutoPrediction uses particular cases to determine the data type, and uses the train_test_split function from sklearn to perform the data split.

Parameters

Use the target_type parameter to specify the target field as numeric or categorical.
The target_type parameter default is auto. When auto is used, AutoPrediction automatically determines the target field type.
AutoPrediction uses the following data types to determine the target_type field as categorical:
Data of type bool, str, or numpy.object
Data of type int and the criterion option is specified
AutoPrediction determines the target_type field as numeric for all other cases.
The test_split_ratio specifies the splitting of data for model training and model validation. Value must be a float between 0 (inclusive) and 1 (exclusive).
The test_split_ratio default is 0. A value of 0 means all data points get used to train the model.
A test_split_ratio value of 0.3, for example, means 30% for the data points get used for testing and 70% are used for training.
Use n_estimators to optionally specify the number of trees.
Use max_depth to optionally set the maximum depth of the tree.
Specify the criterion value for classification (categorical) scenarios.
Ignore the criterion value for regression (numeric) scenarios.
Syntax

fit AutoPrediction Target from Predictors* into PredictorModel target_type=<auto|numeric|categorical> test_split_ratio=<[0-1]>[n_estimators=<int>] [max_depth=<int>] 
[criterion=<gini | entropy>] [random_state=<int>][max_features=<str>] [min_samples_split=<int>] [max_leaf_nodes=<int>]
You can save AutoPrediction models using the into keyword and apply the saved model later to new data using the apply command.

 ... | apply PredictorModel
You can inspect the model learned by AutoPrediction with the summary command.

 .... | summary PredictorModel
Syntax constraints

AutoPrediction does not support partial_fit.
Classification performance output columns for accuracy, f1, precision, and recall only appear if the target_type is categorical.
Regression performance output columns for RMSE and rSquared only appear if the target_type is numeric.
Example

The following example uses AutoPrediction on a test set.

| fit AutoPrediction random_state=42 species from * max_features=0.1 into auto_classify_model test_split_ratio=0.3 random_state=42
BernoulliNB
The BernoulliNB algorithm uses the scikit-learn BernoulliNB estimator to fit a model to predict the value of categorical fields where explanatory variables are assumed to be binary-valued. BernoulliNB is an implementation of the Naive Bayes classification algorithm. This algorithm supports incremental fit.

Parameters

The alpha parameter controls Laplace/ Lidstone smoothing. The default value is 1.0.
The binarize parameter is a threshold that can be used for converting numeric field values to the binary values expected by BernoulliNB. The default value is 0.
If binarize=0 is specified, the default, values > 0 are assumed to be 1, and values <= 0 are assumed to be 0.
The fit_prior Boolean parameter specifies whether to learn class prior probabilities. The default value is True. If fit_prior=f is specified, classes are assumed to have uniform popularity.
Syntax

fit BernoulliNB <field_to_predict> from <explanatory_fields> [into <model name>]
[alpha=<float>] [binarize=<float>] [fit_prior=<true|false>] [partial_fit=<true|false>]
You can save BernoulliNB models using the into keyword and apply the saved model later to new data using the apply command.

 ... | apply TESTMODEL_BernoulliNB
You can inspect the model learned by BernoulliNB with the summary command as well as view the class and log probability information as calculated by the dataset.

 .... | summary My_Incremental_Model
Syntax constraints

The partial_fit parameter controls whether an existing model should be incrementally updated or not. The default value is False, meaning it will not be incrementally updated. Choosing partial_fit=True allows you to update an existing model using only new data without having to retrain it on the full training data set.
Using partial_fit=True on an existing model ignores the newly supplied parameters. The parameters supplied at model creation are used instead. If partial_fit=False or partial_fit is not specified (default is False), the model specified is created and replaces the pre-trained model if one exists.
If My_Incremental_Model does not exist, the command saves the model data under the model name My_Incremental_Model. If My_Incremental_Model exists and was trained using BernoulliNB, the command updates the existing model with the new input. If My_Incremental_Model exists but was not trained by BernoulliNB, an error message displays.
Example

The following example uses BernoulliNB on a test set.

... | fit BernoulliNB type from * into TESTMODEL_BernoulliNB alpha=0.5 binarize=0 fit_prior=f
DecisionTreeClassifier
The DecisionTreeClassifier algorithm uses the scikit-learn DecisionTreeClassifier estimator to fit a model to predict the value of categorical fields. For further information, see the sci-kit learn documentation: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html.

Parameters

To specify the maximum depth of the tree to summarize, use the limit argument. The default value for the limit argument is 5.

 ... | summary model_DTC limit=10
Syntax

fit DecisionTreeClassifier <field_to_predict> from <explanatory_fields> [into <model_name>] 
[max_depth=<int>] [max_features=<str>] [min_samples_split=<int>] [max_leaf_nodes=<int>] 
[criterion=<gini|entropy>] [splitter=<best|random>] [random_state=<int>]
You can save DecisionTreeClassifier models by using the into keyword and apply it to new data later by using the apply command.

... | apply model_DTC
You can inspect the decision tree learned by DecisionTreeClassifier with the summary command.

 ... | summary model_DTC
See a JSON representation of the tree by giving json=t as an argument to the summary command.

 ... | summary model_DTC json=t
Example

The following example uses DecisionTreeClassifier on a test set.

... | fit DecisionTreeClassifier SLA_violation from * into sla_model | ...
GaussianNB
The GaussianNB algorithm uses the scikit-learn GaussianNB estimator to fit a model to predict the value of categorical fields, where the likelihood of explanatory variables is assumed to be Gaussian. GaussianNB is an implementation of Gaussian Naive Bayes classification algorithm. This algorithm supports incremental fit.

Parameters

The partial_fit parameter controls whether an existing model should be incrementally updated or not. This allows you to update an existing model using only new data without having to retrain it on the full training data set.
The partial_fit parameter default is False.
Syntax

fit GaussianNB <field_to_predict> from <explanatory_fields> [into <model name>] [partial_fit=<true|false>]
You can save GaussianNB models using the into keyword and apply the saved model later to new data using the apply command.

... | apply TESTMODEL_GaussianNB
You can inspect models learned by GaussianNB with the summary command.

 ... | summary My_Incremental_Model
Syntax constraints

If My_Incremental_Model does not exist, the command saves the model data under the model name My_Incremental_Model. If My_Incremental_Model exists and was trained using GaussianNB, the command updates the existing model with the new input. If My_Incremental_Model exists but was not trained by GaussianNB, an error message is thrown.
If partial_fit=False or partial_fit is not specified the model specified is created and replaces the pre-trained model if one exists.
Example

The following example uses GaussianNB on a test set.

... | fit GaussianNB species from * into TESTMODEL_GaussianNB
The following example includes the partial_fit command.

| inputlookup iris.csv | fit GaussianNB species from * partial_fit=true into My_Incremental_Model
GradientBoostingClassifier
This algorithm uses the GradientBoostingClassifier from scikit-learn to build a classification model by fitting regression trees on the negative gradient of a deviance loss function. For further information, see the sci-kit learn documentation: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html.

Syntax

fit GradientBoostingClassifier <field_to_predict> from <explanatory_fields>[into <model name>]  
[loss=<deviance | exponential>] [max_features=<str>] 
[learning_rate =<float>] [min_weight_fraction_leaf=<float>] [n_estimators=<int>] 
[max_depth=<int>] [min_samples_split =<int>] [min_samples_leaf=<int>] 
[max_leaf_nodes=<int>] [random_state=<int>]
You can apply the saved model later to new data using the apply command.

... | apply TESTMODEL_GradientBoostingClassifier
You can inspect features learned by GradientBoostingClassifier with the summary command.

 ... | summary TESTMODEL_GradientBoostingClassifier 
Example

The following example uses GradientBoostingClassifier on a test set.

... | fit GradientBoostingClassifier target from * into TESTMODEL_GradientBoostingClassifier  
LogisticRegression
The LogisticRegression algorithm uses the scikit-learn LogisticRegression estimator to fit a model to predict the value of categorical fields.

Parameters

The fit_intercept parameter specifies whether the model includes an implicit intercept term.
The default value of the fit_intercept parameter is True.
The probabilities parameter specifies whether probabilities for each possible field value should be returned alongside the predicted value.
The default value of the probabilities parameter is False.
Syntax

fit LogisticRegression <field_to_predict> from <explanatory_fields> [into <model name>]
[fit_intercept=<true|false>] [probabilities=<true|false>]
You can save LogisticRegression models using the into keyword and apply new data later using the apply command.

... | apply sla_model
You can inspect the coefficients learned by LogisticRegression with the summary command.

 ... | summary sla_model
Example

The following examples uses LogisticRegression on a test set.

... | fit LogisticRegression SLA_violation from IO_wait_time into sla_model | ...
MLPClassifier
The MLPClassifier algorithm uses the scikit-learn Multi-layer Perceptron estimator for classification. MLPClassifier uses a feedforward artificial neural network model that trains using backpropagation. This algorithm supports incremental fit.

For descriptions of the batch_size , random_state and max_iter parameters, see the scikit-learn documentation at http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

Using the MLPClassifier algorithm requires running version 1.3 or above of the Python for Scientific Computing add-on.

Parameters

The partial_fit parameter controls whether an existing model should be incrementally updated on not. This allows you to update an existing model using only new data without having to retrain it on the full training data set.
The partial_fit parameter default is False.
The hidden_layer_sizes parameter format (int) varies based on the number of hidden layers in the data.
Syntax

fit MLPClassifier <field_to_predict> from <explanatory_fields> [into <model name>]
[batch_size=<int>] [max_iter=<int>] [random_state=<int>] [hidden_layer_sizes=<int>-<int>-<int>]
[activation=<str>] [solver=<str>] [learning_rate=<str>]
[tol=<float>} {momentum=<float>]
You can save MLPClassifier models by using the into keyword and apply it to new data later by using the apply command.

You can inspect models learned by MLPClassifier with the summary command.

 ... | summary My_Example_Model
Syntax constraints

If My_Example_Model does not exist, the model is saved to it.
If My_Example_Model exists and was trained using MLPClassifier, the command updates the existing model with the new input.
If My_Example_Model exists but was not trained using MLPClassifier, an error message displays.
Example

The following example uses MLPClassifier on a test set.

... | inputlookup diabetes.csv | fit MLPClassifier response from * into MLP_example_model hidden_layer_sizes='100-100-80' |...
The following example includes the partial_fit command.

| inputlookup iris.csv | fit MLPClassifier species from * partial_fit=true into My_Example_Model
RandomForestClassifier
The RandomForestClassifier algorithm uses the scikit-learn RandomForestClassifier estimator to fit a model to predict the value of categorical fields.

For descriptions of the n_estimators, max_depth, criterion, random_state, max_features, min_samples_split, and max_leaf_nodes parameters, see the scikit-learn documentation at http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html.

Syntax

fit RandomForestClassifier <field_to_predict> from <explanatory_fields> [into <model name>]
[n_estimators=<int>] [max_depth=<int>] [criterion=<gini | entropy>] [random_state=<int>]
[max_features=<str>] [min_samples_split=<int>] [max_leaf_nodes=<int>]
You can save RandomForestClassifier models using the into keyword and apply new data later using the apply command.

... | apply sla_model
You can list the features that were used to fit the model, as well as their relative importance or influence with the summary command.

 ... | summary sla_model
Example

The following example uses RandomForestClassifier on a test set.

... | fit RandomForestClassifier SLA_violation from * into sla_model | ...
SGDClassifier
The SGDClassifier algorithm uses the scikit-learn SGDClassifier estimator to fit a model to predict the value of categorical fields. This algorithm supports incremental fit.

Parameters

The partial_fit parameter controls whether an existing model should be incrementally updated or not. This allows you to update an existing model using only new data without having to retrain it on the full training data set.
The partial_fit parameter default is False.
n_iter=<int> is the number of passes over the training data also known as epochs. The default is 5. The number of iterations is set to 1 if using partial_fit.
The loss=<hinge|log|modified_huber|squared_hinge|perceptron> parameter is the loss function to be used.
Defaults to hinge, which gives a linear SVM.
The log loss gives logistic regression, a probabilistic classifier.
modified_huber is another smooth loss that brings tolerance to outliers as well as probability estimates.
squared_hinge is like hinge but is quadratically penalized.
perceptron is the linear loss used by the perceptron algorithm.
The fit_intercept=<true|false> parameter specifies whether the intercept should be estimated or not. The default is True.
penalty=<l2|l1|elasticnet> is the penalty, also known as regularization term, to be used. The default is l2.
learning_rate=<constant|optimal|invscaling> is the learning rate.
constant: eta = eta0
optimal: eta = 1.0/(alpha * t)
invscaling: eta = eta0 / pow(t, power_t)
The default is invscaling
l1_ratio=<float> is the Elastic Net mixing parameter, with 0 <= l1_ratio <= 1 (default 0.15).
l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
alpha=<float> is the constant that multiplies the regularization term (default 0.0001). Also used to compute learning_rate when set to optimal.
eta0=<float> is the initial learning rate. The default is 0.01.
power_t=<float> is the exponent for inverse scaling learning rate. The default is 0.25.
random_state=<int> is the seed of the pseudo random number generator to use when shuffling the data.
Syntax

fit SGDClassifier <field_to_predict> from <explanatory_fields>
[into <model name>] [partial_fit=<true|false>]
[loss=<hinge|log|modified_huber|squared_hinge|perceptron>]
[fit_intercept=<true|false>]
[random_state=<int>] [n_iter=<int>] [l1_ratio=<float>]
[alpha=<float>] [eta0=<float>] [power_t=<float>]
[penalty=<l1|l2|elasticnet>] [learning_rate=<constant|optimal|invscaling>] 
You can save SGDClassifier models using the into keyword and apply the saved model later to new data using the apply command.

... | apply sla_model
You can inspect the model learned by SGDClassifier with the summary command.

 ... | summary sla_model
Syntax constraints

If My_Incremental_Model does not exist, the command saves the model data under the model name My_Incremental_Model.
If My_Incremental_Model exists and was trained using SGDClassifier, the command updates the existing model with the new input.
If My_Incremental_Model exists but was not trained by SGDClassifier, an error displays.
Using partial_fit=true on an existing model ignores the newly supplied parameters. The parameters supplied at model creation are used instead.
If partial_fit=false or partial_fit is not specified the model specified is created and replaces the pre-trained model if one exists.
Example

The following example uses SGDClassifier on a test set.

... | fit SGDClassifier SLA_violation from * into sla_model 
The following example includes the partial_fit=<true|false> command.

| inputlookup iris.csv | fit SGDClassifier species from * partial_fit=true into My_Incremental_Model
SVM
The SVM algorithm uses the scikit-learn kernel-based SVC estimator to fit a model to predict the value of categorical fields. It uses the radial basis function (rbf) kernel by default. For descriptions of the C and gamma parameters, see the scikit-learn documentation at http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html.

Kernel-based methods such as the scikit-learn SVC tend to work best when the data is scaled, for example, using our StandardScaler algorithm: | fit StandardScaler into scaling_model | fit SVM from into svm_model. For details, see ''A Practical Guide to Support Vector Classification'' at https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf.

Parameters

The gamma parameter controls the width of the rbf kernel. The default value is 1 /number of fields.
The C parameter controls the degree of regularization when fitting the model. The default value is 1.0.
Syntax

fit SVM <field_to_predict> from <explanatory_fields> [into <model name>] [C=<float>] [gamma=<float>]
You can save SVM models using the into keyword and apply new data later using the apply command.

... | apply sla_model
Syntax constraints

You cannot inspect the model learned by SVM with the summary command.

Example

The following example uses SVM on a test set.

... | fit SVM SLA_violation from * into sla_model | ...
Clustering Algorithms
Clustering is the grouping of data points. Results will vary depending upon the clustering algorithm used. Clustering algorithms differ in how they determine if data points are similar and should be grouped. For example, the K-means algorithm clusters based on points in space, whereas the DBSCAN algorithm clusters based on local density.

Birch
The Birch algorithm uses the scikit-learn Birch clustering algorithm to divide data points into set of distinct clusters. The cluster for each event is set in a new field named cluster. This algorithm supports incremental fit.

Parameters

The k parameter specifies the number of clusters to divide the data into after the final clustering step, which treats the sub-clusters from the leaves of the CF tree as new samples.
By default, the cluster label field name is cluster. Change that behavior by using the as keyword to specify a different field name.
The partial_fit parameter controls whether an existing model should be incrementally updated on not. This allows you to update an existing model using only new data without having to retrain it on the full training data set.
The partial_fit parameter default is False.
Syntax

fit Birch <fields> [into <model name>] [k=<int>][partial_fit=<true|false>] [into <model name>] 
You can save Birch models using the into keyword and apply new data later using the apply command.

... | apply Birch_model
Syntax constraints

If My_Incremental_Model does not exist, the command saves the model data under the model name My_Incremental_Model.
If My_Incremental_Model exists and was trained using Birch, the command updates the existing model with the new input.
If My_Incremental_Model exists but was not trained by Birch, an error message displays.
Using partial_fit=true on an existing model ignores the newly supplied parameters. The parameters supplied at model creation are used instead.
If partial_fit=false or partial_fit is not specified the model specified is created and replaces the pre-trained model if one exists.
You cannot inspect the model learned by Birch with the summary command.
Examples

The following example uses Birch on a test set.

... | fit Birch * k=3 | stats count by cluster
The following example includes the partial_fit command.

| inputlookup track_day.csv | fit Birch * k=6 partial_fit=true into My_Incremental_Model
DBSCAN
The DBSCAN algorithm uses the scikit-learn DBSCAN clustering algorithm to divide a result set into distinct clusters. The cluster for each event is set in a new field named cluster. DBSCAN is distinct from K-Means in that it clusters results based on local density, and uncovers a variable number of clusters, whereas K-Means finds a precise number of clusters. For example, k=5 finds 5 clusters.

Parameters

The eps parameter specifies the maximum distance between two samples for them to be considered in the same cluster.
By default, the cluster label field name is cluster. Change that behavior by using the as keyword to specify a different field name.
The min_samples parameter defines the number of samples, or the total weight, in a neighborhood for a point to be considered as a core point - including the point itself. You can choose the min_samples parameter's best value based on preference for cluster density or noise in your dataset.
The min_samples parameter is optional.
The min_samples default value is 5.
The minimum value for the min_samples parameter is 3.
If min_samples=8 you need at least 8 data points to form a dense cluster.
If you choose the min_samples parameter's best value based on noise in your dataset, it's recommended to have a larger data set to pull from.

Syntax

| fit DBSCAN <fields> [eps=<number>] [min_samples=<integer>]
Syntax constraints

You cannot save DBSCAN models using the into keyword. To predict cluster assignments for future data, combine the DBSCAN algorithm with any classifier algorithm. For example, first cluster the data using DBSCAN, then fit RandomForestClassifier to predict the cluster.

Examples

The following example uses DBSCAN without the min_samples parameter.

... | fit DBSCAN * | stats count by cluster
The following example uses DBSCAN with the min_samples parameter.

...| inputlookup track_day.csv | fit DBSCAN eps=0.5 min_samples=1000 speed | table speed cluster
G-means
G-means is a clustering algorithm based on K-means. The G-means algorithm is similar in purpose to the X-means algorithm. G-means uses the Anderson-Darling statistical test to determine when to split a cluster.

Using the G-means algorithm has the following advantages:

The parameter k is computed automatically
G-means can produce more accurate clusters than X-means in some real-world scenarios
Parameters

The cluster splitting decision is done using the Anderson-Darling statistical test.
The cluster for each event is set in a new field named cluster, and the total number of clusters is set in a new field named n_clusters.
By default, the cluster label field name is cluster.
You can change the default behavior by using the as keyword to specify a different field name.
Optionally use the random_state parameter to set a seed value.
random_state must be an integer.
Syntax

| fit GMeans <fields> [into <cluster_model>]
You can apply new data to the saved G-means model using the apply command.

 ... | apply cluster_model 
You can save G-means models using the into command. You can inspect the model learned by G-means with the summary command.

...| summary cluster_model 
Example

The following example uses G-means on a test set.

| inputlookup housing.csv
| fields median_house_value distance_to_employment_center crime_rate
| fit GMeans * random_state=42 into cluster_model
K-means
K-means clustering is a type of unsupervised learning. It is a clustering algorithm that groups similar data points, with the number of groups represented by the variable k. The K-means algorithm uses the scikit-learn K-means implementation. The cluster for each event is set in a new field named cluster. Use the K-means algorithm when you have unlabeled data and have at least approximate knowledge of the total number of groups into which the data can be divided.

Using the K-means algorithm has the following advantages:

Computationally faster than most other clustering algorithms.
Simple algorithm to explain and understand.
Normally produces tighter clusters than hierarchical clustering.
Using the K-means algorithm has the following disadvantages:

Difficult to determine optimal or true value of k. See X-means.
Sensitive to scaling. See StandardScaler.
Each clustering may be slightly different, unless you specify the random_state parameter.
Does not work well with clusters of different sizes and density.
For descriptions of default value of K, see the scikit-learn documentation at http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

Parameters

The k parameter specifies the number of clusters to divide the data into. By default, the cluster label field name is cluster. Change that behavior by using the as keyword to specify a different field name.

Syntax

fit KMeans <fields> [into <model name>]  [k=<int>]  [random_state=<int>] 
You can save K-means models using the into keyword when using the fit command.

You can apply the model to new data using the apply command.

... | apply cluster_model
You can inspect the model using the summary command.

... | summary cluster_model
Example

The following example uses K-means on a test set.

... | fit KMeans * k=3 | stats count by cluster
SpectralClustering
The SpectralClustering algorithm uses the scikit-learn SpectralClustering clustering algorithm to divide a result set into set of distinct clusters. SpectralClustering first transforms the input data using the Radial Basis Function (rbf) kernel, and then performs K-Means clustering on the result. Consequently, SpectralClustering can learn clusters with a non-convex shape. The cluster for each event is set in a new field named cluster.

Parameters

The k parameter specifies the number of clusters to divide the data into after kernel step. By default, the cluster label field name is cluster. Change that behavior by using the as keyword to specify a different field name.

Syntax

fit SpectralClustering <fields> [k=<int>] [gamma=<float>] [random_state=<int>]
Syntax constraints

You cannot save SpectralClustering models using the into keyword. If you want to be able to predict cluster assignments for future data, you can combine the SpectralClustering algorithm with any clustering algorithm. For example, first cluster the data using SpectralClustering, then fit a classifier to predict the cluster using RandomForestClassifier.

Example

The following example uses SpectralClustering on a test set.

... | fit SpectralClustering * k=3 | stats count by cluster
X-means
Use the X-means algorithm when you have unlabeled data and no prior knowledge of the total number of labels into which that data could be divided. The X-means clustering algorithm is an extended K-means that automatically determines the number of clusters based on Bayesian Information Criterion (BIC) scores. Starting with a single cluster, the X-means algorithm goes into action after each run of K-means, making local decisions about which subset of the current centroids should split themselves in order to fit the data better.

Using the X-means algorithm has the following advantages:

Eliminates the requirement of having to provide the value of k.
Normally produces tighter clusters than hierarchical clustering.
Using the X-means algorithm has the following disadvantages:

Sensitive to scaling. See StandardScaler.
Different initializations might result in different final clusters.
Does not work well with clusters of different sizes and density.
Parameters

The splitting decision is done by computing the BIC.
The cluster for each event is set in a new field named cluster, and the total number of clusters is set in a new field named n_clusters.
By default, the cluster label field name is cluster.
You can change the default behavior by using the as keyword to specify a different field name.
Syntax

fit XMeans <fields> [into <model name>]
You can apply new data to the saved X-means model using the apply command.

 ... | apply cluster_model 
You can save X-means models using the into command. You can inspect the model learned by X-means with the summary command.

...| summary cluster_model 
Example

The following example uses X-means on a test set.

... | fit XMeans * | stats count by cluster
Cross-validation
Cross-validation assesses how well a statistical model generalizes on an independent dataset. Cross-validation tells you how well your machine learning model is expected to perform on data that it has not been trained on. There are many types of cross-validation, but K-fold cross-validation (kfold_cv) is one of the most common.

Cross-validation is typically used for the following machine learning scenarios:

Comparing two or more algorithms against each other for selecting the best choice on a particular dataset.
Comparing different choices of hyper-parameters on the same algorithm for choosing the best hyper-parameters for a particular dataset.
An improved method over a train/test split for quantifying model generalization.
Cross-validation is not well suited for time-series charts:

In situations where the data is ordered such as time-series, cross-validation is not well suited because the training data is shuffled. In these situations, other methods such as Forward Chaining are more suitable.
The most straightforward implementation is to wrap sklearn's Time Series Split. Learn more here: https://en.wikipedia.org/wiki/Forward_chaining
K-fold cross-validation
In the kfold_cv parameter, the training set is randomly partitioned into k equal-sized subsamples. Then, each sub-sample takes a turn at becoming the validation (test) set, predicted by the other k-1 training sets. Each sample is used exactly once in the validation set, and the variance of the resulting estimate is reduced as k is increased. The disadvantage of the kfold_cv parameter is that k different models have to be trained, leading to long execution times for large datasets and complex models.

The scores obtained from K-fold cross-validation are generally a less biased and less optimistic estimate of the model performance than a standard training and testing split.

The image is a representative diagram of how the K-fold parameter works. There are 5 rows representing iterations or folds.  Each fold contains equal subsamples that each take a turn as testing and training data.

You can obtain k performance metrics, one for each training and testing split. These k performance metrics can then be averaged to obtain a single estimate of how well the model generalizes on unseen data.

Syntax

The kfold_cv parameter is applicable to to all classification and regression algorithms, and you can append the command to the end of an SPL search.

Here kfold_cv=<int> specifies that k=<int> folds is used. When you specify a classification algorithm, stratified k-fold is used instead of k-fold. In stratified k-fold, each fold contains approximately the same percentage of samples for each class.

..| fit <classification | regression algo> <targetVariable> from <featureVariables> [options] kfold_cv=<int>
The kfold_cv parameter cannot be used when saving a model.

Output

The kfold_cv parameter returns performance metrics on each fold using the same model specified in the SPL - including algorithm and hyper parameters. Its only function is to give you insight into how well you model generalizes. It does not perform any model selection or hyper parameter tuning.

Examples

The first example shows the kfold_cv parameter used in classification. Where the output is a set of metrics for each fold including accuracy, f1_weighted, precision_weighted, and recall_weighted.

This is a screen capture of the Statistics tab of the toolkit. There are three rows of results displayed.

This second example shows the kfold_cv parameter used in classification. Where the output is a set of metrics for each the neg_mean_squared_error and r^2 folds.

This is a screen capture of the Statistics tab of the toolkit. There are three rows of results displayed.

Feature Extraction
Feature extraction algorithms transform fields for better prediction accuracy.

FieldSelector
The FieldSelector algorithm uses the scikit-learn GenericUnivariateSelect to select the best predictor fields based on univariate statistical tests. For descriptions of the mode and param parameters, see the scikit-learn documentation at http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html.

Parameters

The type parameter specifies if the field to predict is categorical or numeric.

Syntax

fit FieldSelector <field_to_predict> from <explanatory_fields>
[into <model name>] [type=<categorical, numeric>]
[mode=<k_best, fpr, fdr, fwe, percentile>] [param=<int>] 
You can save FieldSelector models using the into keyword and apply new data later using the apply command.

... | apply sla_model
You can inspect the model learned by FieldSelector with the summary command.

 | summary sla_model
Example

The following example uses FieldSelector on a test set.

... | fit FieldSelector type=categorical SLA_violation from * into sla_model | ...
HashingVectorizer
The HashingVectorizer algorithm converts text documents to a matrix of token occurrences. It uses a feature hashing strategy to allow for hash collisions when measuring the occurrence of tokens. It is a stateless transformer, meaning that it does not require building a vocabulary of the seen tokens. This reduces the memory footprint and allows for larger feature spaces.

HashingVectorizer is comparable with the TFIDF algorithm, as they share many of the same parameters. However HashingVectorizer is a better option for building models with large text fields provided you do not need to know term frequencies, and only want outcomes.

For descriptions of the ngram_range, analyzer, norm, and token_pattern parameters, see the scikit-learn documentation at https://scikit-learn.org/0.19/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html

Parameters

The reduce parameter is either True or False and determines whether or not to reduce the output to a smaller dimension using TruncatedSVD.
The reduce parameter default is True.
The k=<int> parameter sets the number of dimensions to reduce when the reduce parameter is set to true. Default is 100.
The default for the max_features parameter is 10,000.
The n_iters parameter specifies the number of iterations to to use when performing dimensionality reduction. This is only used when the reduce parameter is set to True. Default is 5.
Syntax

fit HashingVectorizer <field_to_convert> [max_features=<int>] [n_iters=<int>]
[reduce=<bool>] [k=<int>] [ngram_range=<int>-<int>] [analyzer=<str>] 
[norm=<str>] [token_pattern=<str>] [stop_words=english]
Syntax constraints

HashingVectorizer does not support saving models, incremental fit, or K-fold cross validation.

Example

The following example uses HashingVectorizer to hash the text dataset and applies KMeans clustering (where k=3) on the hashed fields.

| inputlookup authorization.csv | fit HashingVectorizer Logs ngram_range=1-2 k=50 stop_words=english | fit KMeans Logs_hashed* k=3 | fields cluster* Logs | sample 5 by cluster | sort by cluster
ICA
ICA (Independent component analysis) separates a multivariate signal into additive sub-components that are maximally independent. Typically, ICA is not used for separating superimposed signals, but for reducing dimensionality. The ICA model does not include a noise term for the model to be correct, meaning whitening must be applied. Whitening can be done internally using the whiten argument, or manually using one of the PCA variants.

Parameters

The n_components parameters determines the number of components ICA uses.
The n_components parameter is optional.
The n_components parameter default is None. If None is selected, all components are used.
Use the algorithm parameter to apply parallel or deflation algorithm for FastICA.
The the algorithm parameter default is algorithm='parallel' .
Use the whiten parameter to set a noise term.
The whiten parameter is optional.
If the whiten parameter is False no whitening is performed.
The whiten parameter default is True.
The max_iter parameter determines the maximum number of iterations during the running of the fit command.
The max_iter parameter is optional.
The max_iter parameter default is 200.
The fun parameter determines the functional form of the G function used in the approximation to neg-entropy.
The fun parameter is optional.
The fun parameter default is logcosh. Other options for this parameter are exp or cube.
The tol parameter sets the tolerance on update at each iteration.
The tol parameter is optional.
The tol parameter default is 0.0001 .
The random_state parameter sets the seed value used by the random number generator.
The random_state parameter default is None.
If random_state=None then a random seed value is used.
Syntax

fit ICA n_components=<int>, algorithm=<""parallel""|""deflation"">, whiten=<bool>, fun=<""logcosh""|""exp""|""cube"">, max_iter=<int>, tol=<float>, random_state=<int> <explanatory_fields> [into <model name>]
You can save ICA models using the into keyword and apply new data later using the apply command.

Syntax constraints

You cannot inspect the model learned by ICA with the summary command.

Example

The following example shows how ICA is able to find the two original sources of data from two measurements that have mixes of both. As a comparison, PCA is used to show the difference between the two – PCA is not able to identify the original sources.

| makeresults count=2
| streamstats count as count
| eval time=case(count=2,relative_time(now(),""+2d""),count=1,now())
| makecontinuous time span=15m
| eval _time=time
| eval s1 = sin(2*time)
| eval s2 = sin(4*time)
| eval m1 = 1.5*s1 + .5*s2, m2 = .1*s1 + s2
| fit ICA m1, m2 n_components=2 as IC
| fit PCA m1, m2 k=2 as PC
| fields _time, *
| fields - count, time
KernelPCA
The KernelPCA algorithm uses the scikit-learn KernelPCA to reduce the number of fields by extracting uncorrelated new features out of data. The difference between KernelPCA and PCA is the use of kernels in the former, which helps with finding nonlinear dependencies among the fields. Currently, KernelPCA only supports the Radial Basis Function (rbf) kernel.

For descriptions of the gamma, degree, tolerance, and max_iteration parameters, see the scikit-learn documentation at http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html.

Kernel-based methods such as KernelPCA tend to work best when the data is scaled, for example, using our StandardScaler algorithm: | fit StandardScaler into scaling_model | fit KernelPCA into kpca_model. For details, see ''A Practical Guide to Support Vector Classification'' at https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf.

Parameters

The k parameter specifies the number of features to be extracted from the data. The other parameters are for fine tuning of the kernel.

Syntax

fit KernelPCA <fields> [into <model name>]
[degree=<int>] [k=<int>] [gamma=<int>]
[tolerance=<int>] [max_iteration=<int>]
You can save KernelPCA models using the into keyword and apply new data later using the apply command.

... | apply user_feedback_model
Syntax constraints

You cannot inspect the model learned by KernelPCA with the summary command.

Example

The following example uses KernelPCA on a test set.

... | fit KernelPCA * k=3 gamma=0.001 | ...
NPR
The Normalized Perlich Ratio (NPR) algorithm converts high cardinality categorical field values into numeric field entries while intelligently handling space optimization. NPR offers low computational costs to perform feature extraction on variables with high cardinalities such as ZIP codes or IP addresses.

NPR does not perform one-hot encoding unlike other algorithms that leverage the fit and apply commands.

Parameters

Use the summary command to inspect the variance information of the saved model.
After running NPR the transformed dataset has calculated ratios for all feature variables (feature_field). Based on the training data NPR calculates a variable of X_unobserved which can be used as a replacement value in the following two scenarios:
In conjunction with the fit command NPR initially replaces missing values in the dataset for feature_field with the keyword unobserved which is then replaced by the calculated NPR value of X_unobserved.
In conjunction with the apply command, any new value for target_field that was not visible during model training but is encountered in the test dataset.
The number of transformed columns created after running NPR is equal to the number of distinct values for feature_field within the search string.
From the saved model, use the variance output field to examine the contribution of a particular feature towards the accuracy of the prediction. Higher variance indicates highly important categorical values whereas low variance indicates the value being of lower importance towards the target prediction. Variance may assist in the process of discarding irrelevant feature variables.
Syntax

fit NPR <target_field> from <feature_field> [into <model name>]
You can couple NPR with existing MLTK algorithms to feed the transformed results to the model as a means to enhance predictions.

| fit NPR <target_field> from <feature_field> | fit SGDClassifier <target_field> from NPR
You can save NPR models using the into keyword and apply new data later using the apply command.

| input lookup disk_failures.csv | tail 1000 | apply npr_disk
You can inspect the model learned by NPR with the summary command.

 | summary npr_disk
Syntax constraints

The wildcard (*) character is not supported.
The maximum matrix size calculated from |X| * |Y| where X is the feature_field and Y is the target_field is 10000000. For example, if number of distinct categorical feature values are 1000 and distinct categorical target values are 100 then the matrix size is 100000.
Examples

The following example uses NPR on a test set.

| inputlookup disk_failures.csv| head 5000 | fit NPR DiskFailure from Model into npr_disk
The following example couples NPR with another MLTK algorithm on a test set.

| inputlookup disk_failures.csv| head 5000 | fit NPR DiskFailure from Model | fit SGDClassifier DiskFailure from NPR_* random_state=42 n_iter=2 | score accuracy_score DiskFailure against predicted*
The following example uses NPR over multiple fields with additional uses of the fit command.

| inputlookup disk_failures.csv | head 5000 
| fit NPR DiskFailure from Model into npr_disk_1 
| fit NPR DiskFailure from SerialNumber into npr_disk_2
PCA
The Principal Component Analysis (PCA) algorithm uses the scikit-learn PCA algorithm to reduce the number of fields by extracting new, uncorrelated features out of the data.

Parameters

The k parameter specifies the number of features to be extracted from the data.
The variance parameter is short for percentage variance ratio explained. This parameter determines the percentage of variance ratio explained in the principal components of the PCA. It computes the number of principal components dynamically by preserving the specified variance ratio.
The variance parameter defaults to 1 if k is not provided.
The variance parameter can take a value between 0 and 1.
The component name parameter represents the name of the selected components from the value specified in n_components.  
The explained_variance parameter measures the proportion to which the principal component accounts for dispersion of a given dataset. A higher value denotes a higher variation.
The explained_variance_ratio parameter is the percentage of variance explained by each of the selected components.
The singular_values parameter represents the singular values corresponding to each of the selected components. Singular values are equal to the 2-norms of the n_components variables in the lower-dimensional space.
Syntax

fit PCA <fields> [into <model name>] [k=<int>] [variance=<float>]
You can save PCA models using the into keyword and apply new data later using the apply command.

...into example_hard_drives_PCA_2 | apply example_hard_drives_PCA_2"; // fill in your text here
string[] words = source.Split(' ');

for (int i = 0; i < words.Length; i++)
{
    if (i % wordsPerLine == 0)
    {
        Console.WriteLine();
        Console.WriteLine();
        Console.WriteLine("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        Console.WriteLine();
        Console.WriteLine();
    }

    Console.Write(words[i]);
    Console.Write(" ");
}
Console.ReadLine();
