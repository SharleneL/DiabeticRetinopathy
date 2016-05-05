from pyspark.ml import Pipeline
from pyspark.sql import *
from pyspark.context import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import GBTClassifier
from pyspark.mllib.tree import DecisionTree

from dataframe import get_df


csv_fpath = '../output/1000_feature_output.csv'
sc = SparkContext()
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")  # forbit log

# GET DATA FRAME
pandas_df = get_df(csv_fpath)
train_df = sqlContext.createDataFrame(pandas_df)

# RANDOM FOREST
# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(train_df)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(train_df)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = train_df.randomSplit([0.7, 0.3])
# trainingData = train_df

# ========== / RANDOM FOREST - START / ========== #
# # Train a RandomForest model
# rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
#
# # Chain indexers and forest in a Pipeline
# pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])
#
# # Train model.  This also runs the indexers.
# model = pipeline.fit(trainingData)
#
# # Make predictions
# predictions = model.transform(testData)
#
# # Select (prediction, true label) and compute test error
# evaluator = MulticlassClassificationEvaluator(
#     labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
# accuracy = evaluator.evaluate(predictions)
#
# print '***prediction***'
# predictions.select("prediction", "indexedLabel", "features").show()
# print "Test Error = %g" % (1.0 - accuracy)
#
# rfModel = model.stages[2]
# print rfModel  # summary only
# # raw_input("prediction...")
# ========== / RANDOM FOREST - END / ========== #


# ========== / GBTREGRESSOR- START / ========== #
# # Train a GBT model.
# gbt = GBTRegressor(featuresCol="indexedFeatures", maxIter=10)
#
# # Chain indexer and GBT in a Pipeline
# pipeline = Pipeline(stages=[featureIndexer, gbt])
#
# # Train model.  This also runs the indexer.
# model = pipeline.fit(trainingData)
#
# # Make predictions.
# predictions = model.transform(testData)
#
# # Select example rows to display.
# predictions.select("prediction", "label", "features").show()
#
# # Select (prediction, true label) and compute test error
# evaluator = RegressionEvaluator(
#     labelCol="label", predictionCol="prediction", metricName="rmse")
# rmse = evaluator.evaluate(predictions)
# print "Root Mean Squared Error (RMSE) on test data = %g" % rmse
# ========== / GBTREGRESSOR- END / ========== #


# ========== / Gradient-Boosted Trees (GBTs)- START / ========== #
# Train a RandomForest model
nb = NaiveBayes(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, nb])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions
predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(predictions)

print '***prediction***'
predictions.select("prediction", "indexedLabel", "features").show()
print "Test Error = %g" % (1.0 - accuracy)
# raw_input("prediction...")
# ========== / Gradient-Boosted Trees (GBTs)- END / ========== #



# # Make predictions on test documents and print columns of interest.
# prediction = model.transform(train_df)
# selected = prediction
# for row in selected.collect():
#     print('*** ***')
#     print(row)