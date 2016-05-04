from pyspark.ml import Pipeline
from pyspark.sql import *
from pyspark.context import *
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer

from preprocess import get_df


csv_fpath = '../output/output.csv'
sc = SparkContext()
sqlContext = SQLContext(sc)

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
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(train_df)

# Split the data into training and test sets (30% held out for testing)
# (trainingData, testData) = train_df.randomSplit([0.7, 0.3])
trainingData = train_df

# Train a RandomForest model
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions
predictions = model.transform(trainingData)

print '***prediction***'
predictions.select("prediction", "indexedLabel", "features").show()


# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
accuracy = evaluator.evaluate(predictions)
print "Test Error = %g" % (1.0 - accuracy)

rfModel = model.stages[2]
print rfModel  # summary only
raw_input("prediction...")



# # Make predictions on test documents and print columns of interest.
# prediction = model.transform(train_df)
# selected = prediction
# for row in selected.collect():
#     print('*** ***')
#     print(row)