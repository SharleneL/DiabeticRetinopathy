from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import *
from pyspark.context import *
from pyspark.ml.classification import RandomForestClassifier
import pandas as pd
from pyspark.ml.feature import StringIndexer, VectorIndexer

from preprocess import get_df


csv_fpath = '../output/output.csv'
sc = SparkContext()
sqlContext = SQLContext(sc)

# taxiFile = sc.textFile(csv_fpath)

# pandas_df = pd.read_csv(csv_fpath)  # assuming the file contains a header
pandas_df = get_df(csv_fpath)
# pandas_df = pd.read_csv('file.csv', names = ['column 1','column 2']) # if no header
train_df = sqlContext.createDataFrame(pandas_df)


# # lr
# lr = LogisticRegression(maxIter=10, regParam=0.01)
# pipeline = Pipeline(stages=[lr])
# # Fit the pipeline to training documents.
# model = pipeline.fit(train_df)




# rf
# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(train_df)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(train_df)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = train_df.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

print '***prediction***'
print predictions.show()






# Make predictions on test documents and print columns of interest.
prediction = model.transform(train_df)
selected = prediction
for row in selected.collect():
    print('*** ***')
    print(row)