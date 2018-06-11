# coding:utf-8
# Created by chen on 11/06/2018
# email: q.chen@student.utwente.nl

from contextlib import contextmanager
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

SPARK_MASTER='local'
SPARK_APP_NAME='Word Count'
SPARK_EXECUTOR_MEMORY='500m'

@contextmanager
def spark_manager():
    conf = SparkConf().setMaster(SPARK_MASTER) \
                      .setAppName(SPARK_APP_NAME) \
                      .set("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
    spark_context = SparkContext(conf=conf)

    try:
        yield spark_context
    finally:
        spark_context.stop()

def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    print("values",values)
    return LabeledPoint(values[0], values[1:])

with spark_manager() as context:
    sc = SparkContext.getOrCreate()
    data = sc.textFile("C:\\Users\chen\PycharmProjects\REDI\pred\pandas.txt")
    print("data from txt file", data)
    parsedData = data.map(parsePoint)
    print("parsedDate", parsedData)
    model = SVMWithSGD.train(parsedData, iterations=100)

    # Evaluating the model on training data
    labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda lp: lp[0] != lp[1]).count() / float(parsedData.count())
    print("Training Error = " + str(trainErr))

    # Save and load model
    # model.save(sc, "pythonSVMWithSGDModel")
    # sameModel = SVMModel.load(sc, "pythonSVMWithSGDModel")

    # Compute raw scores on the test set
    predictionAndLabels = parsedData.map(lambda lp: (float(model.predict(lp.features)), lp.label))

    # Instantiate metrics object
    metrics = MulticlassMetrics(predictionAndLabels)

    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)

    # Statistics by class
    labels = data.map(lambda lp: lp.label).distinct().collect()
    for label in sorted(labels):
        print("Class %s precision = %s" % (label, metrics.precision(label)))
        print("Class %s recall = %s" % (label, metrics.recall(label)))
        print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

    # Weighted stats
    print("Weighted recall = %s" % metrics.weightedRecall)
    print("Weighted precision = %s" % metrics.weightedPrecision)
    print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
    print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
    print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)