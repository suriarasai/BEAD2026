from pyspark.sql import SparkSession
filename = "dummy.txt"
spark = SparkSession.builder.appName("TestApp").getOrCreate()
print(spark.read.text(filename).count())