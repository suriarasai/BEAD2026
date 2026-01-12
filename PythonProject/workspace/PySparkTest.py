from pyspark.sql import SparkSession
filename = "/workspace/dummy.txt"
spark = SparkSession.builder.appName("TestApp").getOrCreate()
print(spark.read.text(filename).count())