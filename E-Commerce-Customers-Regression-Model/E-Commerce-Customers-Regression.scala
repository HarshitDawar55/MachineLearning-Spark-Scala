////////////////////////////////////////////
//// Importing the required Libraries ///////////
/////////////////////////////////////////

import org.apache.spark.ml.regression
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

Logger.getLogger("org").setLevel(Level.ERROR)

// Creating a Spark Session
val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

// Loaded the Data
var data = spark.read.option("header", "true").option("inferSchema",
  "true").format("csv").load(
  "file:////Users/harshitdawar/Github/Classic/MachineLearningWithSpark&Scala/src/main/scala/Datasets/Ecommerce-Customers.csv")

// Printing the Schema of the Project
println(data.printSchema())