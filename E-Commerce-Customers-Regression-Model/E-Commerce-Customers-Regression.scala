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

// Printing few rows from the Dataset
println(data.show(5))

// Printing the number of records before cleaning & after cleaning
println("Number of rows Initially: ",data.count())
data = data.na.drop()
println("Number of rows After Removing the Null Values: ",data.count())