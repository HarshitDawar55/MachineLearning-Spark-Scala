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

////////////////////////////////////////////
//// Data Preprocessing ///////////
/////////////////////////////////////////

// Printing the number of records before cleaning & after cleaning
println("Number of rows Initially: ", data.count())
data = data.na.drop()
println("Number of rows After Removing the Null Values: ", data.count())
println(data.show(5))

// Changing the name of the target column
data = data.withColumnRenamed("Yearly Amount Spent", "label")

////////////////////////////////////////////
//// Arranging the columns of the Dataset into Vector for Spark Processing ///////////
/////////////////////////////////////////

val VA = new VectorAssembler().setInputCols(
  Array("Avg Session Length", "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features")
val AssembledFeatures = VA.transform(data).select("features", "label")

println(AssembledFeatures.show(5, truncate = false) + "\n" + AssembledFeatures.printSchema())

////////////////////////////////////////////
//// Creating Linear Regression Model ///////////
/////////////////////////////////////////

val lr = new LinearRegression()

val trainedModel = lr.fit(AssembledFeatures).summary

// Displaying the Summary of the Model
println(trainedModel.predictions.show(5, truncate = false) +
  "\n" + trainedModel.residuals.show(5) + "\n" +
  s"Mean Squared Error: ${trainedModel.meanSquaredError}" + "\n" +
  s"Mean Absolute Error: ${trainedModel.meanAbsoluteError}" + "\n" +
  s"Root Mean Squared Error: ${trainedModel.rootMeanSquaredError}" + "\n" +
  s"Number of Iterations: ${trainedModel.totalIterations}" + "\n" +
  s"Objective History: ${trainedModel.objectiveHistory.toList}")