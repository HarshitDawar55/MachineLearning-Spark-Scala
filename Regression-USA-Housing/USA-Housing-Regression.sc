//import org.apache.spark.ml.regression.LinearRegression

import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types.DoubleType

Logger.getLogger("org").setLevel(Level.ERROR)

// Creating a Spark Session
val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

// Loaded the Data
var data = spark.read.option("header", "true").option("inferSchema",
  "true").format("csv").load(
  "file:////Users/harshitdawar/Github/Classic/MachineLearningWithSpark&Scala/src/main/scala/Datasets/USA_Housing.csv")

// Printing the Schema of the Project
println(data.printSchema())

// Printing the Top 5 Rows!
println(data.show(5))


//////////////////////////////////////////////////////////////////////////////
// Removing all the Null Values from the Dataset!
data = data.na.drop()

// Printing the Schema of the Project
println(data.printSchema())

// Printing the Top 5 Rows!
println(data.show(5))


//////////////////////////////////////////////////////////////////////////////
// Removing the String Datatype of the columns which are not required!
data = data.withColumn("Avg Area House Age",
  data("Avg Area House Age").cast(DoubleType)).withColumn("Avg Area Income",
  data("Avg Area Income").cast(DoubleType))

println(data.printSchema())
println(data.show(5))
println(s"Rows Present now are: ${data.count()}")


//////////////////////////////////////////////////////////////////////////////
// Removing the Address Column, as it will not have significance in the Model Output
data = data.drop("Address")

println(data.printSchema())
println(data.show(5))
println(s"Rows Present now are: ${data.count()}")

//////////////////////////////////////////////////////////////////////////////
// Selecting the required features
val features = data.drop("Price")

// Printing the columns in the features dataset
println(features.columns.mkString("Array(", ", ", ")"))

// Changing the name of the Price column to 'label' for Ml Model Working
data = data.withColumnRenamed("Price", "label")
println(data.show(5))

//////////////////////////////////////////////////////////////////////////////
// Converting the features into Vector for Processing
val VA = new VectorAssembler().setInputCols(features.columns).setOutputCol("features")
val AssembledFeatures = VA.transform(data).select("label", "features")

println(AssembledFeatures.show(5))


//////////////////////////////////////////////////////////////////////////////
// Creating the Linear Regression Model
val lr = new LinearRegression()

// Fitting the Model with the data
val lrModel = lr.fit(AssembledFeatures)

val trainingSummary = lrModel.summary

// Displaying the Summary of the Model
println(s"Predictions: ${trainingSummary.predictions.show(5, truncate = false)}" +
  "\n" + s"Difference in outputs  ${trainingSummary.residuals.show(5)}" + "\n" +
  s"Mean Squared Error: ${trainingSummary.meanSquaredError}" + "\n" +
  s"Mean Absolute Error: ${trainingSummary.meanAbsoluteError}" + "\n" +
  s"Root Mean Squared Error: ${trainingSummary.rootMeanSquaredError}" + "\n" +
  s"Number of Iterations: ${trainingSummary.totalIterations}")