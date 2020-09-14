//import org.apache.spark.ml.regression.LinearRegression

import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.feature.VectorAssembler
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
data = data.withColumn("Avg Area House Age", data("Avg Area House Age").cast(DoubleType))
println(data.printSchema())

for (i <- data.select("Avg Area House Age")){
  println(List(i)[0], i.getClass)
}
/*
// Selecting the required features
val features = data.drop("Price")

// Printing the columns in the features dataset
println(features.columns.mkString("Array(", ", ", ")"))

// Selecting the Price Label Independently
val labels = data.select("Price")
println(labels.show(5))

// Converting the features into Vector for Processing
val VA = new VectorAssembler().setInputCols(features.columns).setOutputCol("Features")



val AssembledFeatures = VA.transform(features)
*/