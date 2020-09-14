import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

// Creating a Spark Session
val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

// Loaded the Data
val data = spark.read.option("header", "true").option("inferSchema",
  "true").format("csv").load(
  "file:////Users/harshitdawar/Github/Classic/MachineLearningWithSpark&Scala/src/main/scala/Datasets/USA_Housing.csv")

// Printing the Schema of the Project
println(data.printSchema())

// Printing the Top 5 Rows!
println(data.show(5))

// Selecting the required features
val features = data.drop("Price")

// Printing the columns in the features dataset
println(features.columns.mkString("Array(", ", ", ")"))

