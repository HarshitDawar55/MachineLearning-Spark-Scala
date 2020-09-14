import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

// Creating a Spark Session
val spark = SparkSession.builder().config("spark.master", "local").getOrCreate()

//
val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("file:////Users/harshitdawar/Github/Classic/MachineLearningWithSpark&Scala/src/main/scala/Datasets/USA_Housing.csv")

println(data.printSchema())