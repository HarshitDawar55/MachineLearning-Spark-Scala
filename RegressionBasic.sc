import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

var path: String = "<Path of the Dataset File>"
val spark = SparkSession.builder().appName("RegressionBasics").config("spark.master", "local").getOrCreate()

def main(): Unit = {
  // In the below line, provide the format accordingly
  val data = spark.read.option("header", "true").option("inferSchema", "true").format("libsvm").load(path)
  println(s"Schema Of Data %s \n type: %s\n", data.printSchema(), data.getClass)

  val lrObject = new LinearRegression().setMaxIter(150).setElasticNetParam(0.9).setRegParam(0.5)
  val Model = lrObject.fit(data)

  println(s"Coefficients ${Model.coefficients}  Intercept ${Model.intercept}")

  val trainingSummary = Model.summary
  println(s"numIterations: ${trainingSummary.totalIterations}")
  println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")


}
main()
