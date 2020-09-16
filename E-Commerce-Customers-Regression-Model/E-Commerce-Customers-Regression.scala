////////////////////////////////////////////
//// Importing the required Libraries ///////////
/////////////////////////////////////////

import org.apache.spark.ml.regression
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

Logger.getLogger("org").setLevel(Level.ERROR)

