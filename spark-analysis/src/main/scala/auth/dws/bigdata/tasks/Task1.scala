package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler
import org.apache.spark.sql.SparkSession

object Task1 {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task1")
      .master("local")
      .getOrCreate()

    val df = DataHandler.createDataFrame(spark)

    df.printSchema()
    df.show()

    val test = DataHandler.processDataFrame(df)
    test.show()
  }
}
