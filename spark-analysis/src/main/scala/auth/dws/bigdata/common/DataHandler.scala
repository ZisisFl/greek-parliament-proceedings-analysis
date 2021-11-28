package auth.dws.bigdata.common

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{column, to_date}

object DataHandler {

  def createDataFrame(spark: SparkSession): DataFrame = {
    val path = "src/main/resources/gpp.csv"

    spark.read
      .option("header", true)
      .csv(path)
      .withColumn("sitting_date", to_date(column("sitting_date"), "dd/MM/yyyy"))
  }

  def processDataFrame(dataFrame: DataFrame): DataFrame = {
    val processed = dataFrame.filter(column("member_name").isNotNull)
    processed
  }

}
