package auth.dws.bigdata.common

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, column, to_date, udf}

object DataHandler {

  def createDataFrame(spark: SparkSession): DataFrame = {
    val path = "src/main/resources/gpp.csv"

    spark.read
      .option("header", true)
      .csv(path)
      .withColumn("sitting_date", to_date(column("sitting_date"), "dd/MM/yyyy"))
  }

  def processSpeech(dataFrame: DataFrame): DataFrame = {
    // Create UDF to apply text processing functions to speech column
    val processText: String => String = x => TextProcessing.removeStopWords(TextProcessing.removeNonCharacters(x.toLowerCase())).trim
    val processTextUdf = udf(processText)

    dataFrame.withColumn("processed_speech",  processTextUdf(column("speech")))
  }

  def processDataFrame(dataFrame: DataFrame): DataFrame = {
    // remove records of speech procedure
    val processed = dataFrame.filter(column("political_party") =!= "βουλη")

    processed
  }

}
