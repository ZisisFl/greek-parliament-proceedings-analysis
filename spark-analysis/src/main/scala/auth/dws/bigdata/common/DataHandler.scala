package auth.dws.bigdata.common

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{column, to_date, udf, concat_ws}
import auth.dws.bigdata.common.TextProcessing._
import org.apache.spark.ml.feature.Tokenizer

object DataHandler {

  def createDataFrame(spark: SparkSession): DataFrame = {
    val path = "src/main/resources/gpp.csv"

    spark.read
      .option("header", true)
      .csv(path)
      .withColumn("sitting_date", to_date(column("sitting_date"), "dd/MM/yyyy"))
  }

  def processSpeechText(dataFrame: DataFrame): DataFrame = {
    // Create UDF to apply text processing functions to speech column
    val processSpeechText = (speech_text: String) => {
      textProcessingPipeline(speech_text)
    }
    val processSpeechTextUdf = udf(processSpeechText)

    dataFrame.withColumn("processed_speech",  processSpeechTextUdf(column("speech")))
  }

  def processDataFrame(dataFrame: DataFrame): DataFrame = {
    val countTokens = (speech_text: String) => {
      speech_text.split(" ").length
    }
    val countTokensUdf = udf(countTokens)
    // remove records of speech procedure
    val processed = dataFrame.filter(column("political_party") =!= "βουλη")
      .withColumn("tokens_count", countTokensUdf(column("processed_speech")))
      .filter(column("tokens_count") > 10)
      .withColumn("member_name_with_party", concat_ws("_", column("member_name"), column("political_party")))
      //.drop("tokens_count")

    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    tokenizer.transform(processed)
  }

}
