package auth.dws.bigdata.common

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{column, to_date, udf, concat_ws, size}
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
    // remove records of speech procedure
    val processed_df = dataFrame.filter(column("political_party") =!= "βουλη")
      // create a column with concatenation of political party and member name to define a different set of members
      .withColumn("member_name_with_party", concat_ws("_", column("member_name"), column("political_party")))

    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(processed_df)
      .withColumn("tokens_count", size(column("tokens")))
    tokenized_df//.filter(column("tokens_count") > 10)
  }

}
