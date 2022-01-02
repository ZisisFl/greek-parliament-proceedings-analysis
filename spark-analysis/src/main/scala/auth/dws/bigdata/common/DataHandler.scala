package auth.dws.bigdata.common

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{column, to_date, udf, concat_ws}
import org.apache.spark.sql.functions.monotonically_increasing_id
import auth.dws.bigdata.common.TextProcessing._

object DataHandler {
  // fetch existing global spark session
  val spark: SparkSession = SparkSession.builder().getOrCreate()

  def createDataFrame(): DataFrame = {
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
    dataFrame.filter(column("political_party") =!= "βουλη")
      // create a column with concatenation of political party and member name to define a different set of members
      .withColumn("member_name_with_party", concat_ws("_", column("member_name"), column("political_party")))
      // add an unique incremental id
      .withColumn("id", monotonically_increasing_id)
  }
}
