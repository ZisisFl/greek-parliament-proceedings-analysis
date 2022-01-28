package auth.dws.bigdata.common

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{column, to_date, udf, concat_ws}
import org.apache.spark.sql.functions.monotonically_increasing_id

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

  def processSpeechText(dataFrame: DataFrame, removeDomainSpecificStopWords: Boolean): DataFrame = {
    // Broadcast set of stop words
    val stopWords = spark.sparkContext.broadcast(
      if (removeDomainSpecificStopWords){
        (StopWords.loadStopWords ++ StopWords.loadDomainSpecificStopWords).toSet
      }
      else {
        StopWords.loadStopWords.toSet
      }
    )

    // Create UDF to apply text processing functions to speech column
    val textProcessingPipeline = (input_text: String) => {
      input_text
        .toLowerCase()
        .replaceAll("[^Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]+", " ")
        .split("\\s+")
        .filter(_.length > 2)
        .filter(!stopWords.value.contains(_))
        .mkString(" ")
    }

    val processSpeechTextUdf = udf(textProcessingPipeline)

    dataFrame
      .withColumn("processed_speech",  processSpeechTextUdf(column("speech")))
      // exclude speeches that after processing have no tokens left (are empty string "") e.g contained only stop words
      .filter(column("processed_speech") =!= "")
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
