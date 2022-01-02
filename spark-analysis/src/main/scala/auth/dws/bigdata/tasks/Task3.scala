package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{collect_list, column, concat_ws, udf, year}

object Task3 {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task1")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    // load original csv as DataFrame
    val original_df = createDataFrame()//.sample(0.1)

    // process speech column
    val processed_speech_df = processSpeechText(original_df)

    // process dataframe
    val processed_df = processDataFrame(processed_speech_df)

    // extract year from sitting_date field, drop nulls
    val processed_df_w_year = processed_df.withColumn("sitting_year", year(column("sitting_date")))
      .filter(column("sitting_year").isNotNull)

    processed_df_w_year.show()

    //https://towardsdatascience.com/keyword-extraction-methods-the-overview-35557350f8bb
    //https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.YakeKeywordExtraction.html?highlight=yake
    //https://www.analyticsvidhya.com/blog/2020/11/words-that-matter-a-simple-guide-to-keyword-extraction-in-python/


    // create view of proceedings to run queries against
//    processed_df_w_year.createOrReplaceTempView("proceedings")
//
//    val proceedings_per_party = spark.sql(
//      """SELECT political_party, sitting_year, concat(processed_speech, ' ') AS concat_speech
//        |FROM proceedings
//        |GROUP BY political_party, sitting_year""".stripMargin)
//
//    proceedings_per_party.show()

//    val proceedings_per_member = spark.sql(
//      """SELECT political_party, min(sitting_year) as first_year, max(sitting_year) as last_year
//        |FROM proceedings
//        |GROUP BY political_party""".stripMargin)
//
//    proceedings_per_member.show()

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
