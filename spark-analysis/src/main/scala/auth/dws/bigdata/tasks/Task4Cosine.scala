package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.CosineSimilarity.cosineSimilarity
import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, collect_list, column, concat, flatten, lit, monotonically_increasing_id, quarter, size, udf, when}

object Task4Cosine {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task4 - similarity edition")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    val original_df = createDataFrame()//.sample(0.5)

    // The start of Greek financial crisis is set the last quarter of 2008
    // Add a quarter of year column to the data
    val quarters_df = original_df
      .withColumn("sitting_quarter", quarter(column("sitting_date")))
      .withColumn("sitting_quarter", concat(column("sitting_year"), lit(" Q"), column("sitting_quarter")))
      // add pro-crisis and after crisis 5 year periods
      .withColumn("financial_crisis_period",
        when((column("sitting_quarter") >= "2008 Q4") and (column("sitting_quarter") < "2013 Q4"), "Financial Crisis")
          .when((column("sitting_quarter") >= "2003 Q4") and (column("sitting_quarter") < "2008 Q4"), "Pre-financial crisis")
          .otherwise("non_interesting"))

    // keep only the quarters of interest -- 5 years prior to the crisis and 5 years after (main events)
    val crisis_periods = Seq("Financial Crisis", "Pre-financial crisis")
    val crisis_quarters_df = quarters_df
      .filter(column("financial_crisis_period").isin(crisis_periods:_*))
      .filter(column("political_party").notEqual("εξωκοινοβουλευτικος"))

    val processed_speech_df = processSpeechText(crisis_quarters_df, removeDomainSpecificStopWords = false)

    val processed_df = processDataFrame(processed_speech_df)

    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(processed_df)

    //    val processed_df_w_year = tokenized_df
    //      .withColumn("sitting_year", year(column("sitting_date")))
    //      .filter(column("sitting_year").isNotNull)

    val df_per_political_party = tokenized_df
      .groupBy("political_party", "sitting_quarter")
      .agg(flatten(collect_list("tokens")) as "tokens_grouped")
      //generate a unique id based on a concatenation of column values
      .withColumn("extended_id", concat(column("political_party"), lit("_"), column("sitting_quarter")))

    // Using HashingTF
    val hashingtf = new HashingTF()
      .setInputCol("tokens_grouped")
      .setOutputCol("tf")
      .setNumFeatures(10000)

    val featurized_df = hashingtf.transform(df_per_political_party)

    val idf = new IDF().setInputCol("tf")
      .setOutputCol("tfidf")
    val idfModel = idf.fit(featurized_df)

    val complete_df = idfModel.transform(featurized_df)
      .withColumn("tokens_count", size(column("tokens_grouped")))
      .where(column("tokens_count") > 10)

    val cosineSim = (vector_a: Vector, vector_b: Vector) => {
      cosineSimilarity(vector_a, vector_b)
    }
    val cosineSimUdf = udf(cosineSim)

    val pairs_df = complete_df
      .select(
        column("extended_id").as("extended_id_a"),
        column("tfidf").as("features_a"),
        col("political_party").as("political_party_a"),
        col("sitting_quarter").as("sitting_quarter_a"),
      )
      .crossJoin(complete_df
        .select(
          column("extended_id").as("extended_id_b"),
          column("tfidf").as("features_b"),
          col("political_party").as("political_party_b"),
          col("sitting_quarter").as("sitting_quarter_b"),
        )
      )
      .filter(column("extended_id_a") =!= column("extended_id_b"))
      //.filter(column("political_party_a") =!= column("political_party_b"))
      .withColumn("sim", cosineSimUdf(column("features_a"), column("features_b")))


    // write results into parquet files
    val path_to_results = "src/main/scala/auth/dws/bigdata/results/task4_sim"

    pairs_df
      .write
      .format("parquet")
      .option("header", "true")
      .save(s"$path_to_results/cosine_party_quarter.parquet")

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
