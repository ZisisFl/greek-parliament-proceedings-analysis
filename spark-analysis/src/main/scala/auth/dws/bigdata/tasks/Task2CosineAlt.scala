package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.CosineSimilarity.cosineSimilarity
import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, collect_list, column, flatten, monotonically_increasing_id, size, udf, year}

object Task2CosineAlt {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task2")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    val original_df = createDataFrame().sample(0.1)

    val processed_speech_df = processSpeechText(original_df, removeDomainSpecificStopWords = false)

    val processed_df = processDataFrame(processed_speech_df)

    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(processed_df)

    val processed_df_w_year = tokenized_df.withColumn("sitting_year", year(column("sitting_date")))
      .filter(column("sitting_year").isNotNull)

    val df_per_political_party = processed_df_w_year
      .groupBy("political_party", "sitting_year")
      .agg(flatten(collect_list("tokens")) as "tokens_grouped")
      .withColumn("id", monotonically_increasing_id)

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

    val joined_df = complete_df
      .select(
        column("id").as("src"),
        column("tfidf").as("features_a"),
        col("political_party").as("political_party_a"),
        col("sitting_year").as("sitting_year_a"),
      )
      .crossJoin(complete_df
        .select(
          column("id").as("dst"),
          column("tfidf").as("features_b"),
          col("political_party").as("political_party_b"),
          col("sitting_year").as("sitting_year_b"),
        )
      )
      .filter(column("src") =!= column("dst"))
      .filter(column("political_party_a") =!= column("political_party_b"))
      .withColumn("sim", cosineSimUdf(column("features_a"), column("features_b")))

    val uniquePair = (x: String, y: String, k: String, l: String) => {
      Array(x, y, k, l).sorted
    }
    val uniquePairUdf = udf(uniquePair)

    val unique_sim_dif = joined_df
      .withColumn("unique_pair",
        uniquePairUdf(
          col("political_party_a"), col("sitting_year_a"),
          col("political_party_b"), col("sitting_year_b")
        ))

    val sorted_df = unique_sim_dif
      .dropDuplicates("unique_pair")
      .select("unique_pair", "sim")
      .sort(col("sim").desc)

//    sorted_df.show(numRows = 50, truncate = false)

    // write results into parquet files
    val path_to_results = "src/main/scala/auth/dws/bigdata/results/task2"

    sorted_df
      .write
      .format("parquet")
      .option("header", "true")
      .save(s"$path_to_results/cosine_party.parquet")

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
