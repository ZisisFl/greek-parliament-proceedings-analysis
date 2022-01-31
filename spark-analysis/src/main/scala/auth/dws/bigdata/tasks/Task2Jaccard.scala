package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.ml.feature.{HashingTF, IDF, MinHashLSH, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, collect_list, column, flatten, size, udf, year}

object Task2Jaccard {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task2")
      .master("local[*]")
      .getOrCreate()

    // Set logging level to navigate output
    spark.sparkContext.setLogLevel("ERROR")

    val start_time = System.nanoTime

    val initial_df = createDataFrame()
    val temp_df = initial_df.filter(col("sitting_year") > 2000)
    val original_df = temp_df.sample(0.20, seed = 42)

    val processed_speech_df = processSpeechText(original_df, removeDomainSpecificStopWords = false)

    val processed_df = processDataFrame(processed_speech_df)

    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(processed_df)

    val processed_df_w_year = tokenized_df.withColumn("sitting_year", year(column("sitting_date")))
      .filter(column("sitting_year").isNotNull)

    val df_per_political_party = processed_df_w_year.groupBy("political_party", "sitting_year")
      .agg(flatten(collect_list("tokens")) as "tokens_grouped")

//    val df_per_member = processed_df_w_year.groupBy("member_name_with_party", "sitting_year")
//      .agg(flatten(collect_list("tokens")) as "tokens_grouped")

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


    val mh = new MinHashLSH()
      .setNumHashTables(3)
      .setInputCol("tfidf")
      .setOutputCol("hashes")

    val mh_model = mh.fit(complete_df)

    val approxSimilarityJoin = mh_model.approxSimilarityJoin(complete_df, complete_df, 0.9, "JaccardDistance")

//    approxSimilarityJoin.show()
//    approxSimilarityJoin.printSchema()

    val similarity_df = approxSimilarityJoin.toDF()
      .filter(col("JaccardDistance") > 0.1)
      .filter(col("JaccardDistance") < 0.9)
      .select(
        col("JaccardDistance"),
        col("datasetA.political_party").as("political_party_a"),
        col("datasetA.sitting_year").as("sitting_year_a"),
        col("datasetB.political_party").as("political_party_b"),
        col("datasetB.sitting_year").as("sitting_year_b")
      )

    //    val similarity_df = approxSimilarityJoin.toDF()
    //      .filter(col("JaccardDistance") > 0.1)
    //      .filter(col("JaccardDistance") < 0.9)
    //      .select(
    //        col("JaccardDistance"),
    //        col("datasetA.member_name_with_party").as("member_name_with_party_a"),
    //        col("datasetA.sitting_year").as("sitting_year_a"),
    //        col("datasetB.member_name_with_party").as("member_name_with_party_b"),
    //        col("datasetB.sitting_year").as("sitting_year_b")
    //      )

    //    val similarity_df = approxSimilarityJoin.toDF()
    //      .filter(col("JaccardDistance") > 0.5)
    //      .select(
    //        col("JaccardDistance"),
    //        col("datasetA.id").as("id_a"),
    //        col("datasetA.member_name_with_party").as("member_name_a"),
    //        col("datasetA.sitting_year").as("sitting_year_a"),
    //        col("datasetB.id").as("id_b"),
    //        col("datasetB.member_name_with_party").as("member_name_b"),
    //        col("datasetB.sitting_year").as("sitting_year_b")
    //      )

    // Unique Pairs

    val uniquePair = (x: String, y: String, k: String, l: String) => {
      Array(x, y, k, l).sorted
    }
    val uniquePairUdf = udf(uniquePair)

    val unique_sim_df = similarity_df
      .withColumn("speakers_pair",
        uniquePairUdf(
          col("political_party_a"), col("sitting_year_a"),
          col("political_party_b"), col("sitting_year_b")
        ))

    //    val unique_sim_df = similarity_df
    //      .withColumn("speakers_pair", uniquePairUdf(col("id_a"), col("id_b")))

    val sorted_similarities_df = unique_sim_df
      .dropDuplicates("speakers_pair")
      .sort(col("JaccardDistance").asc)

    sorted_similarities_df.show(numRows=50, truncate=false)

    // write results into parquet files
    val path_to_results = "src/main/scala/auth/dws/bigdata/results/task2"

    sorted_similarities_df
      .write
      .format("parquet")
      .option("header", "true")
      .save(s"$path_to_results/jaccard_party.parquet")

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")

  }
}
