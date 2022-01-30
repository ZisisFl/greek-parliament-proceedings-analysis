package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.ml.feature.{HashingTF, IDF, MinHashLSH, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, column, size, udf, year}

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

    val original_df = createDataFrame().sample(0.01)

    val processed_speech_df = processSpeechText(original_df, removeDomainSpecificStopWords = false)

    val processed_df = processDataFrame(processed_speech_df)

    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(processed_df)

    val processed_df_w_year = tokenized_df.withColumn("sitting_year", year(column("sitting_date")))
      .filter(column("sitting_year").isNotNull)

    // Using HashingTF
    val hashingtf = new HashingTF()
      .setInputCol("tokens")
      .setOutputCol("tf")
      .setNumFeatures(10000)

    val featurized_df = hashingtf.transform(processed_df_w_year)

    val idf = new IDF().setInputCol("tf")
      .setOutputCol("tfidf")
    val idfModel = idf.fit(featurized_df)

    val complete_df = idfModel.transform(featurized_df)
      .withColumn("tokens_count", size(column("tokens")))
      .where(column("tokens_count") > 10)


    val mh = new MinHashLSH()
      .setNumHashTables(3)
      .setInputCol("tfidf")
      .setOutputCol("hashes")

    val mh_model = mh.fit(complete_df)

    val approxSimilarityJoin = mh_model.approxSimilarityJoin(complete_df, complete_df, 0.9, "JaccardDistance")

    val similarity_df = approxSimilarityJoin.toDF()
      .filter(col("JaccardDistance") > 0.5)
      .select(
        col("JaccardDistance"),
        col("datasetA.id").as("id_a"),
        col("datasetA.member_name_with_party").as("member_name_a"),
        col("datasetA.sitting_year").as("sitting_year_a"),
        col("datasetB.id").as("id_b"),
        col("datasetB.member_name_with_party").as("member_name_b"),
        col("datasetB.sitting_year").as("sitting_year_b")
      )

    val join_func = (x: String, y: String) => {
      Set(x, y)
    }

    val join_funcUDF = udf(join_func)

    val temp = similarity_df
      .withColumn(
        "speakers_pair",
        join_funcUDF(col("member_name_a"), col("member_name_b"))
      )


    val sorted_similarities_df = temp
      .select(
        col("speakers_pair"),
        col("JaccardDistance")
      )
      .distinct()
      .sort(col("JaccardDistance").desc)

    sorted_similarities_df.distinct().show(false)
    sorted_similarities_df.printSchema()

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")

  }
}
