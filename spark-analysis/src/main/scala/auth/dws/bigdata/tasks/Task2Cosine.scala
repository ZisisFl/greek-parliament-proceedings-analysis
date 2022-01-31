package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.ml.feature.{HashingTF, IDF, MinHashLSH, Tokenizer}
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, column, size, udf, year}

object Task2Cosine {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task2")
      .master("local[*]")
      .getOrCreate()

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

    //    complete_df.show(20, truncate = false)

    // Cosine similarity
    val asDense = (v: SparseVector) => v.toDense //transform to dense matrix
    val asDenseUdf = udf(asDense)

    // Transformation to dense features. Most probably has worse performanse
    val dense_df = complete_df
      .select("id", "tfidf")
    //      .withColumn("dense_features", asDenseUdf(column("tfidf")))

    val dense_rows = dense_df.select("tfidf").rdd
      .map(_.getAs[org.apache.spark.ml.linalg.Vector](0))
      .map(org.apache.spark.mllib.linalg.Vectors.fromML)

    val mat = new RowMatrix(dense_rows)
    val simsEstimate = mat.columnSimilarities(0.8)

    println("Estimated pairwise similarities are: " + simsEstimate.entries.collect.mkString(", "))

    // transform Matrix back to dataframe
    val sqlContext = new org.apache.spark.sql.SQLContext(spark.sparkContext)
    val transformedRDD = simsEstimate.entries.map { case MatrixEntry(row: Long, col: Long, sim: Double) => (row, col, sim) }
    val similarities_df = sqlContext.createDataFrame(transformedRDD).toDF("id_A", "id_B", "sim")

    val sorted_df = similarities_df
      .sort(col("sim").desc)

    sorted_df.show()

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")

  }
}
