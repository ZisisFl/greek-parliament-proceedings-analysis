package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.ml.clustering.{BisectingKMeans, GaussianMixture, KMeans}
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{collect_list, column, flatten, monotonically_increasing_id, size, udf, year}

object Task5Alt {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task5")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    val original_df = createDataFrame().sample(0.1)

    val processed_speech_df = processSpeechText(original_df, removeDomainSpecificStopWords = true)

    val processed_df = processDataFrame(processed_speech_df)

    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(processed_df)

    val processed_df_w_year = tokenized_df.withColumn("sitting_year", year(column("sitting_date")))
      .filter(column("sitting_year").isNotNull)


    // Using CountVectorizer
    val vectorizer = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("tf")
      .setVocabSize(10000)
      .setMinDF(5)
      .fit(processed_df_w_year)

    val featurized_df = vectorizer.transform(processed_df_w_year)

    val idf = new IDF().setInputCol("tf")
      .setOutputCol("tfidf")
    val idfModel = idf.fit(featurized_df)

    val complete_df = idfModel.transform(featurized_df)
      .withColumn("tokens_count", size(column("tokens")))
      .where(column("tokens_count") > 10)

    // Set clustering algorithm
    val clustering_algorithm = "-"
    val numberOfClusters = 4

    //KMeans
    if (clustering_algorithm == "km") {
      // KMeans
      val kmeans = new KMeans()
        .setFeaturesCol("tfidf")
        .setPredictionCol("cluster_prediction")
        .setDistanceMeasure("cosine")
        .setK(numberOfClusters)
        .setSeed(42)

      val km_model = kmeans.fit(complete_df)
      val clusters_df = km_model.transform(complete_df)

      clusters_df.groupBy("cluster_prediction").count().show()

    }
    else if (clustering_algorithm == "bkm") {
      // Bisecting KMeans
      val bkmeans = new BisectingKMeans()
        .setFeaturesCol("tfidf")
        .setPredictionCol("cluster_prediction")
        .setDistanceMeasure("cosine")
        .setK(numberOfClusters)
        .setSeed(42)

      val bkm_model = bkmeans.fit(complete_df)
      val clusters_df = bkm_model.transform(complete_df)

      clusters_df.groupBy("cluster_prediction").count().show()
    }
    else if (clustering_algorithm == "gmm") {
      // Gaussian Mixture Model
      // Does NOT run due to memory
      val gmm = new GaussianMixture()
        .setFeaturesCol("tfidf")
        .setPredictionCol("cluster_prediction")
        .setProbabilityCol("cluster_probability")
        .setTol(0.1)
        .setK(numberOfClusters)
        .setSeed(42)

      val gmm_model = gmm.fit(complete_df)
      val clusters_df = gmm_model.transform(complete_df)

      clusters_df.groupBy("cluster_prediction").count().show()
    }

    // Bisecting KMeans
    val bkmeans = new BisectingKMeans()
      .setFeaturesCol("tfidf")
      .setPredictionCol("cluster_prediction")
      .setDistanceMeasure("cosine")
      .setK(numberOfClusters)
      .setSeed(42)

    val bkm_model = bkmeans.fit(complete_df)
    val clusters_df = bkm_model.transform(complete_df)

    // extract top-N keywords based on tfidf score from each speech token
    val vocabList = vectorizer.vocabulary

    // set N
    val N = 5
    val get_top_keywords = (tfidf: Vector) => {
      tfidf.toArray
        .zipWithIndex
        .filterNot(_._1 == 0)
        .sortWith(_._1 > _._1)
        .take(N)
        .map(_._2)
        .map(vocabList(_))
    }
    val get_top_keywords_udf = udf(get_top_keywords)

    // add mapped terms in dataframe
    val kw_clusters_df = clusters_df.withColumn("topN_keywords", get_top_keywords_udf(column("tfidf")))

    // aggregate topN_keywords into a single Array per year and political party
    val agg_kw_clusters_df = kw_clusters_df.groupBy("cluster_prediction")
      .agg(flatten(collect_list("topN_keywords")) as "topN_keywords_grouped")

    // sort term by frequency and extract top M
    val M = 10
    val term_freq = (tokens: Seq[String]) => {
      tokens.groupBy(identity)
        .mapValues(_.map(_ => 1).sum)
        .toArray
        .sortWith(_._2 > _._2)
        .take(M)
        .map(_._1)
    }

    val term_freq_udf = udf(term_freq)

    // apply udf to get top keywords in both dataframes
    val agg_kw_clusters_df_final = agg_kw_clusters_df
      .withColumn("topN_keywords_freq", term_freq_udf(column("topN_keywords_grouped")))

    agg_kw_clusters_df_final.select("topN_keywords_freq").collect().foreach(println)

    // write results into parquet files
    val path_to_results = "src/main/scala/auth/dws/bigdata/results/task5"

    agg_kw_clusters_df
      .write
      .format("parquet")
      .option("header", "true")
      .save(s"$path_to_results/bkm_clusters_speech.parquet")

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")

  }
}
