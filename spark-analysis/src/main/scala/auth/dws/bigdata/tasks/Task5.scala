package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.ml.clustering.{BisectingKMeans, GaussianMixture, KMeans}
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, IDF, Tokenizer, Word2Vec}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{column, size}

object Task5 {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task5")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    if (args.isEmpty) {
      println("No clustering algorithm provided")
      return
    }

//    val clustering_algorithm = args(0)
    val clustering_algorithm = "bkm"

    val original_df = createDataFrame().sample(0.01)

    val processed_speech_df = processSpeechText(original_df, removeDomainSpecificStopWords = true)
    val processed_df = processDataFrame(processed_speech_df)

    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(processed_df)

    val w2vectorizer = new Word2Vec()
      .setInputCol("tokens")
      .setOutputCol("tfidf") //fix this
      .setMinCount(5)
      .setVectorSize(100)
      .setWindowSize(5)
      .setSeed(42)

    val w2vec_model = w2vectorizer.fit(tokenized_df)
    val full_featurized_df = w2vec_model.transform(tokenized_df)

    val complete_df = full_featurized_df
          .withColumn("tokens_count", size(column("tokens")))
          .where(column("tokens_count") > 10)


    // Using CountVectorizer
    //    val vectorizer = new CountVectorizer()
    //      .setInputCol("tokens")
    //      .setOutputCol("tf")
    //      .setVocabSize(10000)
    //      .setMinDF(5)
    //      .fit(processed_df)
    //
    //    val featurized_df = vectorizer.transform(processed_df)

    // Using HashingTF
//    val hashingtf = new HashingTF()
//      .setInputCol("tokens")
//      .setOutputCol("tf")
//      .setNumFeatures(10000)
//
//    val featurized_df = hashingtf.transform(tokenized_df)
//
//    val idf = new IDF()
//      .setInputCol("tf")
//      .setOutputCol("tfidf")
//      .fit(featurized_df)
//
//    val full_featurized_df = idf.transform(featurized_df)
//
//    val complete_df = full_featurized_df
//      .withColumn("tokens_count", size(column("tokens")))
//      .where(column("tokens_count") > 20)

    // Specify number of requested clusters
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

    else {
      println("No clustering algorithm provided")
      return
    }

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")

//    // write results into parquet files
//    val path_to_results = "src/main/scala/auth/dws/bigdata/results/task5"
//
//    clusters_df
//      .write
//      .format("parquet")
//      .option("header", "true")
//      .save(s"$path_to_results/kmeans_clusters.parquet")

  }
}