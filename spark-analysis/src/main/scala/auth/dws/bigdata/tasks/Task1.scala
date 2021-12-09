package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.clustering.{LDA, OnlineLDAOptimizer}

object Task1 {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task1")
      .master("local[4]")
      .getOrCreate()

    val original_df = DataHandler.createDataFrame(spark)

    val df_with_processed_speech = DataHandler.processSpeechText(original_df)
    df_with_processed_speech.show()

    val processed_df = DataHandler.processDataFrame(df_with_processed_speech)
      .withColumn("id", monotonically_increasing_id)
    processed_df.printSchema()
    processed_df.show()

    //https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3741049972324885/3783546674231782/4413065072037724/latest.html
    val vectorizer = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("features")
      .setVocabSize(10000)
      .setMinDF(5)
      .fit(processed_df)

    val countVectors = vectorizer.transform(processed_df).select("id", "features")

    import spark.implicits._
    val lda_countVector = countVectors.map { case Row(id: Long, countVector: Vector) => (id, countVector) }.rdd

    val numTopics = 20
    // Set LDA params
    val lda = new LDA()
      .setOptimizer(new OnlineLDAOptimizer().setMiniBatchFraction(0.8))
      .setK(numTopics)
      .setMaxIterations(3)
      .setDocConcentration(-1) // use default values
      .setTopicConcentration(-1) // use default values

    val ldaModel = lda.run(lda_countVector)

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 5)
    val vocabList = vectorizer.vocabulary
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.map(vocabList(_)).zip(termWeights)
    }
    println(s"$numTopics topics:")
    topics.zipWithIndex.foreach { case (topic, i) =>
      println(s"TOPIC $i")
      topic.foreach { case (term, weight) => println(s"$term\t$weight") }
      println(s"==========")
    }
  }
}
