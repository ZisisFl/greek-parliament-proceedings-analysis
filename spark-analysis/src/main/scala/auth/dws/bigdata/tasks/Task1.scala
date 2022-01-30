package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer}
import org.apache.spark.sql.functions.{column, size, udf}
import org.apache.spark.ml.clustering.LDA

object Task1 {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task1")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    // load original csv as DataFrame
    val original_df = createDataFrame()//.sample(0.01)

    // process speech column
    val processed_speech_df = processSpeechText(original_df, removeDomainSpecificStopWords = true)

    // process dataframe
    val processed_df = processDataFrame(processed_speech_df)

    // tokenize speech
    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(processed_df)
      .withColumn("tokens_count", size(column("tokens")))
      .where(column("tokens_count") > 10)

//    tokenized_df.printSchema()
//    tokenized_df.show()
//    println(tokenized_df.count())

    // create feature vectors with bag of words technique out of speech text
    val vectorizer = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("features")
      .setVocabSize(10000)
      .setMinDF(5)
      .fit(tokenized_df)

    val transformed_df = vectorizer
      .transform(tokenized_df)
      .select("id", "features", "sitting_year")
      .cache()

    // Map term indices to actual terms from vocabulary
    val vocabList = spark.sparkContext.broadcast(vectorizer.vocabulary)

    val index_to_term_mapping = (termIndices: Seq[Int]) => {
      termIndices.map(vocabList.value(_))
    }
    val term_mapping_udf = udf(index_to_term_mapping)

    // create an array of different sitting years
    import spark.implicits._
    val sitting_years = transformed_df
      .select("sitting_year")
      .distinct()
      .as[Int]
      .collect()
      .sortWith(_ < _)

    // run LDA to extract topics per year
    sitting_years.foreach(sitting_year => runLDA(sitting_year, 5))

    def runLDA(sitting_year: Int, numTopics: Int): Unit = {
      val countVectors = transformed_df
        .filter(column("sitting_year")===sitting_year)
        .select("id", "features")

      val lda = new LDA()
        .setK(numTopics)
        .setMaxIter(20)

      val model = lda.fit(countVectors)

      // Describe topics.
      val topicIndices = model.describeTopics(10)

      // add mapped terms in dataframe
      val topicsWithTerms = topicIndices.withColumn("terms", term_mapping_udf(column("termIndices")))

      // Collect and print results
      println("The topics described by their top-weighted terms for sitting year %s:".format(sitting_year))
      topicsWithTerms.collect.foreach(x => {
        val topic = x.getAs[Int]("topic")
        val topicTerms = x.getAs[Seq[String]]("terms")
        val termWeights = x.getAs[Seq[Double]]("termWeights")

        println(s"Topic $topic")
        println("Terms:")
        topicTerms.zip(termWeights).foreach({case (term, termWeight) => println(s"$term $termWeight")})
        println("\n")
      })

      val path_to_results = "src/main/scala/auth/dws/bigdata/results/task1"

      topicsWithTerms
        .drop("termIndices")
        .write
        .format("parquet")
        .option("header", "true")
        .save("%s/lda_topics_k_%s_%s.parquet".format(path_to_results, numTopics, sitting_year))
    }

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
