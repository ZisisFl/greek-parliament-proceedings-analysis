package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{CountVectorizer, Tokenizer}
import org.apache.spark.sql.functions.{column, udf, size}
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
      .where(column("tokens_count") > 20)

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

    val countVectors = vectorizer.transform(tokenized_df).select("id", "features")

    val numTopics = 10
    val lda = new LDA()
      .setK(numTopics)
      .setMaxIter(20)

    val model = lda.fit(countVectors)

    val ll = model.logLikelihood(countVectors)
    val lp = model.logPerplexity(countVectors)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")

    // Describe topics.
    val topicIndices = model.describeTopics(10)

    // Map term indeces to actual terms from vocabulary
    val vocabList = vectorizer.vocabulary

    val index_to_term_mapping = (termIndices: Seq[Int]) => {
      termIndices.map(vocabList(_))
    }
    val term_mapping_udf = udf(index_to_term_mapping)

    // add mapped terms in dataframe
    val topicsWithTerms = topicIndices.withColumn("terms", term_mapping_udf(column("termIndices")))

    // Collect and print results
    println("The topics described by their top-weighted terms:")
    //topicsWithTerms.show(false)
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
      .save("%s/lda_topics_k_%s.parquet".format(path_to_results, numTopics))


    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
