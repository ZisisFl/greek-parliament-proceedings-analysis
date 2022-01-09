package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{collect_list, column, concat_ws, flatten, size, udf, year}
import org.apache.spark.ml.linalg.Vector
import auth.dws.bigdata.common.{RAKE, RAKEStrategy, StopWords}

object Task3 {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task1")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    // load original csv as DataFrame
    val original_df = createDataFrame().sample(0.1)

    // process speech column
    val processed_speech_df = processSpeechText(original_df)

    // process dataframe
    val processed_df = processDataFrame(processed_speech_df)

    // tokenize speech
    val tokenizer = new Tokenizer()
      .setInputCol("processed_speech")
      .setOutputCol("tokens")

    val tokenized_df = tokenizer.transform(processed_df)

    // extract year from sitting_date field, drop nulls
    val processed_df_w_year = tokenized_df.withColumn("sitting_year", year(column("sitting_date")))
      .filter(column("sitting_year").isNotNull)

    // calculate tfidf for every speech
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
    .where(column("tokens_count") > 20)
//
//    complete_df.show(false)
//    complete_df.printSchema()
//
//    // extract top-N keywords based on tfidf score from each speech token
//    val vocabList = vectorizer.vocabulary
//
//    // set N
//    val N = 5
//    val get_top_keywords = (tfidf: Vector) => {
//      tfidf.toArray
//        .zipWithIndex
//        .filterNot(_._1 ==0)
//        .sortBy(_._1)
//        .reverse
//        .take(N)
//        .map(_._2)
//        .map(vocabList(_))
//    }
//
//    // turn function into udf
//    val get_top_keywords_udf = udf(get_top_keywords)
//
//    // add mapped terms in dataframe
//    val df_with_top_keywords = complete_df.withColumn("topN_keywords", get_top_keywords_udf(column("tfidf")))
//
//    df_with_top_keywords.show()
//    df_with_top_keywords.printSchema()
//
//    // aggregate topN_keywords into a single Array year and political party
//    val grouped_df = df_with_top_keywords.groupBy("political_party", "sitting_year")
//      .agg(flatten(collect_list("topN_keywords")) as "topN_keywords_grouped")
//
//    val token_freq = (tokens: Seq[String]) => {
//      tokens.groupBy(identity).
//        mapValues(_.map(_ => 1).sum)
//        .toArray
//        .sortBy(_._2)
//        .reverse
//    }
//
//    val token_freq_udf = udf(token_freq)
//
//    val tokens_with_freq = grouped_df.withColumn("topN_keywords_freq", token_freq_udf(column("topN_keywords_grouped")))
//
//    tokens_with_freq.printSchema()
//    tokens_with_freq.where("political_party == 'νεα δημοκρατια'").orderBy(column("sitting_year")).show()

    val rake_algorithm = new RAKE(StopWords.loadStopWords.toSet, Array(' '), Array('.', ',', '\n', ';'))
    val rake = (input_text: String) => {
      rake_algorithm.toScoredKeywords(input_text, RAKEStrategy.Ratio)
        .toArray
        .sortBy(_._2)
        .reverse
        .map(x => (x._1.mkString(" "), x._2))
    }
    val rake_udf = udf(rake)

    complete_df.withColumn("rake_terms", rake_udf(column("speech"))).show(false)

    //https://towardsdatascience.com/keyword-extraction-methods-the-overview-35557350f8bb
    //https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.YakeKeywordExtraction.html?highlight=yake
    //https://www.analyticsvidhya.com/blog/2020/11/words-that-matter-a-simple-guide-to-keyword-extraction-in-python/
    //https://www.analyticsvidhya.com/blog/2021/10/rapid-keyword-extraction-rake-algorithm-in-natural-language-processing/
    // KALO PAPER https://sci-hub.se/10.1016/j.ins.2019.09.013
    // mas endiaferoyn language independent tropoi giati den xreiazetai POS, oi algorithmoi me GRAFOUS THELOUN POS

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
