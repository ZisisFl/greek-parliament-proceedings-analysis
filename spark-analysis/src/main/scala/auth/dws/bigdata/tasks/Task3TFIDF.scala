package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Task3TFIDF {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task3 TFIDF")
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

    // turn function into udf
    val get_top_keywords_udf = udf(get_top_keywords)

    // add mapped terms in dataframe
    val df_with_top_keywords = complete_df.withColumn("topN_keywords", get_top_keywords_udf(column("tfidf")))

    // aggregate topN_keywords into a single Array per year and political party
    val df_per_political_party = df_with_top_keywords.groupBy("political_party", "sitting_year")
      .agg(flatten(collect_list("topN_keywords")) as "topN_keywords_grouped")

    // aggregate topN_keywords into a single Array per year and member
    val df_per_member = df_with_top_keywords.groupBy("member_name_with_party", "sitting_year")
      .agg(flatten(collect_list("topN_keywords")) as "topN_keywords_grouped")

    // sort term by frequency and extract top M
    val M = 5
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
    val df_per_political_party_final = df_per_political_party
      .withColumn("topN_keywords_freq", term_freq_udf(column("topN_keywords_grouped")))

    val df_per_member_final = df_per_member
      .withColumn("topN_keywords_freq", term_freq_udf(column("topN_keywords_grouped")))

    // print results
    //df_per_political_party_final.show(false)
    //df_per_member_final.show(false)

    // write results into parquet files
    val path_to_results = "src/main/scala/auth/dws/bigdata/results/task3"

    df_per_political_party_final
      .drop("topN_keywords_grouped")
      .write
      .format("parquet")
      .option("header", "true")
      .save(s"$path_to_results/keywords_political_party_tfidf.parquet")

    df_per_member_final
      .drop("topN_keywords_grouped")
      .write
      .format("parquet")
      .option("header", "true")
      .save(s"$path_to_results/keywords_member_tfidf.parquet")

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
