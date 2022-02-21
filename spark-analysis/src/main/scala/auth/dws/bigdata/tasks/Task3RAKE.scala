package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame}
import auth.dws.bigdata.common.{RAKE, RAKEStrategy, StopWords}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{collect_list, column, flatten, udf, regexp_replace}

object Task3RAKE {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task3 RAKE")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    val removeDomainSpecificStopWords = true

    // load original csv as DataFrame
    val original_df = createDataFrame()//.sample(0.01)

    // process dataframe
    val processed_df = processDataFrame(original_df)
      .withColumn("speech", regexp_replace(column("speech"), " κ\\.", " "))

    processed_df.show()

    // Create udf to identify rows that speech text consists of votes counts
    val nameWithVoteRegex = spark.sparkContext.broadcast("[Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]+\\s*[+-]".r)

    val votingIdentification = (input_text: String) => {
      nameWithVoteRegex
        .value
        .findAllIn(input_text)
        .length
    }

    val votingIdentificationUdf = udf(votingIdentification)

    // extract year from sitting_date field, drop nulls and filter documents
    val complete_df = processed_df
      .withColumn("name_with_vote_matches", votingIdentificationUdf(column("speech")))
      .filter(column("name_with_vote_matches") < 10)
      .drop(column("name_with_vote_matches"))

    // broadcast stop words
    val stopWords = spark.sparkContext.broadcast(
      if (removeDomainSpecificStopWords){
        (StopWords.loadStopWords ++ StopWords.loadDomainSpecificStopWords).toSet
      }
      else {
        StopWords.loadStopWords.toSet
      }
    )

    // initialize RAKE algorithm and create udf
    val tokenSeparators = Array(' ')
    val phraseSeparators = Array('.', ',', '\n', ';')

    val rake_algorithm = new RAKE(stopWords.value, tokenSeparators, phraseSeparators)
    val rake = (input_text: String) => {
      rake_algorithm.toScoredKeywords(input_text, RAKEStrategy.Ratio)
        .toArray
        .sortWith(_._2 > _._2)
        .map(x => (x._1.mkString(" "), x._2))
        .toSeq
    }
    val rake_udf = udf(rake)

    // run rake udf
    val speeches_with_rake_df = complete_df.withColumn("rake_terms", rake_udf(column("speech")))

    // aggregate topN_keywords into a single Array per year and political party
    val df_per_political_party = speeches_with_rake_df.groupBy("political_party", "sitting_year")
      .agg(flatten(collect_list("rake_terms")) as "rake_keywords_grouped")

    // aggregate topN_keywords into a single Array per year and member
    val df_per_member = speeches_with_rake_df.groupBy("member_name_with_party", "sitting_year")
      .agg(flatten(collect_list("rake_terms")) as "rake_keywords_grouped")

    // sum rake score of grouped phrases and get Top N
    val N = 5
    val get_top_rake_keywords = (terms: Seq[Row]) => {
      // casting an Array of tuples in udf requires using Seq[Rows]
      // https://stackoverflow.com/questions/41551410/passing-a-list-of-tuples-as-a-parameter-to-a-spark-udf-in-scala
      terms.map(x => (x.getAs[String](0), x.getAs[Double](1)))
        .groupBy(_._1)
        .mapValues(_.map(_._2).sum)
        .toArray
        .sortWith(_._2 > _._2)
        .take(N)
        .map(_._1)
    }

    val get_top_rake_keywords_udf = udf(get_top_rake_keywords)

    val df_per_political_party_final = df_per_political_party
      .withColumn("topN_keywords_freq", get_top_rake_keywords_udf(column("rake_keywords_grouped")))

    val df_per_member_final = df_per_member
      .withColumn("topN_keywords_freq", get_top_rake_keywords_udf(column("rake_keywords_grouped")))

    // print results
    //df_per_political_party_final.show(false)
    //df_per_member_final.show(false)

    // write results into parquet files
    val path_to_results = "src/main/scala/auth/dws/bigdata/results/task3"

    df_per_political_party_final
      .drop("rake_keywords_grouped")
      .write
      .format("parquet")
      .option("header", "true")
      .save("%s/keywords_political_party_rake.parquet".format(path_to_results))

    df_per_member_final
      .drop("rake_keywords_grouped")
      .write
      .format("parquet")
      .option("header", "true")
      .save("%s/keywords_member_rake.parquet".format(path_to_results))

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
