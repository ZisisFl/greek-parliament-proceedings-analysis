package auth.dws.bigdata.common

import auth.dws.bigdata.common.DataHandler.createDataFrame
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object TrainWord2Vec {
  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Train Word2Vec embeddings")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    val removeDomainSpecificStopWords = false

    // load original csv as DataFrame
    val original_df = createDataFrame()

    // split speeches in sentences and explode to different row per sentence
    val sentenceSplit = (input_text: String) => {
      input_text
        .replaceAll(" κ\\. ", " ")
        .split("(?<=.[.;]) +(?=[Α-ΩΆΈΌΊΏΉΎΫΪ́])")
    }
    val sentenceSplitUdf = udf(sentenceSplit)

    val sentence_df = original_df
      .withColumn("speech_sentences", explode(sentenceSplitUdf(column("speech"))))

    // load stop words
    val stopWords = spark.sparkContext.broadcast(
      if (removeDomainSpecificStopWords) {
        (StopWords.loadStopWords ++ StopWords.loadDomainSpecificStopWords).toSet
      }
      else {
        StopWords.loadStopWords.toSet
      }
    )

    // process speech as sentence level
    val processSentence = (input_text: String) => {
      input_text
        .toLowerCase()
        .replaceAll("[^Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]+", " ")
        .split("\\s+")
        .filter(_.length > 2)
        .filter(!stopWords.value.contains(_))
    }

    val processSentenceUdf = udf(processSentence)

    val processed_sentence_df = sentence_df
      .withColumn("processed_speech_sentences", processSentenceUdf(column("speech_sentences")))
      // keep sentences with more than 2 and less than 20 tokens
      .filter(size(column("processed_speech_sentences")) > 2 and size(column("processed_speech_sentences")) < 20)
      .persist()

    // train Word2Vec model
    val word2Vec = new Word2Vec()
      .setInputCol("processed_speech_sentences")
      .setOutputCol("vectors")
      .setVectorSize(100)
      .setMinCount(30)
      .setWindowSize(5)

    val model = word2Vec.fit(processed_sentence_df)

    // save model
    model.save("src/main/resources/all100")

    // corpus statistics
    processed_sentence_df.select(
      mean(size(column("processed_speech_sentences") as "avg sentence length")),
      count(column("processed_speech_sentences") as "number of sentences"),
      sum(size(column("processed_speech_sentences") as "number of tokens")),
    ).show()

    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
