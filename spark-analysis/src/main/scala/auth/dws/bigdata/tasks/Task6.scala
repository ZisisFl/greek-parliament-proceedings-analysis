package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.{createDataFrame, processDataFrame, processSpeechText}
import auth.dws.bigdata.common.StopWords
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{udf, explode, column, size, monotonically_increasing_id}
import org.apache.spark.ml.linalg.Vector
//import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import auth.dws.bigdata.common.CosineSimilarity.cosineSimilarity
import org.graphframes.GraphFrame

object Task6 {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession
      .builder()
      .appName("Proceedings Analysis Task6")
      .master("local[*]")
      .getOrCreate()

    val start_time = System.nanoTime

    val removeDomainSpecificStopWords = false

    // load original csv as DataFrame
    val original_df = createDataFrame().sample(0.01)

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
        .filter(_.length > 3)
        .filter(!stopWords.value.contains(_))
    }

    val processSentenceUdf = udf(processSentence)

    val processed_sentence_df = sentence_df
      .withColumn("processed_speech_sentences", processSentenceUdf(column("speech_sentences")))
      // keep sentences with more than 2 and less than 20 tokens
      .filter(size(column("processed_speech_sentences")) > 2 and size(column("processed_speech_sentences")) < 20)
      .filter(column("parliamentary_period") === "period 5")
      //.filter(column("parliamentary_session") === "session 1")
      //.filter(column("parliamentary_sitting") === "sitting 5")
      .persist()

    // load model
    val model = Word2VecModel.load("src/main/resources/all100_old")

    val transformed_sentence_df = model
      .setInputCol("processed_speech_sentences")
      .setOutputCol("sentence_vectors")
      .transform(processed_sentence_df)
      .withColumn("id", monotonically_increasing_id)

    val cosineSim = (vector_a: Vector, vector_b: Vector) => {
      cosineSimilarity(vector_a, vector_b)
    }

    val vertex_df =  transformed_sentence_df
      .select(column("id"), column("speech_sentences").as("src"))

    println(vertex_df.count())

    val cosineSimUdf = udf(cosineSim)

    val edge_df = transformed_sentence_df.select(column("id").as("src") ,column("sentence_vectors").as("vectors_a"))
      .crossJoin(transformed_sentence_df.select(column("id").as("dst"),column("sentence_vectors").as("vectors_b")))
      .filter(column("src")=!=column("dst"))
      .withColumn("weight", cosineSimUdf(column("vectors_a"), column("vectors_b")))


    //val g = GraphFrame(vertex_df, edge_df)
    // graphX
    //val results = g.toGraphX.pageRank(0.0001)//pageRank.maxIter(20).run()

    // graphframes https://graphframes.github.io/graphframes/docs/_site/api/python/graphframes.html
    //val results = g.pageRank.resetProbability(0.15).maxIter(20).run()


    edge_df
      .select("src","dst", "weight")
      .write
      .format("parquet")
      .option("header", "true")
      .save("src/main/scala/auth/dws/bigdata/results/edgelist.parquet")
    //results.vertices.show(false)
    //results.vertices.take(200).foreach(println)
    //results.edges.take(200).foreach(println)
    //results.vertices.printSchema()





//    val yo = transformed_sentence_df
//      .select("sentence_vectors")
//      .rdd
//      .map(_.getAs[org.apache.spark.ml.linalg.Vector]("sentence_vectors"))
//      .map(org.apache.spark.mllib.linalg.Vectors.fromML)

//
//    val mat = new RowMatrix(yo)
//
//    // Compute similar columns perfectly, with brute force.
//    val exact = mat.columnSimilarities()
//
//    val exactEntries = exact.entries.map { case MatrixEntry(i, j, u) => ((i, j), u) }
//
//    exactEntries.foreach(println)




    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
