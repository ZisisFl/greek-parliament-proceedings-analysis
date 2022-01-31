package auth.dws.bigdata.tasks

import auth.dws.bigdata.common.DataHandler.createDataFrame
import auth.dws.bigdata.common.StopWords
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{column, explode, lit, monotonically_increasing_id, size, udf}
import org.apache.spark.ml.linalg.Vector
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
    val original_df = createDataFrame()
      .filter(column("sitting_date")===lit("2019-11-18"))
      //.filter(column("sitting_year")===2020)
      //.filter(column("political_party")==="νεα δημοκρατια")
      // .filter(column("parliamentary_period") === "period 5")
      //.filter(column("parliamentary_session") === "session 1")
      //.filter(column("parliamentary_sitting") === "sitting 5")
      //.sample(0.01)


    // split speeches in sentences and explode to different row per sentence
    val sentenceSplit = (input_text: String) => {
      input_text
        .replaceAll(" κ\\. ", " ")
        .split("(?<=.[.;]) +(?=[Α-ΩΆΈΌΊΏΉΎΫΪ́])")
    }
    val sentenceSplitUdf = udf(sentenceSplit)

    val sentence_df = original_df
      .withColumn("speech_sentence", explode(sentenceSplitUdf(column("speech"))))

    // load stop words
    val stopWords = spark.sparkContext.broadcast(
      if (removeDomainSpecificStopWords) {
        (StopWords.loadStopWords ++ StopWords.loadDomainSpecificStopWords).toSet
      }
      else {
        StopWords.loadStopWords.toSet
      }
    )

    // process speech at sentence level
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
      .withColumn("processed_speech_sentence", processSentenceUdf(column("speech_sentence")))
      // keep sentences with more than 2 and less than 20 tokens
      .filter(size(column("processed_speech_sentence")) > 2 and size(column("processed_speech_sentence")) < 20)
      .cache()

    // load pretrained Word2Vec model
    val model = Word2VecModel.load("src/main/resources/all100")

    // transform sentences
    val transformed_sentence_df = model
      .setInputCol("processed_speech_sentence")
      .setOutputCol("sentence_vectors")
      .transform(processed_sentence_df)
      .withColumn("id", monotonically_increasing_id)

    // compute cosine similarity for sentence pairs
    val cosineSim = (vector_a: Vector, vector_b: Vector) => {
      cosineSimilarity(vector_a, vector_b)
    }

    val vertices_df = transformed_sentence_df
      .select(column("id"),
        column("speech_sentence"),
        column("member_name"),
        column("political_party"),
        column("sitting_date"),
        column("parliamentary_period"),
        column("parliamentary_session"),
        column("parliamentary_sitting")
      )

    println(vertices_df.count())

    val cosineSimUdf = udf(cosineSim)

    val edges_df = transformed_sentence_df.select(column("id").as("src"),
                                                  column("sentence_vectors").as("vectors_a"))
      .crossJoin(transformed_sentence_df.select(column("id").as("dst"),
                                                column("sentence_vectors").as("vectors_b")))
      // remove edges with same src and dst node
      .filter(column("src")=!=column("dst"))
      // calculate cosine similarity
      .withColumn("cosine_similarity", cosineSimUdf(column("vectors_a"), column("vectors_b")))

    // udf to retain or remove edges based on Cosine similarity
    val randomGenerator = new scala.util.Random

    val biasedCoinFlip = (similarity: Double) => if (randomGenerator.nextDouble() <= similarity) 1 else 0

    val biasedCoinFlipUdf = udf(biasedCoinFlip)

    val edges_kept = edges_df
      .withColumn("keep_edge", biasedCoinFlipUdf(column("cosine_similarity")))
      // keep edges with weight value = 1
      .filter(column("keep_edge")===1)

    // GraphX RDD way
//    import org.apache.spark.rdd.RDD
//    import org.apache.spark.graphx.{Edge, Graph}
//    val nodes: RDD[(Long, String)] = vertex_df.rdd.map(p => (p(0).asInstanceOf[Long], p(1).asInstanceOf[String]))
//    val edges: RDD[Edge[Int]] = edges_df.select("src", "dst", "weight").rdd.map(p => Edge(p(0).asInstanceOf[Long], p(1).asInstanceOf[Long], randomGenerator.nextInt(100)))
//    val g: Graph[String, Int] = Graph(nodes, edges)
//    val result = g.pageRank(0.0001)
//    result.vertices.foreach(println)

    val g = GraphFrame(vertices_df, edges_kept)

    // graphX https://spark.apache.org/docs/latest/graphx-programming-guide.html#pagerank
    //val results = g.toGraphX.pageRank(0.0001)//pageRank.maxIter(20).run()

    //https://github.com/graphframes/graphframes/issues/148
    // graphframes https://graphframes.github.io/graphframes/docs/_site/api/python/graphframes.html#graphframes.GraphFrame.pageRank
    val results = g.pageRank
      .resetProbability(0.15)
      .maxIter(20)
      .run()

    results.vertices.sort(column("pagerank").desc).show(false)
    //results.vertices.take(200).foreach(println)
    //results.edges.take(200).foreach(println)
    //results.vertices.printSchema()

//    vertex_df
//      .write
//      .format("parquet")
//      .option("header", "true")
//      .save("src/main/scala/auth/dws/bigdata/results/vertexlist.parquet")
//
//    edge_df
//      .select("src","dst", "weight")
//      .write
//      .format("parquet")
//      .option("header", "true")
//      .save("src/main/scala/auth/dws/bigdata/results/edgelist.parquet")


    val duration = (System.nanoTime - start_time) / 1e9d
    println(s"Execution time was $duration seconds")
  }
}
