import auth.dws.bigdata.common.{DataHandler, StopWords, TextProcessing}
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import auth.dws.bigdata.common.{RAKE, RAKEStrategy}

class RAKEtest extends AnyFlatSpec {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("Test")
    .master("local")
    .getOrCreate()

  "test" should "show results" in {
    val stopwords = StopWords.loadStopWords.toSet

    val rake = new RAKE(stopwords, Array(' '), Array('.', ',', '\n'))
    val text =
      """ Τελειώνω, κύριε Πρόεδρε. Εδώ θα είμαστε και θα τα ξαναπούμε με περισσότερη ευχέρεια.
        | Και όσα δε λέμε εδώ, υπάρχουν και κάποιοι ραδιοφωνικοί σταθμοί που πάμε και τα λέμε εκεί.
        | Δεν πρόκειται να μας φιμώσετε τη φωνή, με τη δήλωση ότι το ΠΑΣΟΚ θα είναι νικητής στις επόμενες
        | εκλογές. Και καμία καταδίωξη δεν είναι ικανή να το σπιλώσει και προπαντός τον Ανδρέα Παπανδρέου.  .""".stripMargin

    val rankedKeywords = rake.toScoredKeywords(text, RAKEStrategy.Ratio).toArray

    println(rankedKeywords
      .sortBy(_._2)
      .reverse
      .map(x => (x._1.mkString(" "), x._2))
      .mkString("Array(", ", ", ")"))
  }
}
