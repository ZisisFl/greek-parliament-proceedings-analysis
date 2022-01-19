import auth.dws.bigdata.common.DataHandler.spark
import auth.dws.bigdata.common.{DataHandler, StopWords, TextProcessing}
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class Test extends AnyFlatSpec {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("Test")
    .master("local")
    .getOrCreate()

  "dataframe schema" should "be printed" in {
    val df = DataHandler.createDataFrame()

    df.printSchema()
  }

  "dataframe rows" should "be printed" in {
    val df = DataHandler.createDataFrame()

    val test = DataHandler.processSpeechText(df, false)
    test.show()
  }

  "dataframe " should "be processed" in {
    val df = DataHandler.createDataFrame()

    val test = DataHandler.processDataFrame(df)
    test.show()
  }

  "test" should "show results" in {
    val text =
      """ Από τα Κόμματα θα τις κάνουν, φυσικά, αλλά θα πρέπει να είναι ένας αριθμός συναδέλφων,
        | οι οποίοι θα λάβουν γνώση. Test English text here. Σχεδόν θα πρέπει να είναι όλοι, όπως αντιλαμβάνεσθε.
        | Γι` αυτό εφιστώ την προσοχή του Προεδρείου, ούτως ώστε η διανομή να γίνει το συντομότερο δυνατόν.""".stripMargin

    println(TextProcessing.textProcessingPipeline(text))
    println(TextProcessing.textProcessingPipeline(text).split("\\s").mkString("Array(", ", ", ")"))
  }

  "test" should "show results for Single text processing function" in {
    val text =
      """ Από τα Κόμματα θα τις κάνουν, φυσικά, αλλά θα πρέπει να είναι ένας αριθμός συναδέλφων,
        | οι οποίοι θα λάβουν γνώση. Test English text here. Σχεδόν θα πρέπει να είναι όλοι, όπως αντιλαμβάνεσθε.
        | Γι` αυτό εφιστώ την προσοχή του Προεδρείου, ούτως ώστε η διανομή να γίνει το  συντομότερο   δυνατόν  .  """.stripMargin

    val stopWords = StopWords.loadStopWords.toSet
    println(TextProcessing.textProcessingSingle(text, stopWords))
    println(TextProcessing.textProcessingSingle(text, stopWords).split("\\s").mkString("Array(", ", ", ")"))
  }

  "test" should "merge 2 arrays into one set" in {
    val removeDomainSpecificStopWords = false
    val stopWords = spark.sparkContext.broadcast(
      if (removeDomainSpecificStopWords){
        (StopWords.loadStopWords ++ StopWords.loadDomainSpecificStopWords).toSet
      }
      else {
        StopWords.loadStopWords.toSet
      }
    )
    print(stopWords.value.toArray.length)
  }
}
