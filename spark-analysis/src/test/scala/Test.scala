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

    val test = DataHandler.processSpeechText(df)
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
  };
}
