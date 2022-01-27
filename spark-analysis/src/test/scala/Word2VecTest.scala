import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec

class Word2VecTest extends AnyFlatSpec {
  val spark: SparkSession = SparkSession
    .builder()
    .appName("Test")
    .master("local")
    .getOrCreate()

  "test" should "show synonyms" in {
    val model = Word2VecModel.load("src/main/resources/all100")
    model.getVectors.show()

    def showSynonyms(target_word: String): Unit ={
      println("Synonyms of %s".format(target_word))
      model.findSynonyms(target_word, 10).show()
    }

    showSynonyms("τσίπρας")
    showSynonyms("μητσοτάκης")
    showSynonyms("πασοκ")
    showSynonyms("ελλάδα")
    showSynonyms("βουλή")
    showSynonyms("ευρώπη")
  }
}
