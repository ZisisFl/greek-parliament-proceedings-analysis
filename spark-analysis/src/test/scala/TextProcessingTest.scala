import auth.dws.bigdata.common.{DataHandler, StopWords, TextProcessing}
import org.apache.spark.sql.SparkSession
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class TextProcessingTest extends AnyFlatSpec {
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

  "test" should "split text in sentences" in {
    val text =
      """Εάν βιάζονται οι φίλοι του ΠΑΣΟΚ να τις διώξουν, αν βιάζονται να υπογραφεί η συμφωνία, πάλι να πάρουν θέση.
        |Η Κυβέρνηση του κ. Τζαννετάκη είπε, εγώ το θέμα αυτό το παγώνω. Εσείς τι θέση παίρνετε; Σεις που υπογράψατε
        |μια συμφωνία ότι σε 5 χρόνια φεύγουν οι βάσεις; Έφυγαν οι βάσεις; Καινούργιες διαπραγματεύσεις κάνετε για
        |την παραμονή των βάσεων; Αφήστε τα λοιπόν αυτά.   Θα παγώσει το θέμα του Ρασίντ. Δεν ξέρω αν αυτό το θεωρείτε
        |ακυβερνησία. Αυτά, λέμε, θα παγώσουν. Και επί τέλους, αν νομίζετε ότι πρέπει να ξεκαθαρίσει αν θα έχουμε
        |ακυβερνησία ή εφαρμογή του προγράμματος της Νέας Δημοκρατίας, γιατί όπως λέει ο Λαός μας"" δυο καρπούζια
        |στην ίδια μασχάλη δεν κρατιούνται""... """.stripMargin.replaceAll("\n", " ")

    text.replaceAll("κ. ", " ")
      .split("(?<=.[.;]) +(?=[Α-ΩΆΈΌΊΏΉΎΫΪ́])")
      .foreach(println(_))

  }
}
