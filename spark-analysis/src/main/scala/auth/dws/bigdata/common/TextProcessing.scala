package auth.dws.bigdata.common

object TextProcessing {
  def removeStopWords(input_text: String): String ={
    val stop_words = StopWords.loadStopWords
    input_text.split(" ").filterNot(token => stop_words.contains(token)).mkString(" ").trim
  }

  def removeNonGreekCharacters(input_text: String): String ={
    input_text.replaceAll("[^Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]+", " ")
  }

  def removeNonCharacters(input_text: String): String ={
    input_text.replaceAll("[^a-zA-ZΑ-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]+", " ")
  }

  def normalizeApostrofos(input_text: String): String ={
    input_text
  }

  def removeIntonation(input_text: String): String = {
    val substitutions = Map(
      "ά" -> "α",
      "έ" -> "ε",
      "ή" -> "η",
      "ἡ" -> "η",
      "ί" -> "ι",
      "ἰ" -> "ι",
      "ϊ" -> "ι",
      "ΐ" -> "ι",
      "ό" -> "ο",
      "ύ" -> "υ",
      "ϋ" -> "υ",
      "ῦ" -> "υ",
      "ῦ" -> "υ",
      "ώ" -> "ω",
      "ῶ" -> "ω"
    )
    substitutions.foldLeft(input_text.toLowerCase()) { case (cur, (from, to)) => cur.replaceAll(from, to)}
  }

//  https://stackoverflow.com/questions/42039355/scala-apply-list-of-functions-to-a-object/42040011
//  val f1 = (input_string: String) => input_string.replaceAll("[^Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]+", " ")
//  val f2 = (input_string: String) => input_string.replaceAll("[^Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]+", " ")
//  //Function.chain()
//  val yo = Seq(f1, f2)
//
//  val item = yo.reduce((a,b) => a.andThen(b))("το τεστ")
}
