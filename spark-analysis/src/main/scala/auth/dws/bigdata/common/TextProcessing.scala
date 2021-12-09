package auth.dws.bigdata.common

object TextProcessing {
  def conditionWhitespaces(input_text: String): String ={
    input_text.trim.replaceAll(" +", " ")
  }

  def removeStopWords(input_text: String): String ={
    val stop_words = StopWords.loadStopWords
    input_text.split(" ").filterNot(token => stop_words.contains(token)).mkString(" ")
  }

  def removeNonGreekCharacters(input_text: String): String ={
    input_text.replaceAll("[^Α-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]+", " ")
  }

  def removeNonCharacters(input_text: String): String ={
    input_text.replaceAll("[^a-zA-ZΑ-ΩΆΈΌΊΏΉΎΫΪ́α-ωάέόίώήύϊΐϋΰ]+", " ")
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

  def textProcessingPipeline(input_text: String): String = {
    val lower_text = input_text.toLowerCase()
    val text_with_removedIntonation = removeIntonation(lower_text)
    val text_with_removedStopWords = removeStopWords(text_with_removedIntonation)
    val text_with_removedNonGreekCharacters = removeNonGreekCharacters(text_with_removedStopWords)
    val text_with_conditionedWhitespaces = conditionWhitespaces(text_with_removedNonGreekCharacters)
    text_with_conditionedWhitespaces
  }
}
