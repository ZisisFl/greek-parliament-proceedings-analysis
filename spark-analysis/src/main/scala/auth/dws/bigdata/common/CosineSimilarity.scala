package auth.dws.bigdata.common

import org.apache.spark.ml.linalg.Vector

object CosineSimilarity {

  def clipValue(inputValue: Double): Double = math.max(0, math.min(1, inputValue))

  def module(vec:Vector): Double ={
    math.sqrt(vec.toArray.map(math.pow(_,2)).sum)
  }

  def innerProduct(v1:Vector, v2:Vector): Double ={
    v1.toArray
      .zip(v2.toArray)
      .map(x => x._1 * x._2)
      .sum
  }

  def cosineSimilarity(v1:Vector, v2:Vector):Double ={
    val cos=innerProduct(v1, v2) / (module(v1) * module(v2))
    clipValue(cos)
  }
}
