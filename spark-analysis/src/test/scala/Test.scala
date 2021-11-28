import auth.dws.bigdata.common.DataHandler
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
    val df = DataHandler.createDataFrame(spark)

    df.printSchema()
  }

  "dataframe rows" should "be printed" in {
    val df = DataHandler.createDataFrame(spark)

    df.show()
  }
}
