name := "spark-analysis"

version := "0.1"

scalaVersion := "2.12.8"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.12" % "3.0.1",
  "org.apache.spark" % "spark-sql_2.12" % "3.0.1",

  "org.scalatest" %% "scalatest-flatspec" % "3.3.0-SNAP3" % Test,
  "org.scalatest" %% "scalatest" % "3.3.0-SNAP3" % Test,
)
