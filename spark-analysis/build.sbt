name := "spark-analysis"

version := "0.1"

scalaVersion := "2.12.8"

resolvers += "Spark Packages Repo" at "https://repos.spark-packages.org/"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.12" % "3.0.1",
  "org.apache.spark" % "spark-sql_2.12" % "3.0.1",
  "org.apache.spark" %% "spark-mllib" % "3.0.1",

  "graphframes" % "graphframes" % "0.8.2-spark3.0-s_2.12",

  "org.scalatest" %% "scalatest-flatspec" % "3.3.0-SNAP3" % Test,
  "org.scalatest" %% "scalatest" % "3.3.0-SNAP3" % Test,
)