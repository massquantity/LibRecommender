package com.libreco.utils

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.udf

import scala.collection.Map
import scala.util.matching.Regex


object ItemNameConverter extends Context{
  import spark.implicits._

  def getId2ItemName(): Map[Int, String] = {
    val itemDataPath = this.getClass.getResource("/ml-1m/movies.dat").toString
    val itemData: Dataset[String] = spark.read.textFile(itemDataPath)

    itemData.flatMap { line: String =>
      val Array(id, movieName, _*): Array[String] = line.split("::")
      if (id.isEmpty) {
        None
      } else {
        val pattern = new Regex("(.+)(\\(\\d+\\))")
        val name = for (m <- pattern.findFirstMatchIn(movieName)) yield m.group(1)
        Some(id.toInt, name.mkString)
      }
    }.collect().toMap
  }

  def getItemName(): UserDefinedFunction = {
    val itemDataPath = this.getClass.getResource("/ml-1m/movies.dat").toString
    val itemData: Dataset[String] = spark.read.textFile(itemDataPath)

    val itemMap = itemData.map { line: String =>
      val Array(_, movieName, _*): Array[String] = line.split("::")
      val pattern = new Regex("(.+)(\\(\\d+\\))")
      val name = for (m <- pattern.findFirstMatchIn(movieName)) yield m.group(1)
      (movieName, name.mkString)
    }.collect().toMap
    udf((origName: String) => itemMap(origName))
  }

  def main(args: Array[String]): Unit = {
    val mm = getId2ItemName()
    for (i <- 1 to 5) {
      println(s"Id2ItemName: $i -> ${mm(i)}")
    }
  }
}
