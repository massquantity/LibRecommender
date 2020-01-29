package com.libreco.feature

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCols}
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.functions.{array_contains, col, split, trim}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable.ArrayBuffer


class MultiHotEncoder(override val uid: String) extends Transformer
  with HasInputCol with HasOutputCols with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("multiHotEncoder"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  override def transformSchema(schema: StructType): StructType = {
    var outputFields = ArrayBuffer[StructField]()
    schema.fields.foreach(f => outputFields += f)
    $(outputCols).foreach(o => outputFields += StructField(o, IntegerType, nullable = false))
    StructType(outputFields.toArray)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    var data = dataset
    $(outputCols).foreach { g =>
      data = data.withColumn(g, array_contains(split(trim(col($(inputCol))), "\\|"), g).cast("int"))
    }
    data.toDF()
  }

  override def copy(extra: ParamMap): MultiHotEncoder = defaultCopy(extra)
}


object MultiHotEncoder {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("com").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("MultiHotEncoder")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder
      .config(conf)
      .getOrCreate()
    import spark.implicits._

    var data = spark.read.textFile("distributed/spark/src/main/resources/ml-1m/movies.dat")
      .map {line =>
        val Array(id, name, genre) = line.split("::")
        (id, name, genre)}
      .toDF("id", "name", "genre")
    data.show(4, truncate = false)

    val genreList = data
      .select("genre")
      .rdd
      .map(_.getAs[String]("genre"))
      .flatMap(_.trim().split("\\|"))
      .distinct
      .collect()
    genreList.foreach(x => print(x + " "))

    val multihot = new MultiHotEncoder(uid = "multi_hot_encoder")
      .setInputCol("genre")
      .setOutputCols(genreList)
    val pipeline = new Pipeline().setStages(Array(multihot))
    val pipelineModel = pipeline.fit(data)
    pipelineModel.transform(data).show(4, truncate = false)
  }
}

