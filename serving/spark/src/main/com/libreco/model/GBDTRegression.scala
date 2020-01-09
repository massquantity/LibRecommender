package libreco.model

import java.io.{File, FileOutputStream}

import javax.xml.transform.stream.StreamResult
import libreco.Context
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.{col, when}
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.log4j.{Level, Logger}
import org.dmg.pmml.PMML
import org.jpmml.model.JAXBUtil
import org.jpmml.sparkml.PMMLBuilder

import scala.collection.mutable.ArrayBuffer

class GBDTRegression (evaluate: Boolean = false,
                      paramTuning: Boolean = false,
                      convertImplicit: Boolean = false) extends Context{
  import spark.implicits._
  var pipelineModel: PipelineModel = _
  var data: DataFrame = _

  def train(dataset: DataFrame): Unit = {
    val prePipelineStages = FeatureEngineering.preProcessPipeline(dataset)
    if (convertImplicit) {
      data = dataset.withColumn("label", when($"rating" >= 8, 1).otherwise(0))
    } else {
      data = dataset
    }
    data.cache()

    val gbr = new GBTRegressor()
      .setFeaturesCol("featureVector")
      .setLabelCol("rating")
      .setPredictionCol("pred")
      .setFeatureSubsetStrategy("auto")
      .setMaxDepth(3)
      .setMaxIter(5)
      .setStepSize(0.1)
      .setSubsamplingRate(0.8)
      .setSeed(2020L)

    val pipelineStages = prePipelineStages ++ Array(gbr)
    val pipeline = new Pipeline().setStages(pipelineStages)
    if (evaluate) {
      var Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2), 2020L)
    //  trainData.cache()
    //  testData.cache()
      pipelineModel = pipeline.fit(trainData)
    //  trainData = pipelineModel.transform(trainData)
    //  testData = pipelineModel.transform(testData)
    }
    else {
      pipelineModel = pipeline.fit(data)
    }

    data.unpersist()
  }

  def transform(dataset: DataFrame): DataFrame = {
    pipelineModel.transform(dataset)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("com").setLevel(Level.ERROR)

    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("GBDTRegression")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder
      .config(conf)
      .getOrCreate()

    val ratingSchema = new StructType(Array(
      StructField("user_id", IntegerType, nullable = false),
      StructField("anime_id", IntegerType, nullable = false),
      StructField("rating", IntegerType, nullable = false)
    ))
    val animeSchema = new StructType(Array(
      StructField("anime_id", IntegerType, nullable = false),
      StructField("name", StringType, nullable = true),
      StructField("genre", StringType, nullable = true),
      StructField("type", StringType, nullable = true),
      StructField("episodes", IntegerType, nullable = true),
      StructField("rating", DoubleType, nullable = true),
      StructField("members", IntegerType, nullable = true)
    ))

    val ratingPath = this.getClass.getResource("/anime_rating.csv").toString
    val animePath = this.getClass.getResource("/anime_info.csv").toString
    val rating = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .schema(ratingSchema)
      .csv(ratingPath)
    var anime = spark.read
      .option("inferSchema", "true")
      .option("header", "true")
      .schema(animeSchema)
      .csv(animePath)
    anime = anime.withColumnRenamed("rating", "web_rating").drop("rating")
    var data = rating.join(anime, Seq("anime_id"), "inner")
  //  println(s"find NA numbers for each column...")
  //  data.columns.foreach(x => println(s"$x -> ${data.filter(data(x).isNull).count}"))
    data = data.na.fill("Missing", Seq("genre"))
    data = data.na.fill("Missing", Seq("type"))
    data = data.na.fill(7.6, Seq("web_rating"))
    val dataColumns = data.columns diff Array("rating", "user_id", "anime_id", "name")
    val otherCols = Seq("episodes", "web_rating", "members")
    val cateCols = Set(dataColumns: _*) -- otherCols -- Seq("genre")

    val pipelineStages = ArrayBuffer[PipelineStage]()
    val vectorAassembleCols = ArrayBuffer[String]()
    cateCols.foreach { col =>
      val (encoder, vecCol) = oneHotPipeline(col)
      vectorAassembleCols += vecCol
      pipelineStages += encoder
    }

    val genreList = data
      .select("genre")
      .rdd
      .map(_.getAs[String]("genre"))
      .flatMap(_.split(", ")).distinct.collect()
    genreList.foreach(g => vectorAassembleCols += g)

    val multiHot = new MultiHotEncoder()
      .setInputCol("genre")
      .setOutputCols(genreList)
    pipelineStages += multiHot

    val assembler = new VectorAssembler()
      .setInputCols(vectorAassembleCols.toArray ++ otherCols)
      .setOutputCol("featureVector")
    val pipeline = new Pipeline().setStages(pipelineStages.toArray ++ Array(assembler))
  //  assembler.getInputCols.foreach(x => print(x + " "))
  //  println("----")
  //  pipeline.getStages.foreach(println)
    println(s"featureVector length: ${assembler.getInputCols.length}")
  //  data.schema.fields.foreach(println)

    var Array(trainData, testData) = data.randomSplit(Array(0.8, 0.2), 2020L)
    trainData.cache()  // otherwise model will fail
    testData.cache()
    val pipelineModel = pipeline.fit(trainData)
  //  pipelineModel.transform(testData).show(4)

    trainData = pipelineModel.transform(trainData)
    testData = pipelineModel.transform(testData)

    val jpmmlModelPath = "src/main/resources/jpmml_model/GBDT_model.xml"
    val pmml: PMML = new PMMLBuilder(trainData.schema, pipelineModel).build()
    val output = new FileOutputStream(new File(jpmmlModelPath))
    JAXBUtil.marshalPMML(pmml, new StreamResult(output))


    val gbr = new GBTRegressor()
      .setFeaturesCol("featureVector")
      .setLabelCol("rating")
      .setPredictionCol("pred")
      .setFeatureSubsetStrategy("auto")
      .setMaxDepth(3)
      .setMaxIter(25)
      .setStepSize(0.1)
      .setSubsamplingRate(0.8)
      .setSeed(2020L)
    val gbrModel = gbr.fit(trainData)

    val trainPredAndLabel = trainData.select("rating", "featureVector").join(
      gbrModel.transform(trainData.select("featureVector")), Seq("featureVector"))
    val testPredAndLabel = testData.select("rating", "featureVector").join(
      gbrModel.transform(testData.select("featureVector")), Seq("featureVector"))

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setPredictionCol("pred")
      .setLabelCol("rating")
    println(s"train rmse: ${evaluator.evaluate(trainPredAndLabel)}")
    println(s"test rmse: ${evaluator.evaluate(testPredAndLabel)}")
  }


  def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol + "_indexed")
      .setHandleInvalid("skip")
    val encoder = new OneHotEncoder()
      .setInputCol(inputCol + "_indexed")
      .setOutputCol(inputCol + "_vec")
    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    (pipeline, encoder.getOutputCol)
  }

}
