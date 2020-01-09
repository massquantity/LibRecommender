package libreco.model

import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer

object FeatureEngineering {
  def preProcessPipeline(dataset: DataFrame): Array[PipelineStage] = {
    var data = dataset
    val dataColumns = data.columns diff Array("rating", "user_id", "anime_id", "name")
    val otherCols = Seq("episodes", "web_rating", "members")
    val cateCols = Set(dataColumns: _*) -- otherCols -- Seq("genre")

    val pipelineStages = ArrayBuffer[PipelineStage]()
    val stringIndexerCols = ArrayBuffer[String]()
    cateCols.foreach { col =>
      val (indexer, vecCol) = StringIndexerPipeline(col)
      stringIndexerCols += vecCol
      pipelineStages += indexer
    }

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(stringIndexerCols.toArray)
      .setOutputCols(Array("OneHotVector"))
    pipelineStages += encoder

    val assembler = new VectorAssembler()
      .setInputCols(Array("OneHotVector") ++ otherCols)
      .setOutputCol("featureVector")
    // assembler.getInputCols.foreach(x => print(x + " "))
    // pipeline.getStages.foreach(println)
    println(s"featureVector length: ${assembler.getInputCols.length}")
    //  data.schema.fields.foreach(println)

    //  val pipeline = new Pipeline().setStages(pipelineStages.toArray ++ Array(assembler))
    pipelineStages.toArray ++ Array(assembler)

  }


  def StringIndexerPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer()
      .setInputCol(inputCol)
      .setOutputCol(inputCol + "_indexed")
    val pipeline = new Pipeline().setStages(Array(indexer))
    (pipeline, indexer.getOutputCol)
  }
}
