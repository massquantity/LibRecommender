package com.libreco.model

import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.DataFrame


object FeatureEngineering {
  def preProcessPipeline(dataset: DataFrame): Array[PipelineStage] = {
  //  val dataColumns = dataset.columns diff Array("rating", "user_id", "anime_id", "name")
    val continuousFeatures = Array("episodes", "web_rating", "members")  // name, genre
    val categoricalFeatures = Array("type")

    // deal with  continuous features
    val continuousFeatureAssembler = new VectorAssembler(uid = "continuous_feature_assembler")
      .setInputCols(continuousFeatures)
      .setOutputCol("unscaled_continuous_features")
    val continuousFeatureScaler = new StandardScaler(uid = "continuous_feature_scaler")
      .setInputCol("unscaled_continuous_features")
      .setOutputCol("scaled_continuous_features")
      .setWithMean(true)
      .setWithStd(true)

    // deal with categorical features
    val categoricalFeatureIndexers = categoricalFeatures.map {feature =>
      new StringIndexer(uid = s"string_indexer_$feature")
        .setInputCol(feature)
        .setOutputCol(s"${feature}_index")
    }
    val categoricalIndexerCols = categoricalFeatureIndexers.map(_.getOutputCol)
    val categoricalFeatureEncoder = new OneHotEncoderEstimator(uid = "one_hot_encoder")
      .setInputCols(categoricalIndexerCols)
      .setOutputCols(Array("one_hot_vector"))
      .setHandleInvalid("keep")  //// keep

    // deal with textual features
    val regexTokenizer = new RegexTokenizer(uid = "regex_tokenizer")
      .setInputCol("name")
      .setOutputCol("words")
      .setPattern("\\w+")
      .setGaps(false)
      .setToLowercase(true)
    val word2Vec = new Word2Vec(uid = "word2vec")
      .setInputCol("words")
      .setOutputCol("word_vectors")
      .setMinCount(0)
      .setVectorSize(20)
      .setSeed(2020L)

    // assemble all features
    val featureCols = categoricalFeatureEncoder.getOutputCols
      .union(Seq("scaled_continuous_features"))
      .union(Seq("word_vectors"))
    val assembler = new VectorAssembler(uid = "feature_assembler")
      .setInputCols(featureCols)
      .setOutputCol("featureVector")
    println(s"featureVector length: ${assembler.getOutputCol.length}")

    val estimators = Array(continuousFeatureAssembler, continuousFeatureScaler) ++
      categoricalFeatureIndexers ++
      Seq(categoricalFeatureEncoder) ++
      Seq(regexTokenizer, word2Vec) ++
      Seq(assembler)
    estimators.asInstanceOf[Array[PipelineStage]]
  }
}
