package com.libreco.model

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession, Column}
import org.apache.log4j.{Level, Logger}
import com.libreco.data.DataSplitter
import com.libreco.evaluate.EvalRecommender
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.functions.{split, explode, struct, udf}
import com.libreco.utils.Context

class Recommender extends Serializable with Context{
  var model: ALSModel = _

  def train(df: DataFrame, evaluate: Boolean = false): Unit = {
    if (evaluate) {
      val splitter = new DataSplitter()
      val Array(trainData, testData) = splitter.stratified_chrono_split(df, 0.8, "user")
      trainData.cache()
      testData.cache()



    }
    else {
      df.cache()
      val als = new ALS()
        .setMaxIter(5)
        .setRegParam(0.01)
        .setUserCol("user")
        .setItemCol("item")
        .setRank(50)
        .setImplicitPrefs(true)
        .setRatingCol("label")
      model = als.fit(df)
      model.setColdStartStrategy("drop")
      df.unpersist()
    }
  }

  def transform(df: DataFrame): DataFrame = {
    model.transform(df)
  }

  def recommendForUsers(df: DataFrame, num: Int): DataFrame = {
    model.recommendForUserSubset(df, num)
      .selectExpr("user", "explode(recommendations) as predAndProb")
      .select("user", "predAndProb.*").toDF("user", "item", "prob") // pred means specific item
  }
}
