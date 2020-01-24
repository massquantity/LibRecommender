package com.libreco.model

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.functions.{coalesce, typedLit}
import com.libreco.utils.Context
import com.libreco.evaluate.EvalRecommender

import scala.collection.Map

class Recommender extends Serializable with Context{
  import spark.implicits._
  var model: ALSModel = _

  def train(df: DataFrame, evaluate: Boolean = false, num: Int = 10): Unit = {
    df.cache()
    if (evaluate) {
      val evalModel = new EvalRecommender(num, "ndcg")
      evalModel.eval(df)
    }
    else {
      val als = new ALS()
        .setMaxIter(20)
        .setRegParam(0.01)
        .setUserCol("user")
        .setItemCol("item")
        .setRank(50)
        .setImplicitPrefs(true)
        .setRatingCol("label")
      model = als.fit(df)
      model.setColdStartStrategy("drop")
    }
    df.unpersist()
  }

  def transform(df: DataFrame): DataFrame = {
    model.transform(df)
  }

  def recommendForUsers(df: DataFrame,
                        num: Int,
                        ItemNameMap: Map[Int, String] = Map.empty): DataFrame = {

    val rec = model.recommendForUserSubset(df, num)
        .selectExpr("user", "explode(recommendations) as predAndProb")
        .select("user", "predAndProb.*").toDF("user", "item", "prob") // pred means specific item

    val nameMapCol = typedLit(ItemNameMap)
    rec.withColumn("name", coalesce(nameMapCol($"item")))
  }
}
