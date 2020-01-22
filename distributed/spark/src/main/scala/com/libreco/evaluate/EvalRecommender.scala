package com.libreco.evaluate

import org.apache.spark.internal.Logging
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.mllib.evaluation.RankingMetrics
import org.apache.spark.sql.functions.{collect_list, expr, row_number}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.rdd.RDD
import com.libreco.data.DataSplitter
import com.libreco.utils.Context


class EvalRecommender(num: Int = 10,
                      metrics: String = "ndcg") extends Logging with Serializable with Context {
  import spark.implicits._

  def eval(data: DataFrame): Unit = {
    val splitter = new DataSplitter()
    val Array(trainData, testData) = splitter.stratified_chrono_split(data, 0.8, "user")
  //  trainData.cache()
  //  testData.cache()

    val nFactors = Seq(20, 50, 100)
    val nRegs = Seq(0.001, 0.01, 0.1)
    var round = 1
    var bestScore: Double = 0
    var bestParams: Map[String, Double] = Map("nFactors" -> 0, "nRegs" -> -1.0)

    nFactors.foreach { f =>
      nRegs.foreach { r =>
        println(s"round $round start...")
        val als = new ALS()
          .setMaxIter(20)
          .setRegParam(r)
          .setUserCol("user")
          .setItemCol("item")
          .setRank(f)
          .setImplicitPrefs(true)
          .setRatingCol("label")

        val model = als.fit(trainData)
        model.setColdStartStrategy("drop")
        val pred = model.recommendForUserSubset(testData, num)
          // .select("user", "recommendations.item", "recommendations.rating")
          .selectExpr("user", "explode(recommendations) as predAndProb")
          .select("user", "predAndProb.*").toDF("user", "item", "prob")

        val score = metrics match {
          case "precision" => precisionAtK(testData, pred, num)
          case "map" => meanAveragePrecision(testData, pred, num)
          case "recall" => recallAtK(testData, pred, num)
          case "ndcg" => ndcgAtK(testData, pred, num)
          case _ =>
            println("please choose the correct metrics...")
            0.0
        }

        if (score > bestScore) {
          bestScore = score
          bestParams += ("nFactors" -> f, "nRegs" -> r)
        }
        println(s"\tnFactors = $f, nReg = $r, score = $score")
        println()
        round += 1
      }
    }
    println(s"best params: nFactors = ${bestParams("nFactors")}, nReg = ${bestParams("nRegs")}")
  }

  def getPredTruePair(labels: DataFrame,
                      preds: DataFrame,
                      k: Int): RDD[(Array[Integer], Array[Integer])] = {
    val userTrue = labels.groupBy("user").agg(expr("collect_set(item) as trueItems"))
    //  val userPred = preds.orderBy($"user", $"prediction".desc)
    //    .groupBy("user").agg(expr("collect_list(item) as predItems"))

    val windowSpec = Window.partitionBy("user").orderBy($"prob".desc)
    val userPred = preds.select($"user", $"item", row_number().over(windowSpec).alias("rank"))
      .where($"rank" <= k)
      .groupBy("user")
      .agg(collect_list($"item").alias("predItems"))

    userPred.join(userTrue, Seq("user")).map { row => (
      row(1).asInstanceOf[Seq[Integer]].toArray,
      row(2).asInstanceOf[Seq[Integer]].toArray
    )}.rdd
  }

  def precisionAtK(labels: DataFrame, preds: DataFrame, k: Int): Double = {
    val userPredAndTrue = getPredTruePair(labels, preds, k)
    val rm = new RankingMetrics[Integer](userPredAndTrue)
    rm.precisionAt(k)
  }

  def meanAveragePrecision(labels: DataFrame, preds: DataFrame, k: Int): Double = {
    val userPredAndTrue = getPredTruePair(labels, preds, k)
    val rm = new RankingMetrics[Integer](userPredAndTrue)
    rm.meanAveragePrecision
  }

  def ndcgAtK(labels: DataFrame, preds: DataFrame, k: Int): Double = {
    val userPredAndTrue = getPredTruePair(labels, preds, k)
    val rm = new RankingMetrics[Integer](userPredAndTrue)
    rm.ndcgAt(k)
  }

  def recallAtK(labels: DataFrame, preds: DataFrame, k: Int): Double = {
    require(k > 0, "rank position k must be positive.")

    val userPredAndTrue = getPredTruePair(labels, preds, k)
    userPredAndTrue.map { case (pred, label) =>
      val labelSet = label.toSet
        if (labelSet.nonEmpty) {
          val n = math.min(pred.length, k)
          var i, count = 0
          while (i < n) {
            if (labelSet.contains(pred(i)))
              count += 1
            i += 1
          }
          count.toDouble / labelSet.size
        } else {
          logWarning("Empty ground truth set, check input data")
          0.0
        }
    }.mean()
  }
}


