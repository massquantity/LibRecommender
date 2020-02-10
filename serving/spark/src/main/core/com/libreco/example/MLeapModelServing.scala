package com.libreco.example

import com.libreco.serving.mleap.ModelServer
import ml.combust.mleap.core.types.{ScalarType, StructField, StructType}
import ml.combust.mleap.runtime.frame.Row
import ml.combust.mleap.tensor.{DenseTensor, Tensor}


object MLeapModelServing {
  def main(args: Array[String]): Unit = {
    val dataSchema = StructType(
      StructField("name", ScalarType.String),
      StructField("type", ScalarType.String),
      StructField("episodes", ScalarType.Int),
      StructField("web_rating", ScalarType.Double),
      StructField("members", ScalarType.Int)
    ).get

    val features = Seq(Row("Gintama", "TV", 13, 8.8, 100000),
                       Row("Fate/Zero 2nd Season", "TV", 24, 7.7, 40000),
                       Row("Hs", "OVA", 100, 9.0, 10000))
    val modelPath = "jar:" + this.getClass.getResource("/mleap_model/mleap_model.zip").toString
    val modelServer = new ModelServer(modelPath, dataSchema)
    val result: Seq[Row] = modelServer.predict(features)
  /*
    for (i <- result.indices)
      for (j <- 0 until result(i).size)
        println(j, result(i).get(j), '\n')
  */
    for (i <- result.indices) {
      println(s"result for sample ${i + 1}: ")
      val pred = result(i).getDouble(14)
      println(s"prediction: $pred")
      val probs: DenseTensor[Double] = result(i).getTensor(13).toDense
      val prob0 = probs.get(0).head
      val prob1 = probs.get(1).head
      val prob2 = probs.get(2).head
      println(s"probabilities(size = ${probs.size}): ")
      println("label  prob")
      println(f" 0.0   $prob0%.2f")
      println(f" 1.0   $prob1%.2f")
      println(f" 2.0   $prob2%.2f%n")
    }
  }
}
