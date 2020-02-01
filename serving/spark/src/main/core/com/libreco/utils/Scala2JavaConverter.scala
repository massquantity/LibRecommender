package com.libreco.utils

import ml.combust.mleap.tensor.Tensor

object Scala2JavaConverter {
  def parseTensor(prob: Tensor[Double], i: Int): java.lang.Double = {
    prob.get(i).head
  }
}
