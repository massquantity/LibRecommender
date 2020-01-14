package com.libreco.utils

import ml.combust.mleap.tensor.Tensor

object Scala2JavaConverter {
  def parseCtr(prob:Tensor[Double]): java.lang.Double = {
    prob.get(0).head
  }
}
