package com.libreco.serving.mleap

import ml.combust.bundle.BundleFile
import ml.combust.mleap.core.types._
import resource.managed
import ml.combust.mleap.runtime.MleapSupport._
import ml.combust.mleap.runtime.frame.{DefaultLeapFrame, Row, Transformer}


class ModelServer(private val modelPath: String,
                  private val dataSchema: StructType) {

  var _model: Transformer = _

  private def loadModel(): Unit = {
    val bundle = (for (bundleFile <- managed(BundleFile(modelPath))) yield {
      bundleFile.loadMleapBundle().get
    }).opt.get
    this._model = bundle.root
  }

  def predict(features: Seq[Row]): Seq[Row] = {
    if (this._model == null)
      loadModel()
    if (features == null) {
      throw null
    }

    val frame = DefaultLeapFrame(dataSchema, features)
    val resultFrame = _model.transform(frame).get
    resultFrame.dataset
  }
}
