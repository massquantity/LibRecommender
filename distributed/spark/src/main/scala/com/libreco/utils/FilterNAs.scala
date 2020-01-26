package com.libreco.utils

import org.apache.spark.sql.{DataFrame, Dataset, Column}
import org.apache.spark.sql.functions.col


object FilterNAs {
  def filter(data: DataFrame): DataFrame = {
    println(s"find and filter NAs for each column...")
    data.columns.foreach(col => println(s"$col -> ${data.filter(data(col).isNull).count}"))
    val allCols: Array[Column] = data.columns.map(col) // col is a function to get Column
    val nullFilter: Column = allCols.map(_.isNotNull).reduce(_ && _)
    data.select(allCols: _*).filter(nullFilter)
  }
}
