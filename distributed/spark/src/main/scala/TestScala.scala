import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCols}
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.functions.{array_contains, col, split}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.util.Identifiable


object TestScala {
  def main(args: Array[String]): Unit = {
    printf("test scala...")
  }
}
