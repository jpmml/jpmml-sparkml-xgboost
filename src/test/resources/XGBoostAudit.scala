import java.nio.file.{Files, Paths}

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.jpmml.sparkml.PMMLBuilder
import org.jpmml.sparkml.xgboost.SparseToDenseTransformer

var df = spark.read.option("header", "true").option("inferSchema", "true").csv("csv/Audit.csv")
df = df.withColumn("AdjustedTmp", df("Adjusted").cast(StringType)).drop("Adjusted").withColumnRenamed("AdjustedTmp", "Adjusted")

val formula = new RFormula().setFormula("Adjusted ~ .")
val sparse2dense = new SparseToDenseTransformer().setInputCol(formula.getFeaturesCol).setOutputCol("denseFeatures")

var classifier = new XGBoostClassifier(Map("objective" -> "binary:logistic", "num_round" -> 101)).setLabelCol(formula.getLabelCol).setFeaturesCol(sparse2dense.getOutputCol)

val pipeline = new Pipeline().setStages(Array(formula, sparse2dense, classifier))
val pipelineModel = pipeline.fit(df)

val vectorToColumn = udf{ (vec: Vector, index: Int) => vec(index).toFloat }

var xgbDf = pipelineModel.transform(df)
xgbDf = xgbDf.selectExpr("prediction as Adjusted", "probability")
xgbDf = xgbDf.withColumn("AdjustedTmp", xgbDf("Adjusted").cast(IntegerType).cast(StringType)).drop("Adjusted").withColumnRenamed("AdjustedTmp", "Adjusted")
xgbDf = xgbDf.withColumn("probability(0)", vectorToColumn(xgbDf("probability"), lit(0))).withColumn("probability(1)", vectorToColumn(xgbDf("probability"), lit(1))).drop("probability")

xgbDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("csv/XGBoostAudit.csv")

val pmmlBytes = new PMMLBuilder(df.schema, pipelineModel).buildByteArray()
Files.write(Paths.get("pmml/XGBoostAudit.pmml"), pmmlBytes)
