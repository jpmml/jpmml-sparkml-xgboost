import java.nio.file.{Files, Paths}

import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.types.StringType
import org.jpmml.sparkml.ConverterUtil

var df = spark.read.option("header", "true").option("inferSchema", "true").csv("Auto.csv")
df = df.withColumn("originTmp", df("origin").cast(StringType)).drop("origin").withColumnRenamed("originTmp", "origin")

val formula = new RFormula().setFormula("mpg ~ .")

var estimator = new XGBoostEstimator(Map("objective" -> "reg:linear"))
estimator = estimator.set(estimator.round, 101)

val pipeline = new Pipeline().setStages(Array(formula, estimator))
val pipelineModel = pipeline.fit(df)

var xgbDf = pipelineModel.transform(df)
xgbDf = xgbDf.selectExpr("prediction as mpg")
xgbDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("XGBoostAuto.csv")

val pmmlBytes = ConverterUtil.toPMMLByteArray(df.schema, pipelineModel)
Files.write(Paths.get("XGBoostAuto.pmml"), pmmlBytes)
