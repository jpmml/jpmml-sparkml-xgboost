import java.nio.file.{Files, Paths}

import ml.dmlc.xgboost4j.scala.spark.XGBoostEstimator
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.jpmml.sparkml.PMMLBuilder

var df = spark.read.option("header", "true").option("inferSchema", "true").csv("csv/Audit.csv")
df = df.withColumn("AdjustedTmp", df("Adjusted").cast(StringType)).drop("Adjusted").withColumnRenamed("AdjustedTmp", "Adjusted")

val formula = new RFormula().setFormula("Adjusted ~ Age + Income + Gender + Deductions + Hours")

var estimator = new XGBoostEstimator(Map("objective" -> "binary:logistic"))
estimator = estimator.set(estimator.round, 101)

val pipeline = new Pipeline().setStages(Array(formula, estimator))
val pipelineModel = pipeline.fit(df)

val vectorToColumn = udf{ (x:DenseVector, index: Int) => x(index).toFloat }

var xgbDf = pipelineModel.transform(df)
xgbDf = xgbDf.selectExpr("prediction as Adjusted", "probabilities")
xgbDf = xgbDf.withColumn("AdjustedTmp", xgbDf("Adjusted").cast(IntegerType).cast(StringType)).drop("Adjusted").withColumnRenamed("AdjustedTmp", "Adjusted")
xgbDf = xgbDf.withColumn("probability(0)", vectorToColumn(xgbDf("probabilities"), lit(0))).withColumn("probability(1)", vectorToColumn(xgbDf("probabilities"), lit(1))).drop("probabilities")
xgbDf.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("csv/XGBoostAudit.csv")

val pmmlBytes = new PMMLBuilder(df.schema, pipelineModel).buildByteArray()
Files.write(Paths.get("pmml/XGBoostAudit.pmml"), pmmlBytes)
