JPMML-SparkML-XGBoost
=====================

JPMML-SparkML plugin for converting [XGBoost4J-Spark](https://github.com/dmlc/xgboost/tree/master/jvm-packages) models to PMML.

# Prerequisites #

* [Apache Spark](http://spark.apache.org/) 2.3.2.
* [XGBoost4J-Spark](https://github.com/dmlc/xgboost/tree/master/jvm-packages) 0.82.

# Installation #

Enter the project root directory and build using [Apache Maven](http://maven.apache.org/):
```
mvn clean install
```

The build installs JPMML-SparkML-XGBoost library into local repository using coordinates `org.jpmml:jpmml-sparkml-xgboost:1.0-SNAPSHOT`.

# Usage #

The JPMML-SparkML-XGBoost library extends the [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library with support for `ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel` and `ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel` prediction model classes.

Launch the Spark shell with **XGBoost-extended** JPMML-SparkML-Package; use `--packages` to include the XGBoost4J-Spark runtime dependency:
```
spark-shell --packages ml.dmlc:xgboost4j-spark:0.82 --jars jpmml-sparkml-package-1.1-SNAPSHOT.jar
```

Fitting and exporting an example pipeline model:
```scala
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RFormula
import org.jpmml.sparkml.PMMLBuilder

val df = spark.read.option("header", "true").option("inferSchema", "true").csv("Iris.csv")

val formula = new RFormula().setFormula("Species ~ .")
var estimator = new XGBoostClassifier(Map("objective" -> "multi:softmax", "num_class" -> 3))
estimator = estimator.set(estimator.numRound, 11)

val pipeline = new Pipeline().setStages(Array(formula, estimator))
val pipelineModel = pipeline.fit(df)

val pmml = new PMMLBuilder(df.schema, pipelineModel).buildByteArray()
println(new String(pmmlBytes, "UTF-8"))
```

# License #

JPMML-SparkML-XGBoost is licensed under the [GNU Affero General Public License (AGPL) version 3.0](http://www.gnu.org/licenses/agpl-3.0.html). Other licenses are available on request.

# Additional information #

Please contact [info@openscoring.io](mailto:info@openscoring.io)
