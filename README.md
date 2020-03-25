JPMML-SparkML-XGBoost
=====================

JPMML-SparkML plugin for converting [XGBoost4J-Spark](https://github.com/dmlc/xgboost/tree/master/jvm-packages) models to PMML.

# Prerequisites #

* [Apache Spark](http://spark.apache.org/) 2.3.X or 2.4.X.
* [XGBoost4J-Spark](https://github.com/dmlc/xgboost/tree/master/jvm-packages) 0.7 or newer.

# Installation #

Enter the project root directory and build using [Apache Maven](http://maven.apache.org/):
```
mvn clean install
```

The build installs JPMML-SparkML-XGBoost library into local repository using coordinates `org.jpmml:jpmml-sparkml-xgboost:1.0-SNAPSHOT`.

# Usage #

The JPMML-SparkML-XGBoost library extends the [JPMML-SparkML](https://github.com/jpmml/jpmml-sparkml) library with support for `ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel` and `ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel` prediction model classes.

Launch the Spark shell; use the `--packages` command-line option to include XGBoost4J-Spark, JPMML-SparkML and JPMML-XGBoost runtime dependencies, and the `--jars` command-line option to include the JPMML-SparkML-XGBoost runtime dependency:
```
spark-shell --packages ml.dmlc:xgboost4j-spark:0.90,org.jpmml:jpmml-sparkml:1.5.7,org.jpmml:jpmml-xgboost:1.3.15 --jars target/jpmml-sparkml-xgboost-1.0-SNAPSHOT.jar
```

Fitting and exporting an example pipeline model:
```scala
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.RFormula
import org.jpmml.sparkml.PMMLBuilder

val df = spark.read.option("header", "true").option("inferSchema", "true").csv("Iris.csv")

val formula = new RFormula().setFormula("Species ~ .")
var classifier = new XGBoostClassifier(Map("objective" -> "multi:softmax", "num_class" -> 3))
classifier = classifier.set(classifier.numRound, 11)

val pipeline = new Pipeline().setStages(Array(formula, classifier))
val pipelineModel = pipeline.fit(df)

val pmmlBytes = new PMMLBuilder(df.schema, pipelineModel).buildByteArray()
println(new String(pmmlBytes, "UTF-8"))
```

# License #

JPMML-SparkML-XGBoost is licensed under the terms and conditions of the [GNU Affero General Public License, Version 3.0](https://www.gnu.org/licenses/agpl-3.0.html).

If you would like to use JPMML-SparkML-XGBoost in a proprietary software project, then it is possible to enter into a licensing agreement which makes JPMML-SparkML-XGBoost available under the terms and conditions of the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause) instead.

# Additional information #

JPMML-SparkML-XGBoost is developed and maintained by Openscoring Ltd, Estonia.

Interested in using [Java PMML API](https://github.com/jpmml) software in your company? Please contact [info@openscoring.io](mailto:info@openscoring.io)
