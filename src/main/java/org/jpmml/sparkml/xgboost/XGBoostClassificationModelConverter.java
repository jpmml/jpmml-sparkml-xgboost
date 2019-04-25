/*
 * Copyright (c) 2017 Villu Ruusmann
 *
 * This file is part of JPMML-SparkML
 *
 * JPMML-SparkML is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * JPMML-SparkML is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with JPMML-SparkML.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.jpmml.sparkml.xgboost;

import ml.dmlc.xgboost4j.scala.Booster;
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel;
import org.dmg.pmml.mining.MiningModel;
import org.jpmml.converter.Schema;
import org.jpmml.sparkml.ClassificationModelConverter;

public class XGBoostClassificationModelConverter extends ClassificationModelConverter/*<XGBoostClassificationModel>*/ implements HasXGBoostOptions {

	public XGBoostClassificationModelConverter(XGBoostClassificationModel model){
		super(model);
	}

	@Override
	public MiningModel encodeModel(Schema schema){
		XGBoostClassificationModel model = (XGBoostClassificationModel)getTransformer();

		Booster booster = model.nativeBooster();

		return BoosterUtil.encodeBooster(this, booster, schema);
	}
}