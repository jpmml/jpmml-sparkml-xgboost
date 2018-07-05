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

import java.util.function.Predicate;

import org.dmg.pmml.FieldName;
import org.jpmml.evaluator.Batch;
import org.jpmml.evaluator.FloatEquivalence;
import org.jpmml.evaluator.IntegrationTest;
import org.junit.Test;

public class XGBoostTest extends IntegrationTest {

	public XGBoostTest(){
		super(new FloatEquivalence(2));
	}

	@Test
	public void evaluateAudit() throws Exception {
		evaluate("XGBoost", "Audit", new FloatEquivalence(128));
	}

	@Test
	public void evaluateAuto() throws Exception {
		evaluate("XGBoost", "Auto");
	}

	@Override
	protected Batch createBatch(String name, String dataset, Predicate<FieldName> predicate){
		predicate = excludeFields(FieldName.create("prediction"), FieldName.create("pmml(prediction)"));

		return super.createBatch(name, dataset, predicate);
	}
}