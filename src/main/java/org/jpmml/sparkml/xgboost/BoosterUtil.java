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

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import com.google.common.base.Function;
import ml.dmlc.xgboost4j.scala.Booster;
import org.dmg.pmml.DataType;
import org.dmg.pmml.Output;
import org.dmg.pmml.OutputField;
import org.dmg.pmml.mining.MiningModel;
import org.dmg.pmml.mining.Segment;
import org.dmg.pmml.mining.Segmentation;
import org.dmg.pmml.mining.Segmentation.MultipleModelMethod;
import org.dmg.pmml.regression.RegressionModel;
import org.jpmml.converter.BinaryFeature;
import org.jpmml.converter.ContinuousFeature;
import org.jpmml.converter.Feature;
import org.jpmml.converter.Schema;
import org.jpmml.xgboost.Learner;
import org.jpmml.xgboost.XGBoostUtil;
import org.spark_project.guava.collect.Iterables;

public class BoosterUtil {

	private BoosterUtil(){
	}

	static
	public MiningModel encodeBooster(Booster booster, Schema schema){
		byte[] bytes = booster.toByteArray();

		Learner learner;

		try(InputStream is = new ByteArrayInputStream(bytes)){
			learner = XGBoostUtil.loadLearner(is);
		} catch(IOException ioe){
			throw new RuntimeException(ioe);
		}

		Function<Feature, Feature> function = new Function<Feature, Feature>(){

			@Override
			public Feature apply(Feature feature){

				if(feature instanceof BinaryFeature){
					BinaryFeature binaryFeature = (BinaryFeature)feature;

					return binaryFeature;
				} else

				{
					ContinuousFeature continuousFeature = feature.toContinuousFeature(DataType.FLOAT);

					return continuousFeature;
				}
			}
		};

		Schema xgbSchema = schema.toTransformedSchema(function);

		return learner.encodeMiningModel(xgbSchema);
	}

	static
	public MiningModel relocateOutputFields(MiningModel miningModel){
		Segmentation segmentation = miningModel.getSegmentation();

		MultipleModelMethod multipleModelMethod = segmentation.getMultipleModelMethod();
		switch(multipleModelMethod){
			case MODEL_CHAIN:
				break;
			default:
				return miningModel;
		}

		Output sparkOutput = miningModel.getOutput();
		if(sparkOutput == null || !sparkOutput.hasOutputFields()){
			return miningModel;
		}

		List<Segment> segments = segmentation.getSegments();

		Segment lastSegment = Iterables.getLast(segments);

		RegressionModel lastModel = (RegressionModel)lastSegment.getModel();

		Output xgbOutput = lastModel.getOutput();
		if(xgbOutput == null){
			xgbOutput = new Output();

			lastModel.setOutput(xgbOutput);
		}

		List<OutputField> outputFields = xgbOutput.getOutputFields();
		outputFields.addAll(0, sparkOutput.getOutputFields());

		miningModel.setOutput(null);

		return miningModel;
	}
}