package com.fyp.machineLearning.test;


import java.io.*;

import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.*;
import weka.filters.unsupervised.attribute.Remove;

public class using_model_NN {

    public static void main(String[] args) {
        String filepath = System.getProperty("user.dir")+"\\data\\agri.arff";
        simpleWekaTrain(filepath);
    }

    //just to run a simple neural network
    public static void simpleWekaTrain(String filepath) {
        int kfolds = 10;
        System.out.println("Starting the training");
        try {
    //Reading training arff or csv file
            FileReader trainreader = new FileReader(filepath);
            Instances train = new Instances(trainreader);
            train.setClassIndex(train.numAttributes() - 1);
    //Instance of NN
            MultilayerPerceptron mlp = new MultilayerPerceptron();
    //Setting Parameters
            mlp.setLearningRate(0.1);
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(2000);
            mlp.setHiddenLayers("3");
            mlp.buildClassifier(train);

            System.out.println("Ending training...");

            //evaluation
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(mlp, train);

            //output result to console
            System.out.println("Training Mean root squared Error: "+eval.errorRate()); //Printing Training Mean root squared Error
            System.out.println("Summary of Training: \n"+eval.toSummaryString()); //Summary of Training


            //k-fold validation
            eval.crossValidateModel(mlp, train, kfolds, new Debug.Random(1));

            //using test data
            String testFilePath = System.getProperty("user.dir")+"\\data\\agri_test_dataset.arff";
            Instances dataPredict = new Instances(
                    new BufferedReader(
                            new FileReader(testFilePath)));
            dataPredict.setClassIndex(dataPredict.numAttributes() - 1);
            Instances predictedData = new Instances(dataPredict);
            //Predict Part
            for (int i = 0; i < dataPredict.numInstances(); i++) {
                double clsLabel = mlp.classifyInstance(dataPredict.instance(i));
                predictedData.instance(i).setClassValue(clsLabel);
            }

            //evaluation - second round
            //Evaluation eval2 = new Evaluation(predictedData);
            eval.evaluateModel(mlp, predictedData);

            //output result to console
            System.out.println("2) Training Mean root squared Error: "+eval.errorRate()); //Printing Training Mean root squared Error
            System.out.println("2) Summary of Training: \n"+eval.toSummaryString()); //Summary of Training


            //k-fold validation
            eval.crossValidateModel(mlp, predictedData, kfolds, new Debug.Random(1));

            //print out predictions
            StringBuffer predsBuffer = new StringBuffer();
            PlainText plainText = new PlainText();
            //plainText.setHeader(predictedData);
            plainText.setBuffer(predsBuffer);

            //eval.evaluateModel(mlp, predictedData, plainText);
            eval.crossValidateModel(mlp, predictedData, kfolds, new Debug.Random(1),plainText);
            System.out.println("Final output prediction");
            System.out.println(predsBuffer.toString());

        } catch (Exception ex) {
            ex.printStackTrace();
        }
        System.out.println("Ending the method");



    }

}
