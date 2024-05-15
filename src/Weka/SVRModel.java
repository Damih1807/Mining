package Weka;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMOreg;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import java.io.File;

public class SVRModel {
    private Instances trainData;
    private Instances evalData;
    private SMOreg model;

    public SVRModel(String arffFilePath) throws Exception {
        File arffFile = new File(arffFilePath);
        
        // Kiểm tra sự tồn tại của file ARFF
        if (!arffFile.exists()) {
            throw new Exception("File ARFF không tồn tại.");
        }

        // Load dữ liệu từ file ARFF
        ArffLoader loader = new ArffLoader();
        loader.setFile(arffFile);
        Instances data = loader.getDataSet();

        // Chia tập dữ liệu thành tập huấn luyện và tập đánh giá
        data.randomize(new java.util.Random(1));
        int trainSize = (int) Math.round(data.numInstances() * 0.8); // 80% dữ liệu cho tập huấn luyện
        int evalSize = data.numInstances() - trainSize;
        trainData = new Instances(data, 0, trainSize);
        evalData = new Instances(data, trainSize, evalSize);

        // Chỉ định cột output là WIND
        trainData.setClassIndex(trainData.attribute("WIND").index());
        evalData.setClassIndex(evalData.attribute("WIND").index());

        // Khởi tạo mô hình SVR
        model = new SMOreg();

        // Huấn luyện mô hình trên tập huấn luyện
        model.buildClassifier(trainData);

        // Đánh giá mô hình bằng phương pháp cross-validation
        Evaluation eval = new Evaluation(trainData);
        eval.evaluateModel(model, evalData);

        // In kết quả đánh giá
        System.out.println("Evaluation Results:");
        System.out.println(eval.toSummaryString());

        // In độ chính xác
        printRSquared();
    }

    public double[] predict() throws Exception {
        double[] predictions = new double[evalData.numInstances()];
        for (int i = 0; i < evalData.numInstances(); i++) {
            predictions[i] = model.classifyInstance(evalData.instance(i));
        }
        return predictions;
    }

    public double calculateCorrelation() throws Exception {
        double[] predictions = predict(); // Dự đoán giá trị cho dữ liệu đánh giá
        double[] actual = new double[evalData.numInstances()];
        for (int i = 0; i < evalData.numInstances(); i++) {
            actual[i] = evalData.instance(i).classValue(); // Lấy giá trị thực tế của biến mục tiêu
        }
        return calculateCorrelation(predictions, actual);
    }

    private double calculateCorrelation(double[] predicted, double[] actual) {
        // Tính toán giá trị tương quan giữa dự đoán và giá trị thực tế
        double sumXY = 0;
        double sumX2 = 0;
        double sumY2 = 0;
        double sumX = 0;
        double sumY = 0;
        int n = predicted.length;
        for (int i = 0; i < n; i++) {
            sumXY += predicted[i] * actual[i];
            sumX += predicted[i];
            sumY += actual[i];
            sumX2 += Math.pow(predicted[i], 2);
            sumY2 += Math.pow(actual[i], 2);
        }
        double correlation = (n * sumXY - sumX * sumY) / Math.sqrt((n * sumX2 - Math.pow(sumX, 2)) * (n * sumY2 - Math.pow(sumY, 2)));
        return correlation;
    }

    public void printCorrelation() throws Exception {
        double correlation = calculateCorrelation();
        System.out.println("Correlation coefficient: " + correlation);
    }

    public double calculateRSquared() throws Exception {
        double correlation = calculateCorrelation(); // Tính toán giá trị tương quan
        return Math.pow(correlation, 2); // Tính toán R-squared bằng cách bình phương giá trị tương quan
    }

    public void printRSquared() throws Exception {
        double rSquared = calculateRSquared();
        System.out.println("R-squared: " + rSquared);
    }
}
