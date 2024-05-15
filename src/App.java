import Preprocessing.Preprocessing;
import Weka.LinearRegressionModel;
import Weka.SVRModel;
public class App {
    public static void main(String[] args) throws Exception {
        String filePath = "C:/My Downloads/Preprocessing/Data/wind_dataset.csv";
        Preprocessing p = new Preprocessing(filePath);
        p.start();
            // Đường dẫn đến tệp CSV chứa dữ liệu
        String csvFilePath = "C:/My Downloads/Preprocessing/Data/CleanedData.arff";
        LinearRegressionModel model1 = new LinearRegressionModel(csvFilePath);
        SVRModel svmModel = new SVRModel(csvFilePath);
        
    

            // double[] predictions = model1.predict();
            // In kết quả dự đoán
            // System.out.println("Predictions:");
            // for (double prediction : predictions) {
            //     System.out.println(prediction);
            // }
    }
}
