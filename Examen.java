import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import java.util.Arrays;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StandardScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import static org.apache.spark.sql.functions.col;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.param.ParamMap;


//import org.apache.spark.api.java;
import scala.Tuple2;

public class Examen {
    public static void main(String[] args) {

SparkSession spark = SparkSession
  .builder()
  .appName("DataScientest")
  .getOrCreate();
SparkContext sc_raw = spark.sparkContext();
JavaSparkContext sc = JavaSparkContext.fromSparkContext(sc_raw);

JavaRDD<String> rawRDD = sc.textFile("housing.data");
JavaRDD< String[]> RDD = rawRDD.map(line -> line.substring(1).split("\\s+"));

//Question 1:
JavaPairRDD< String, Integer> pairs = RDD.mapToPair(line -> new Tuple2(line[3], 1));
JavaPairRDD< String, Integer> counts = pairs.reduceByKey((a, b) -> a+b);
System.out.println("Le nombre de villes près du fleuve (1) et le nombre de villes non près du fleuve (0):");
counts.take(10).forEach(line -> System.out.println(line));

//Question 2 :
JavaPairRDD< String, Integer> pairs3 = RDD.mapToPair(line -> new Tuple2(line[5], 1));
JavaPairRDD< String, Integer> counts3 = pairs3.reduceByKey((a, b) -> a+b);
System.out.println("Le nombre de maison par occurence:");
counts3.take(10).forEach(line -> System.out.println(line));

//Question 3 : 
JavaPairRDD< String, Integer> pairs2 = RDD.mapToPair(line -> new Tuple2(line[8], 1));
JavaPairRDD< String, Integer> counts2 = pairs2.reduceByKey((a, b) -> a+b);
System.out.println("Les différentes modalités de la variable RAD sont :");
counts2.take(10).forEach(line -> System.out.println(line));


JavaRDD<Row> rowRDD = RDD.map(line -> RowFactory.create(line));
StructType schema = DataTypes.createStructType(new StructField[] {
  DataTypes.createStructField("CRIM", DataTypes.StringType, true),
  DataTypes.createStructField("ZN", DataTypes.StringType, true),
  DataTypes.createStructField("INDUS", DataTypes.StringType, true),
  DataTypes.createStructField("CHAS", DataTypes.StringType, true),
  DataTypes.createStructField("NOX", DataTypes.StringType, true),
  DataTypes.createStructField("RM", DataTypes.StringType, true),
  DataTypes.createStructField("AGE", DataTypes.StringType, true),
  DataTypes.createStructField("DIS", DataTypes.StringType, true),
  DataTypes.createStructField("RAD", DataTypes.StringType, true),
  DataTypes.createStructField("TAX", DataTypes.StringType, true),
  DataTypes.createStructField("PTRATIO", DataTypes.StringType, true), 
  DataTypes.createStructField("BK", DataTypes.StringType, true), 
  DataTypes.createStructField("LSTAT", DataTypes.StringType, true), 
  DataTypes.createStructField("MEDV", DataTypes.StringType, true)
});

Dataset<Row> df = spark.createDataFrame(rowRDD, schema);
System.out.println("Affichage du DataFrame:");
df.show();

//Conversion des colonnes au format double
Dataset<Row> df2 = df.select(col("CRIM").cast("double"),
                             col("ZN").cast("double"),
                             col("INDUS").cast("double"),
                             col("CHAS").cast("double"),
                             col("NOX").cast("double"),
                             col("RM").cast("double"),
                             col("AGE").cast("double"),
                             col("DIS").cast("double"),
                             col("RAD").cast("double"),
                             col("TAX").cast("double"), 
                             col("PTRATIO").cast("double"),
                             col("BK").cast("double"), 
                             col("LSTAT").cast("double"), 
                             col("MEDV").cast("double"));

//Affichage des statistiques
System.out.println("Le nombre de logement près du fleuve par occurence de nombre de piece");
df2.filter(col("CHAS").equalTo(1)).groupBy("RM").count().show();

System.out.println("Le nombre de logement ayant un nombre de pièce supérieur à 6 par occurence d'index d'accéssibilité aux autoroutes:");
df2.filter(col("RM").geq(6)).groupBy("RAD").count().show();

System.out.println("Les statistiques du dataframe:");
df2.describe().show();

//Modèle Machine learning 
StringIndexer indexer = new StringIndexer()
  .setInputCols(new String[] {"CHAS"})
  .setOutputCols(new String[] {"CHASIndex"});
Dataset<Row> indexed = indexer.fit(df2).transform(df2);
VectorAssembler assembler = new VectorAssembler()
  .setInputCols(new String[] {"CHASIndex","CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "BK", "LSTAT"})
  .setOutputCol("features_pre");
Dataset<Row> data = assembler.transform(indexed).select("MEDV", "features_pre").withColumnRenamed("MEDV", "label");

StandardScaler scaler = new StandardScaler()
      .setInputCol("features_pre")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(false);
LinearRegression lr = new LinearRegression();

Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {scaler, lr});
Dataset<Row>[] data_split = data.randomSplit(new double[] {0.8, 0.2}, 12345);
Dataset<Row> train = data_split[0];
Dataset<Row> test = data_split[1];
ParamMap[] paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam(), new double[] {0.001, 0.01, 0.1, 0.5, 1.0, 2.0})
  .addGrid(lr.elasticNetParam(), new double[] {0.0, 0.25, 0.5, 0.75, 1.0})
  .addGrid(lr.maxIter(),new int[] {1, 5, 10, 20, 50})
  .build();
CrossValidator cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new MulticlassClassificationEvaluator())
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(4);  
CrossValidatorModel cvModel = cv.fit(train);
Dataset<Row> predictions2 = cvModel.transform(test);
predictions2.show();
Dataset<Row> forEvaluationDF = predictions2.select(col("label"), 
                    col("prediction"));
RegressionEvaluator evaluteR2 = new RegressionEvaluator().setMetricName("r2");
RegressionEvaluator evaluteRMSE = new RegressionEvaluator().setMetricName("rmse");
double r2 = evaluteR2.evaluate(forEvaluationDF);
double rmse = evaluteRMSE.evaluate(forEvaluationDF);

System.out.println("Les résultats de la régression linéaire avec validation croisée : ");
System.out.println("R2 =" + r2);
System.out.println("RMSE =" + rmse);
System.out.println("La statistique R2 est de 75%, ce qui nous indique que le modele s'ajuste plus au moins bien aux données.");
System.out.println("La statistique RMSE est de 4.40, ce qui nous indique qu'il y a un petit écart entre le valeurs prédites et les valeurs réelles.");
System.out.println("Concernant la grille de paramètre, j'ai joué sur les trois paramètres suivants :la régularisation, le type de regression avec elasticNetParam(si 1 alors Lasso Regression), et le nombre d'itération. ");

sc.stop();

}
}
