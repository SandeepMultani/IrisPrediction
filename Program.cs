using Microsoft.Data.DataView;
using Microsoft.ML;
using System;
using System.IO;

namespace IrisPrediction
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data.txt");
        static void Main(string[] args)
        {
            Console.WriteLine("Processing...");

            MLContext mlContext = new MLContext();

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<IrisData>(path: _dataPath, hasHeader: false, separatorChar: ',');


            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .AppendCacheCheckpoint(mlContext)
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainingDataView);

            Console.WriteLine("Enter Sepal Length:");
            float sepalLength = float.Parse(Console.ReadLine());
            Console.WriteLine("Enter Sepal Width:");
            float sepalWidth = float.Parse(Console.ReadLine());
            Console.WriteLine("Enter Petal Length:");
            float petalLength = float.Parse(Console.ReadLine());
            Console.WriteLine("Enter Petal Width:");
            float petalWidth = float.Parse(Console.ReadLine());

            var prediction = model.CreatePredictionEngine<IrisData, IrisPrediction>(mlContext).Predict(
                new IrisData()
                {
                    SepalLength = sepalLength,
                    SepalWidth = sepalWidth,
                    PetalLength = petalLength,
                    PetalWidth = petalWidth,
                });

            Console.WriteLine($"Predicted flower type is: {prediction.PredictedLabels}");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}
