import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.Random


sealed trait ITree
case class ITreeBranch(left: ITree, right: ITree, split_column: Int, split_value: Double) extends ITree
case class ITreeLeaf(size: Long) extends ITree

case class IsolationForest(num_samples: Long, trees: Array[ITree]) {
    def predict(x:Array[Double]): Double = {
        val predictions = trees.map(s => pathLength(x, s, 0)).toList
        math.pow(2, -(predictions.sum/predictions.size)/cost(num_samples)) //Anomaly Score
    }

    def cost(num_items:Long): Int =
        //二叉搜索树的平均路径长度。0.5772156649:欧拉常数
        (2*(math.log(num_items-1) + 0.5772156649)-(2*(num_items-1)/num_items)).toInt

    @scala.annotation.tailrec
    final def pathLength(x:Array[Double], tree:ITree, path_length:Int): Double ={
        tree match{
            case ITreeLeaf(size) =>
                if (size > 1)
                    path_length + cost(size)
                else 
                    path_length + 1

            case ITreeBranch(left, right, split_column, split_value) =>
                val sample_value = x(split_column)

                if (sample_value < split_value)
                    pathLength(x, left, path_length + 1)
                else
                    pathLength(x, right, path_length + 1)
        }
    }
}


object IsolationForest {

    def getRandomSubsample(data: RDD[Array[Double]], sampleRatio: Double, seed: Long = Random.nextLong): RDD[Array[Double]] = {
        data.sample(false, sampleRatio, seed=seed)
    }

    def buildForest(data: RDD[Array[Double]], numTrees: Int = 2, subSampleSize: Int = 256, seed: Long = Random.nextLong) : IsolationForest = {
        val numSamples = data.count()
        val numColumns = data.take(1)(0).size
        val maxHeight = math.ceil(math.log(subSampleSize)).toInt
        val trees = Array.fill[ITree](numTrees)(ITreeLeaf(1))

        val trainedTrees = trees.map(s=>growTree(getRandomSubsample(data, subSampleSize/numSamples.toDouble, seed), maxHeight, numColumns))

        IsolationForest(numSamples, trainedTrees)
    }

    def growTree(data: RDD[Array[Double]], maxHeight:Int, numColumns:Int, currentHeight:Int = 0): ITree = {
        val numSamples = data.count()
        if(currentHeight>=maxHeight || numSamples <= 1){
            return new ITreeLeaf(numSamples)
        }

        val split_column = Random.nextInt(numColumns)
        val column = data.map(s => s(split_column))

        val col_min = column.min()
        val col_max = column.max()
        val split_value = col_min + Random.nextDouble()*(col_max-col_min)

        val X_left = data.filter(s => s(split_column) < split_value).cache()
        val X_right = data.filter(s => s(split_column) >= split_value).cache()


        new ITreeBranch(growTree(X_left, maxHeight, numColumns, currentHeight + 1),
            growTree(X_right, maxHeight, numColumns, currentHeight + 1),
            split_column,
            split_value)
    }
}


object Runner{
    def main(args:Array[String]): Unit ={
        Random.setSeed(1337)

        val conf = new SparkConf()
            .setAppName("IsolationTree")
            .setMaster("local")

        val sc = new SparkContext(conf)
        sc.hadoopConfiguration.set("mapred.output.compress", "false")

        val lines = sc.textFile("file:///tmp/spark_data/spark_if_train.csv")

        val data =
            lines
                .map(line => line.split(","))
                .map(s => s.slice(1,s.length)) //lines in rows

        val header = data.first()
        val rows = data.filter(line => line(0) != header(0)).map(s => s.map(_.toDouble))

        println("Loaded CSV File...")
        println(header.mkString("\n"))
        println(rows.take(5).deep.mkString("\n"))

        val forest = IsolationForest.buildForest(rows, numTrees=10)

        val result_rdd = rows.map(row => row ++ Array(forest.predict(row)))

        result_rdd.map(lines => lines.mkString(",")).repartition(1).saveAsTextFile("file:///tmp/predict_label")

        val local_rows = rows.take(10)
        for(row <- local_rows){
            println("ForestScore", forest.predict(row))
        }
        println("Finished Isolation")

    }

}

