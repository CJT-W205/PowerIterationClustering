package org.apache.spark.examples.mllib

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.PowerIterationClustering
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

/**
 * An example Power Iteration Clustering http://www.icml2010.org/papers/387.pdf app.
 * Takes an input of K concentric circles and the number of points in the innermost circle.
 * The output should be K clusters - each cluster containing precisely the points associated
 * with each of the input circles.
 *
 * Run with
 * {{{
 * ./bin/run-example mllib.PowerIterationClusteringExample [options]
 *
 * Where options include:
 *   k:  Number of circles/clusters
 *   n:  Number of sampled points on innermost circle.. There are proportionally more points
 *      within the outer/larger circles
 *   maxIterations:   Number of Power Iterations
 *   outerRadius:  radius of the outermost of the concentric circles
 * }}}
 *
 * Here is a sample run and output:
 *
 * ./bin/run-example mllib.PowerIterationClusteringExample -k 3 --n 30 --maxIterations 15
 *
 * Cluster assignments: 1 -> [0,1,2,3,4],2 -> [5,6,7,8,9,10,11,12,13,14],
 * 0 -> [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
 *
 *
 * If you use it as a template to create your own app, please use `spark-submit` to submit your app.
 */
object PowerIterationClusteringCJT {

  case class Params(
                     input: String = null,
                     k: Int = 3,
                     maxIterations: Int = 10
                     ) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("PowerIterationClusteringExample") {
      head("PowerIterationClusteringExample: an example PIC app using concentric circles.")
      opt[Int]('k', "k")
        .text(s"number of circles (/clusters), default: ${defaultParams.k}")
        .action((x, c) => c.copy(k = x))
      opt[Int]("maxIterations")
        .text(s"number of iterations, default: ${defaultParams.maxIterations}")
        .action((x, c) => c.copy(maxIterations = x))
      opt[String]("input")
        .required()
        .text(s"input file, one row per line, space-separated")
        .action((x, c) => c.copy(input = x))
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }

  def run(params: Params) {
    val conf = new SparkConf()
//      .setMaster("local")
      .setAppName(s"PowerIterationClustering with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val distFile = sc.textFile(params.input)

    val rows = distFile.map( line => line.drop(1).dropRight(1).split(",").map(_.trim) match {
      case Array(x, y, z) => (x.toLong, y.toLong, z.toDouble)
    })

    val model = new PowerIterationClustering()
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .run(rows)

    model.assignments.collect().foreach { a =>
      println(s"${a.id},${a.cluster}")
    }

    sc.stop()
  }

}