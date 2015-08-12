/**
 * Created by LiuShifeng on 2015/8/11.
 */
package wti.st.MatrixFactor.NMF
import scala.math
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import scala.package$

class NMF(matrix:Array[Array[Double]], latentD:Int = 10, alpha:Double=0.002, beta:Double=0.02, MaxIteration:Int=5000) extends Serializable{
  val N = matrix.length()
  val M = matrix(0).length()
  val K = latentD
  val P = Array.ofDim(N,K)
  val Q = Array.ofDim(K,M)

  for(i<-0 until N){
    for(j<-0 until K){
      P(i)(j) = math.random()
    }
  }
  for(i<-0 until K){
    for(j<-0 until M){
      Q(i)(j) = math.random()
    }
  }

  /**
   *
   * @param alpha
   * @param beta
   * @return
   */
  def updatePQ(alpha:Double,beta:Double): Unit ={
    for(i<-0 until N){
      for(j<-0 until M){
        var PQ = 0.0
        for(k<-0 until K){
          PQ += P(i)(k)*Q(k)(j)
        }
        val eij = matrix(i)(j)-PQ
        for(k<-0 until K){
          val oldPQ = matrix(i)(k)
          P(i)(k) += alpha*(2*eij*Q(k)(j)-beta*P(i)(k))
          Q(k)(j) += alpha*(2*eij*oldPQ-beta*Q(k)(j))
        }
      }
    }
  }

  /**
   *
   * @param beta
   * @return
   */
  def getSSE(beta:Double): Double ={
    var sse = 0.0

    for(i<-0 until N){
      for(j<-0 until M){
        var PQ = 0.0
        for(k<-0 until K){
          PQ += P(i)(k)*Q(k)(j)
        }
        sse += math.pow(matrix(i)(j)-PQ,2)
      }
    }

    for(i<-0 until M){
      for(j<-0 until K){
        sse += beta/2*math.pow(P(i)(j),2)
      }
    }

    for(i<-0 until K){
      for(j<-0 until M){
        sse += beta/2*math.pow(Q(i)(j),2)
      }
    }

    sse
  }

  def doNMF(iteration:Int,alpha:Double,beta:Double): Unit ={
    for(i<-0 until iteration){
      updatePQ(alpha,beta)
      val sse = getSSE(beta)
      if(i%100 == 0){
        println("step "+i+" sse "+sse)
      }
    }
  }

  /**
   *
   * @return
   */
  def printRawMatrix(): Unit ={
    println("RawMatrix")
    for(i<-0 until N){
      println(matrix(i).mkString(" "))
    }
    println()
  }

  /**
   *
   * @return
   */
  def printFacMatrix(): Unit ={
    println("FactorMatrix")
    for(i<-0 until N){
      for(j<-0 until M){
        var PQ = 0.0
        for(k<-0 until K){
          PQ += P(i)(k)*Q(k)(j)
        }
        print(PQ + "  ")
      }
      print("\r\n")
      println()
    }
  }

  /**
   *
   * @return
   */
  def printP(): Unit ={
    println("MatrixP")
    for(i<-0 until N)
      println(P(i).mkString(" "))
    println
  }

  /**
   *
   * @return
   */
  def printQ(): Unit = {
    println("MatrixQ")
    for(i<-0 until K)
      println(Q(i).mkString(" "))
    println
  }

}

