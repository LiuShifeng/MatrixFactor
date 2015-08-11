/**
 * Created by LiuShifeng on 2015/8/11.
 */
package wti.st.MatrixFactor.NMF

import org.apache.spark.SparkConf
import org.apache.spark.{SparkContext, SparkConf}
import org.apacche.spark.RDD

class NMF(val matrix:RDD[String],val alpha:Double,val beta:Double,val MaxInteration){

}

