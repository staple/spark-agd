/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tfocs

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.rdd.RDD

/**
 * Trait for a vector space.
 */
trait VectorSpace[V] {

  /** Linear combination of two vectors. */
  def combine(alpha: Double, a: V, beta: Double, b: V): V

  /** Inner product of two vectors. */
  def dot(a: V, b: V): Double

  /** Cache a vector. */
  def cache(a: V): Unit = {}

  // TEMP Testing/Validation
  def toStr(a: V): String = ""
}

object VectorSpace {

  implicit object SimpleVectorSpace extends VectorSpace[Vector] {

    override def combine(alpha: Double, a: Vector, beta: Double, b: Vector): Vector = {
      val ret = a.copy
      if (alpha != 1.0) BLAS.scal(alpha, ret)
      BLAS.axpy(beta, b, ret)
      ret
    }

    override def dot(a: Vector, b: Vector): Double = BLAS.dot(a, b)

    override def toStr(a: Vector): String = s"${a.toArray.slice(0,10).mkString(" ")}"
  }

  implicit object RDDDoubleVectorSpace extends VectorSpace[RDD[Double]] {

    override def combine(alpha: Double, a: RDD[Double], beta: Double, b: RDD[Double]): RDD[Double] =
      a.zip(b).map(x => alpha * x._1 + beta * x._2)
      // Cases where alpha or beta == 0 may be special cased if used.

    override def dot(a: RDD[Double], b: RDD[Double]): Double =
      a.zip(b).aggregate(0.0)((sum, x) => sum + x._1 * x._2, _ + _)

    override def cache(a: RDD[Double]): Unit = a.cache()

    override def toStr(a: RDD[Double]): String = s"${a.take(10).mkString(" ")}"

  }
}
