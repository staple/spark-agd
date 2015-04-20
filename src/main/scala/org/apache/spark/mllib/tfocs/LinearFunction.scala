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

import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

/**
 * Trait for linear functions.
 */
trait LinearFunction[X, Y] {
  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Y

  /**
   * The transpose of this function.
   */
  def t: LinearFunction[Y, X]
}

class ProductRDDVector(private val matrix: RDD[Vector]) extends LinearFunction[Vector, RDD[Double]] {

  matrix.cache()

  override def apply(x: Vector): RDD[Double] = {
    val bcX = matrix.context.broadcast(x)
    matrix.map(row => BLAS.dot(row, bcX.value))
  }

  override def t: LinearFunction[RDD[Double], Vector] = new TransposeProductRDDVector(matrix)
}

class TransposeProductRDDVector(private val matrix: RDD[Vector]) extends LinearFunction[RDD[Double], Vector] {

  matrix.cache()

  private lazy val n = matrix.first.size

  override def apply(x: RDD[Double]): Vector = {
    matrix.zip(x).aggregate(Vectors.zeros(n))(
      seqOp = (sum, row) => {
        BLAS.axpy(row._2, row._1, sum)
        sum
      },
      combOp = (s1, s2) => {
        BLAS.axpy(1.0, s2, s1)
        s1
      }
    )
  }

  override def t: LinearFunction[Vector, RDD[Double]] = new ProductRDDVector(matrix)
}
