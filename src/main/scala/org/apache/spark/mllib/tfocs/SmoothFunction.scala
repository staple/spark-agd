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

import org.apache.spark.rdd.RDD

/**
 * Trait for smooth functions.
 */
trait SmoothFunction[X] {
  /**
   * Evaluates this function at x and returns the function value and its gradient based on the mode
   * specified.
   */
  def apply(x: X, mode: Mode): Value[Double, X]

  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Double = apply(x, Mode(f=true, g=false)).f.get
}

class SquaredErrorRDDDouble(x0: RDD[Double]) extends SmoothFunction[RDD[Double]] {
  
  x0.cache()

  override def apply(x: RDD[Double], mode: Mode): Value[Double, RDD[Double]] = {
    val g = x.zip(x0).map(x => x._1 - x._2)
    if (mode.f && mode.g) g.cache()
    val f = if (mode.f) Some(g.aggregate(0.0)((sum, x) => sum + x * x, _ + _) / 2) else None
    Value(f, Some(g))
  }

}
