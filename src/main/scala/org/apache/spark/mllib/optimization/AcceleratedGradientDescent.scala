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

package org.apache.spark.mllib.optimization

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._

import breeze.linalg.{DenseVector => BDV, norm}

import org.apache.spark.Logging
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.rdd.RDDFunctions._

/**
 * :: DeveloperApi ::
 * This class optimizes a vector of weights via accelerated (proximal) gradient descent.
 * The implementation is based on TFOCS [[http://cvxr.com/tfocs]], described in Becker, Candes, and
 * Grant 2010.
 * @param gradient Delegate that computes the loss function value and gradient for a vector of
 *                 weights.
 * @param updater Delegate that updates weights in the direction of a gradient.
 */
@DeveloperApi
class AcceleratedGradientDescent (private var gradient: Gradient, private var updater: Updater)
  extends Optimizer {

  private var convergenceTol: Double = 1e-4
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var L0: Double = 1.0
  private var Lexact: Double = Double.PositiveInfinity
  private var beta: Double = 0.5
  private var alpha: Double = 0.9
  private var mayRestart: Boolean = true

  /**
   * Set the optimization convergence tolerance. Default 1e-4.
   * Smaller values will increase accuracy but require additional iterations.
   */
  def setConvergenceTol(tol: Double): this.type = {
    this.convergenceTol = tol
    this
  }

  /**
   * Set the maximum number of iterations. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  def setL0(L0: Double): this.type = {
    this.L0 = L0
    this
  }

  def setLexact(Lexact: Double): this.type = {
    this.Lexact = Lexact
    this
  }

  def setBeta(beta: Double): this.type = {
    this.beta = beta
    this
  }

  def setAlpha(alpha: Double): this.type = {
    this.alpha = alpha
    this
  }

  def setMayRestart(mayRestart: Boolean): this.type = {
    this.mayRestart = mayRestart
    this
  }

  /**
   * Set a Gradient delegate for computing the loss function value and gradient.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  /**
   * Set an Updater delegate for updating weights in the direction of a gradient.
   * If regularization is used, the Updater will implement the regularization term's proximity
   * operator. Thus the type of regularization penalty is configured by providing a corresponding
   * Updater implementation.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
   * Run accelerated gradient descent on the provided training data.
   * @param data training data
   * @param initialWeights initial weights
   * @return solution vector
   */
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = AcceleratedGradientDescent.run(
      data,
      gradient,
      updater,
      convergenceTol,
      numIterations,
      regParam,
      initialWeights,
      L0,
      Lexact,
      beta,
      alpha,
      mayRestart)
    weights
  }
}

/**
 * :: DeveloperApi ::
 * Top-level method to run accelerated (proximal) gradient descent.
 */
@DeveloperApi
object AcceleratedGradientDescent extends Logging {
  /**
   * Run accelerated proximal gradient descent.
   * The implementation is based on TFOCS [[http://cvxr.com/tfocs]], described in Becker, Candes,
   * and Grant 2010. A limited but useful subset of the TFOCS feature set is implemented, including
   * support for composite loss functions, the Auslender and Teboulle acceleration method,
   * backtracking line search, and automatic restart using the gradient test. On each iteration the
   * loss function and gradient are caclculated once or twice from the full training dataset,
   * requiring one Spark map reduce each time.
   *
   * @param data Input data. RDD containing data examples of the form (label, [feature values]).
   * @param gradient Delegate that computes the loss function value and gradient for a vector of
   *                 weights (for one single data example).
   * @param updater Delegate that updates weights in the direction of a gradient.
   * @param convergenceTol Tolerance for convergence of the optimization algorithm. When the norm of
   *                       the change in weight vectors between successive iterations falls below
   *                       this relative tolerance, optimization is complete.
   * @param numIterations Maximum number of iterations to run the algorithm.
   * @param regParam The regularization parameter.
   * @param initialWeights The initial weight values.
   * @param TODO
   *
   * @return A tuple containing two elements. The first element is a Vector containing the optimized
   *         weight for each feature, and the second element is an array containing the approximate
   *         loss computed on each iteration.
   */
  def run(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: Updater,
      convergenceTol: Double,
      numIterations: Int,
      regParam: Double,
      initialWeights: Vector,
      L0: Double,
      Lexact: Double,
      beta: Double,
      alpha: Double,
      mayRestart: Boolean): (Vector, Array[Double]) = {

    /** Returns the loss function and gradient for the provided weights 'x'. */
    def applySmooth(x: BDV[Double]): (Double, BDV[Double]) = {
      val bcX = data.context.broadcast(Vectors.fromBreeze(x))

      // Sum the loss function and gradient computed for each training example.
      val (loss, grad, count) = data.treeAggregate((0.0, BDV.zeros[Double](x.size), 0L))(
        seqOp = (c, v) => (c, v) match { case ((loss, grad, count), (label, features)) =>
          val l = gradient.compute(features, label, bcX.value, Vectors.fromBreeze(grad))
          (loss + l, grad, count + 1)
        },
        combOp = (c1, c2) => (c1, c2) match {
          case ((loss1, grad1, count1), (loss2, grad2, count2)) =>
            (loss1 + loss2, grad1 += grad2, count1 + count2)
        })

      // Divide the summed loss and gradient by the number of training examples.
      (loss / count, grad / (count: Double))
    }

    /**
     * Returns the regularization loss and updates weights according to the gradient and the
     * proximity operator.
     */
    def applyProjector(x: BDV[Double], g: BDV[Double], step: Double): (Double, BDV[Double]) = {
      val (weights, regularization) = updater.compute(Vectors.fromBreeze(x),
                                                      Vectors.fromBreeze(g),
                                                      step,
                                                      iter = 1, // Passing 1 avoids step size
                                                                // rescaling within the updater.
                                                      regParam)
      (regularization, BDV[Double](weights.toArray))
    }

    var x = BDV[Double](initialWeights.toArray)
    var z = x
    var theta = Double.PositiveInfinity
    val lossHistory = new ArrayBuffer[Double](numIterations)

    var f_y = 0.0
    var g_y = BDV[Double](Vectors.zeros(initialWeights.size).toArray) 

    var L = L0

    var backtrack_simple = true
    val backtrack_tol = 1e-10

    breakable { for (nIter <- 1 to numIterations) {

      // Auslender and Teboulle's accelerated method.

      val (x_old, z_old) = (x, z)
      val L_old = L
      L = L * alpha
      val theta_old = theta

      breakable { while (true) {

        theta = 2.0 / (1.0 + math.sqrt(1.0 + 4.0 * (L / L_old) / (theta_old * theta_old)))
        val y = x_old * (1.0 - theta) + z_old * theta
        val (f_y_, g_y_) = applySmooth(y)
        f_y = f_y_
        g_y = g_y_
        val step = 1.0 / ( theta * L )
        z = applyProjector(z_old, g_y, step)._2
        x = x_old * (1.0 - theta) + z * theta

        if (beta >= 1.0) {
          break
        }

        // Backtracking.

        val xy = x - y
        val xy_sq = math.pow(norm(xy), 2)
        if (xy_sq == 0) {
          break
        }

        val (f_x, g_x) = applySmooth(x)
        var localL = 0.0

        if (backtrack_simple) {
          val q_x = f_y + xy.dot(g_y) + 0.5 * L * xy_sq
          localL = L + 2.0 * math.max(f_x - q_x, 0.0) / xy_sq
          backtrack_simple = (math.abs(f_y - f_x) >= backtrack_tol * math.max(math.abs(f_x), math.abs(f_y)))
        }
        else {
          localL = 2.0 * (x - y).dot(g_x - g_y) / xy_sq
        }

        if (localL <= L || L >= Lexact) {
          break
        }

        if (!localL.isInfinity) {
          L = math.min(Lexact, localL)
        }
        else if (localL.isInfinity) {
          localL = L
        }

        L = math.min(Lexact, math.max(localL, L / beta))
      } }

      // Track loss history using the loss function at y, since f_y is already available and
      // computing f_x would require another (distributed) call to applySmooth. Start by finding
      // c_y, the regularization component of the loss function at y.
      // val (c_y, _) = applyProjector(y, g_y, 0.0)
      // lossHistory.append(f_y + c_y)
      // TODO Use f_x when available.

      {
        // Temporarily tracking loss history at x instead of y, to validate against TFOCS.
        val (f_x, g_x) = applySmooth(x)
        val (c_x, _) = applyProjector(x, g_x, 0.0)
        lossHistory.append(f_x + c_x)
      }

      if (f_y.isNaN || f_y.isInfinity) {
        logWarning("Unable to compute loss function.")
        break
      }

      // Check convergence.
      val norm_x = norm(x)
      val norm_dx = norm(x - x_old)
      if (norm_dx == 0.0) {
        if (nIter > 1) {
          break
        }
      }
      if (norm_dx < convergenceTol * math.max(norm_x, 1)) {
        break
      }

      // Restart acceleration if indicated by the gradient test from O'Donoghue and Candes 2013.
      if (mayRestart && g_y.dot(x - x_old) > 0.0) {
        z = x
        theta = Double.PositiveInfinity
        backtrack_simple = true
      }
    } }

    logInfo("AcceleratedGradientDescent.run finished. Last 10 losses %s".format(
      lossHistory.takeRight(10).mkString(", ")))

    (Vectors.fromBreeze(x), lossHistory.toArray)
  }
}
