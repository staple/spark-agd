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

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._

import org.apache.spark.Logging

object TFOCS extends Logging {

  def minimize[X, Y](
    f: SmoothFunction[Y],
    A: LinearFunction[X, Y],
    h: ProxCapableFunction[X],
    x0: X)(
    implicit vx: VectorSpace[X],
    vy: VectorSpace[Y]): (X, Array[Double]) = {

      val convergenceTol = 1e-13
      val numIterations = 500
      val L0 = 10.0
      val Lexact = Double.PositiveInfinity
      val beta = 0.5
      val alpha = 0.9
      val mayRestart = true

      var x = x0
      var z = x
      vx.cache(x)
      var a_x = A(x)
      var a_z = a_x
      vy.cache(a_x)
      var theta = Double.PositiveInfinity
      val lossHistory = new ArrayBuffer[Double](numIterations)

      var f_y = 0.0
      var g_y = x // Dummy initialization value.

      var L = L0

      var backtrack_simple = true
      val backtrack_tol = 1e-10

      var cntr_Ay = 0
      var cntr_Ax = 0
      val cntr_reset = 10

      breakable { for (nIter <- 1 to numIterations) {

        // Auslender and Teboulle's accelerated method.

        val (x_old, z_old) = (x, z)
        val (a_x_old, a_z_old) = (a_x, a_z)
        val L_old = L
        L = L * alpha
        val theta_old = theta

        breakable { while (true) {

          theta = 2.0 / (1.0 + math.sqrt(1.0 + 4.0 * (L / L_old) / (theta_old * theta_old)))

          val y = vx.combine(1.0 - theta, x_old, theta, z_old)
          val a_y = if (cntr_Ay >= cntr_reset) {
            cntr_Ay = 0
            A(y)
          } else {
            cntr_Ay = cntr_Ay + 1
            vy.combine(1.0 - theta, a_x_old, theta, a_z_old)
          }
          if (!backtrack_simple) vy.cache(a_y)

          val Value(Some(f_y_), Some(g_Ay)) = f(a_y, Mode(true, true)) // Spark job
          // TODO f_y is not always necessary.
          f_y = f_y_
          if (!backtrack_simple) vy.cache(g_Ay)
          g_y = A.t(g_Ay) // Spark job
          vx.cache(g_y)
          val step = 1.0 / (theta * L)
          z = h(vx.combine(1.0, z_old, -step, g_y), step, Mode(false, true)).g.get
          vx.cache(z)
          a_z = A(z)
          vy.cache(a_z)

          x = vx.combine(1.0 - theta, x_old, theta, z)
          vx.cache(x)
          a_x = if (cntr_Ax >= cntr_reset) {
            cntr_Ax = 0
            A(x)
          } else {
            cntr_Ax = cntr_Ax + 1
            vy.combine(1.0 - theta, a_x_old, theta, a_z)
          }
          vy.cache(a_x)

          if (beta >= 1.0) {
            break
          }

          // Backtracking.

          val xy = vx.combine(1.0, x, -1.0, y)
          vx.cache(xy)
          val xy_sq = vx.dot(xy, xy)

          if (xy_sq == 0.0) {
            break
          }

          var localL = 0.0

          if (backtrack_simple) {
            val Value(Some(f_x), _) = f(a_x, Mode(true, false)) // Spark job
            val q_x = f_y + vx.dot(xy, g_y) + 0.5 * L * xy_sq
            localL = L + 2.0 * math.max(f_x - q_x, 0.0) / xy_sq
            backtrack_simple = (math.abs(f_y - f_x) >= backtrack_tol * math.max(math.abs(f_x), math.abs(f_y)))
          }
          else {
            val Value(_, Some(g_Ax)) = f(a_x, Mode(false, true))
            localL = 2.0 * vy.dot(vy.combine(1.0, a_x, -1.0, a_y), vy.combine(1.0, g_Ax, -1.0, g_Ay)) / xy_sq // Spark job
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
        // c_y, the regularization component of the loss function at y. But use f_x when available.
        // TODO

        {
          // TEMP Temporarily always tracking loss history at x instead of y, to validate against TFOCS.
          val Value(Some(f_x), _) = f(a_x, Mode(true, false))
          val Value(Some(c_x), _) = h(x, 0.0, Mode(true, false))
          lossHistory.append(f_x + c_x)
        }

        if (f_y.isNaN || f_y.isInfinity) {
          logWarning("Unable to compute loss function.")
          break
        }

        // Check convergence.
        val norm_x = math.sqrt(vx.dot(x, x))
        val dx = vx.combine(1.0, x, -1.0, x_old)
        vx.cache(dx)
        val norm_dx = math.sqrt(vx.dot(dx, dx))
        if (norm_dx == 0.0) {
          if (nIter > 1) {
            break
          }
        }
        if (norm_dx < convergenceTol * math.max(norm_x, 1)) {
          break
        }

        // Restart acceleration if indicated by the gradient test from O'Donoghue and Candes 2013.
        // NOTE TFOCS uses <g_Ay, a_x - a_x_old> here, but we do it this way to avoid a spark job.
        if (mayRestart && vx.dot(g_y, vx.combine(1.0, x, -1.0, x_old)) > 0.0) {
          z = x
          a_z = a_x
          theta = Double.PositiveInfinity
          backtrack_simple = true
        }
      } }

      logInfo("TFOCS.minimize finished. Last 10 losses %s".format(
        lossHistory.takeRight(10).mkString(", ")))

      (x, lossHistory.toArray)
    }
}
