package tfocs

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
}

/**
 * Trait for smooth functions.
 */
trait SmoothFunction[X, Y] {
  /**
   * Evaluates this function at x and returns the function value and its gradient based on the mode
   * specified.
   */
  def apply(x: X, mode: Mode): Value[Double, Y]

  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Double = apply(x, Mode(f=true, g=false)).f.get
}

/**
 * Trait for linear functions.
 */
trait LinearFunction[X, Y] extends SmoothFunction[X, Y] {
  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Y

  /**
   * The transpose of this function.
   */
  def t: LinearFunction[Y, X]
}

/**
 * Trait for prox-capable functions.
 */
trait ProxCapableFunction[X, Y] {
  /**
   * Evaluates this function at x.
   */
  def apply(x: X): Double

  /**
   * Evaluates this function at x with smoothing parameter t.
   */
  def apply(x: X, t: Double, mode: Mode): Value[X, Y]

  // another option is
  // def prox(t: Double): SmoothFunction[X, Y]
}

/**
 * Evaluation mode.
 * @param f whether to compute function value
 * @param g whether to compute gradient
 */
case class Mode(f: Boolean, g: Boolean)

/**
 * Evaluation result.
 * @param f optional function value
 * @param g optional gradient value
 */
case class Value[X, Y](f: Option[X], g: Option[Y])

object TFOCS {

  def minimize[X, Y](
    f: SmoothFunction[Y, Double],
    A: LinearFunction[X, Y],
    b: X,
    h: ProxCapableFunction[X, Double],
    x0: X)(
    implicit vx: VectorSpace[X],
    vy: VectorSpace[Y]): X = ???
}

