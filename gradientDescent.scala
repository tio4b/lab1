import scala.annotation.tailrec
import scala.collection.mutable.ArrayBuffer
import java.io.PrintWriter
import scala.util.Random.nextGaussian

object gradientDescent {

  var functionEvalCount: Long = 0
  var gradientEvalCount: Long = 0

  def fCounting(p: Point, f: Point => Double): Double = {
    functionEvalCount += 1
    f(p)
  }

  val p2: Double => Double = math.pow(_, 2)
  val eps = 1e-8
  val noise_power = 0.000001
  val max_iter = 1e5
  val log_file = "gradient_log.csv"
  val record_flag = true

  object testFunction {
    def GD1(p: Point): Double = p2(p.x) + p2(p.y)

    def GD2(p: Point): Double = 3 * p2(p.x) + p2(p.y) - 2 * p.x * p.y

    def Boota(p: Point): Double = p2(p.x + 2 * p.y - 7) + p2(2 * p.x + p.y - 5)

    def Matyas(p: Point): Double = 0.26 * (p2(p.x) + p2(p.y)) - 0.48 * p.x * p.y

    def Himmelblau_plus_eps(p: Point): Double =
      Himmelblau(p) + 1e-10

    def Himmelblau(p: Point): Double =
      p2(p2(p.x) + p.y - 11) + p2(p.x + p2(p.y) - 7)

    def NoisyHimmelblau(p: Point): Double =
      Himmelblau(p) + nextGaussian() * noise_power
      
    def NoisyGD1(p: Point): Double =
      GD1(p) + nextGaussian() * noise_power
  }

  @tailrec
  def dichotomy(f: Double => Double, a: Double, b: Double): Double = {
    val middle: Double = (a + b) / 2.0
    val middleLeft: Double = (a + middle) / 2.0
    val middleRight: Double = (middle + b) / 2.0

    if (b - a < eps) middle
    else if (f(middleLeft) < f(middle)) dichotomy(f, a, middle)
    else if (f(middleRight) < f(middle)) dichotomy(f, middle, b)
    else dichotomy(f, middleLeft, middleRight)
  }

  def fDichotomy(x: Double): Double = math.pow(math.E, p2(x - 10))

  def printInfo(res: (Point, Int)): Unit = {
    println("Result coordinates: " + res._1.printCords)
    println("Iterations: " + res._2)
  }

  case class Point(coords: Array[Double]) {
    def -(other: Point): Point = Point(coords.zip(other.coords).map { case (a, b) => a - b })

    def +(other: Point): Point = Point(coords.zip(other.coords).map { case (a, b) => a + b })

    def *(scalar: Double): Point = Point(coords.map(_ * scalar))

    def N: Double = math.sqrt(coords.map(math.pow(_, 2)).sum)

    def x: Double = coords(0)

    def y: Double = coords(1)

    def z: Double = if (coords.length > 2) coords(2) else 0.0

    def printCords: String =
      coords.map(c => "%.20f".format(c)).mkString(" , ")

    def equals(that: Point): Boolean = {
      val eps = 1e-4
      this.coords.zip(that.coords).forall { case (a, b) => math.abs(a - b) <= eps }
    }
  }

  sealed trait Scheduling {
    def func: (Int, Point) => Double

    override def toString(): String
  }

  object hConst extends Scheduling {
    override def func = (_, _) => 1.0 / 100.0

    override def toString(): String = "Const"
  }

  object hSequence extends Scheduling {
    override def func = (k, _) => 1.0 / (k + 1)

    override def toString(): String = "Sequence"
  }

  object hFunc extends Scheduling {
    override def func = (k, _) => 1.0 / (k + 1)

    override def toString(): String = "Function"
  }

  case class armijoRule(f: Point => Double) extends Scheduling {
    override def toString(): String = "Armijo Rule"

    override def func = { (k: Int, x: Point) =>
      val c: Double = 0.5
      val gradF: Point = QuadFunc(f).findGradient(x)
      val N2: Double = math.pow(gradF.N, 2)

      @tailrec
      def rec(h: Double, iter: Int): Double = {
        val armijoBool =
          fCounting(x - gradF * h, f) <= fCounting(x, f) - c * h * N2
        if (iter > max_iter) h
        else if (armijoBool) h
        else rec(h * c, iter + 1)
      }

      rec(1, 0)
    }
  }

  case class wolfeRule(f: Point => Double) extends Scheduling {
    override def toString(): String = "Wolfe Rule"

    override def func = { (k: Int, x: Point) =>
      val c1: Double = 0.001
      val c2: Double = 0.9
      val function: QuadFunc = QuadFunc(f)
      val gradF: Point = function.findGradient(x)
      val N2: Double = math.pow(gradF.N, 2)

      @tailrec
      def rec(h: Double, iter: Int): Double = {
        val armijoBool =
          fCounting(x - gradF * h, f) <= fCounting(x, f) - c1 * h * N2
        val wolfeBool =
          function.findGradient(x - gradF * h).N <= c2 * gradF.N

        if (iter > max_iter) h
        else if (!armijoBool) rec(h * 0.5, iter + 1)
        else if (!wolfeBool) rec(h / c2, iter + 1)
        else h
      }

      rec(1, 0)
    }
  }

  case class QuadFunc(f: Point => Double) {

    val eps = 1e-8

    def findGradient(p: Point): Point = {
      gradientEvalCount += 1
      Point(
        p.coords.zipWithIndex.map { case (coord, index) =>
          val leftArg = Point(p.coords.updated(index, coord + eps))
          val rightArg = Point(p.coords.updated(index, coord - eps))
          val diff = fCounting(leftArg, f) - fCounting(rightArg, f)
          diff / (2 * eps)
        }
      )
    }

    private def step(x_k: Point, k: Int, h: Scheduling): Point =
      x_k - findGradient(x_k) * h.func(k, x_k)

    def gradientDescent(x_0: Point, h: Scheduling): (Point, Int) = {

      functionEvalCount = 0
      gradientEvalCount = 0

      val logBuffer = ArrayBuffer[String]()
      if (record_flag) logBuffer.append(x_0.printCords)

      @tailrec
      def recursion(x_k: Point, k: Int): (Point, Int) = {
        val gradNorm = findGradient(x_k).N
        if (gradNorm < eps * 1e-5 || k > max_iter) (x_k, k)
        else {
          val next = step(x_k, k, h)
          if (record_flag) logBuffer.append(next.printCords)
          recursion(next, k + 1)
        }
      }

      val (finalPoint, iterCount) = recursion(x_0, 0)

      if (record_flag) {
        val writer = new PrintWriter(log_file)
        logBuffer.foreach(writer.println)
        writer.close()
      }
      (finalPoint, iterCount)
    }
  }

  def main(args: Array[String]): Unit = {
    val func = testFunction.NoisyGD1
    val quadFunc = QuadFunc(func)
    printResult(quadFunc, hConst)
    printResult(quadFunc, hSequence)
    printResult(quadFunc, armijoRule(func))
    printResult(quadFunc, wolfeRule(func))
  }


  private def printResult(quadFunc: QuadFunc, scheduling: Scheduling): Unit = {
    val startPoint = Point(Array(-1, 2))
    val (minCords, iterations) = quadFunc.gradientDescent(startPoint, scheduling)

    println(s"Scheduling: ${scheduling.toString}")
    println(f"Found point: (${minCords.x}%.20f, ${minCords.y}%.20f)")
    println(s"Iterations: $iterations")
    println(s"Function evaluations: ${functionEvalCount}")
    println(s"Gradient evaluations:  ${gradientEvalCount}")
    println("------------------------------------------------------\n")
  }
}
