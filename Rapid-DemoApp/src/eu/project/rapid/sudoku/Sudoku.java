/*******************************************************************************
 * Copyright (C) 2015, 2016 RAPID EU Project
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *******************************************************************************/
package eu.project.rapid.sudoku;

import java.lang.reflect.Method;

import eu.project.rapid.ac.DFE;
import eu.project.rapid.ac.Remote;
import eu.project.rapid.ac.Remoteable;

/**
 * The <code>Sudoku</code> class povides a static <code>main</code> method allowing it to be called
 * from the command line to print the solution to a specified Sudoku problem.
 * 
 * <p>
 * The following is an example of a Sudoku problem:
 * 
 * <pre>
 * -----------------------
 * |   8   | 4   2 |   6   |
 * |   3 4 |       | 9 1   |
 * | 9 6   |       |   8 4 |
 *  -----------------------
 * |       | 2 1 6 |       |
 * |       |       |       |
 * |       | 3 5 7 |       |
 *  -----------------------
 * | 8 4   |       |   7 5 |
 * |   2 6 |       | 1 3   |
 * |   9   | 7   1 |   4   |
 *  -----------------------
 * </pre>
 * 
 * The goal is to fill in the missing numbers so that every row, column and box contains each of the
 * numbers <code>1-9</code>. Here is the solution to the problem above:
 * 
 * <pre>
 *  -----------------------
 * | 1 8 7 | 4 9 2 | 5 6 3 |
 * | 5 3 4 | 6 7 8 | 9 1 2 |
 * | 9 6 2 | 1 3 5 | 7 8 4 |
 *  -----------------------
 * | 4 5 8 | 2 1 6 | 3 9 7 |
 * | 2 7 3 | 8 4 9 | 6 5 1 |
 * | 6 1 9 | 3 5 7 | 4 2 8 |
 *  -----------------------
 * | 8 4 1 | 9 6 3 | 2 7 5 |
 * | 7 2 6 | 5 8 4 | 1 3 9 |
 * | 3 9 5 | 7 2 1 | 8 4 6 |
 *  -----------------------
 * </pre>
 * 
 * Note that the first row <code>187492563</code> contains each number exactly once, as does the
 * first column <code>159426873</code>, the upper-left box <code>187534962</code>, and every other
 * row, column and box.
 * 
 * <p>
 * The {@link #main(String[])} method encodes a problem as an array of strings, with one string
 * encoding each constraint in the problem in row-column-value format. Here is the problem again
 * with the indices indicated:
 * 
 * <pre>
 *     0 1 2   3 4 5   6 7 8
 *    -----------------------
 * 0 |   8   | 4   2 |   6   |
 * 1 |   3 4 |       | 9 1   |
 * 2 | 9 6   |       |   8 4 |
 *    -----------------------
 * 3 |       | 2 1 6 |       |
 * 4 |       |       |       |
 * 5 |       | 3 5 7 |       |
 *   -----------------------
 * 6 | 8 4   |       |   7 5 |
 * 7 |   2 6 |       | 1 3   |
 * 8 |   9   | 7   1 |   4   |
 *    -----------------------
 * </pre>
 * 
 * The <code>8</code> in the upper left box of the puzzle is encoded as <code>018</code> (
 * <code>0</code> for the row, <code>1</code> for the column, and <code>8</code> for the value). The
 * <code>4</code> in the lower right box is encoded as <code>874</code>.
 * 
 * <p>
 * The full command-line invocation for the above puzzle is:
 * 
 * <pre>
 * % java -cp . Sudoku 018 034 052 076 \
 * 
 *                     113 124 169 171 \
 * 
 *                     209 216 278 284 \
 * 
 *                     332 341 356     \
 * 
 *                     533 545 557     \
 * 
 *                     608 614 677 685 \
 * 
 *                     712 726 761 773 \
 * 
 *                     819 837 851 874 \
 * 
 * </pre>
 * 
 * <p>
 * See <a href="http://en.wikipedia.org/wiki/Sudoku">Wikipedia: Sudoku</a> for more information on
 * Sudoku.
 * 
 * <p>
 * The algorithm employed is similar to the standard backtracking
 * <a href="http://en.wikipedia.org/wiki/Eight_queens_puzzle">eight queens algorithm</a>.
 * 
 * @version 1.0
 * @author <a href="http://www.colloquial.com/carp">Bob Carpenter</a>
 */
public class Sudoku extends Remoteable {

  private static final long serialVersionUID = -3962977915411306215L;

  private transient DFE dfe;

  private int[][] matrix;

  private String[] input =
      {"006", "073", "102", "131", "149", "217", "235", "303", "345", "361", "378", "422", "465",
          "514", "521", "548", "582", "658", "679", "743", "752", "784", "818", "883"};

  public Sudoku(DFE dfe) {
    this.dfe = dfe;
    matrix = parseProblem(input);
  }

  @Remote
  public boolean localhasSolution() {
    return solve(0, 0, matrix);
  }

  public boolean solve(int i, int j, int[][] cells) {
    if (i == 9) {
      i = 0;
      if (++j == 9)
        return true;
    }
    if (cells[i][j] != 0)
      return solve(i + 1, j, cells);
    for (int val = 1; val <= 9; ++val) {
      if (legal(i, j, val, cells)) {
        cells[i][j] = val;
        if (solve(i + 1, j, cells))
          return true;
      }
    }
    cells[i][j] = 0;
    return false;
  }

  private boolean legal(int i, int j, int val, int[][] cells) {
    for (int k = 0; k < 9; ++k)
      if (val == cells[k][j])
        return false;
    for (int k = 0; k < 9; ++k)
      if (val == cells[i][k])
        return false;
    int boxRowOffset = (i / 3) * 3;
    int boxColOffset = (j / 3) * 3;
    for (int k = 0; k < 3; ++k)
      for (int m = 0; m < 3; ++m)
        if (val == cells[boxRowOffset + k][boxColOffset + m])
          return false;
    return true;
  }

  static int[][] parseProblem(String[] input) {
    int[][] problem = new int[9][9];
    for (int n = 0; n < input.length; ++n) {
      int i = Integer.parseInt(input[n].substring(0, 1));
      int j = Integer.parseInt(input[n].substring(1, 2));
      int val = Integer.parseInt(input[n].substring(2, 3));
      problem[i][j] = val;
    }
    return problem;
  }

  @Override
  public void copyState(Remoteable state) {}

  public boolean hasSolution() {
    Method toExecute;
    Class<?>[] paramTypes = null;
    Object[] paramValues = null;
    boolean result = false;
    try {
      toExecute = this.getClass().getDeclaredMethod("localhasSolution", paramTypes);
      result = (Boolean) dfe.execute(toExecute, paramValues, this);
    } catch (SecurityException e) {
      // Should never get here
      e.printStackTrace();
      throw e;
    } catch (NoSuchMethodException e) {
      // Should never get here
      e.printStackTrace();
    } catch (Throwable e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return result;
  }
}
