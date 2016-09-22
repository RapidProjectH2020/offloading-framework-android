/*******************************************************************************
 * Copyright (C) 2015, 2016 RAPID EU Project
 *
 * This library is free software; you can redistribute it and/or modify it under the terms of the
 * GNU Lesser General Public License as published by the Free Software Foundation; either version
 * 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with this library;
 * if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301 USA
 *******************************************************************************/
package eu.project.rapid.synthBenchmark;

import java.lang.reflect.Method;
import java.util.Random;

import eu.project.rapid.ac.DFE;
import eu.project.rapid.ac.Remoteable;

/**
 * Simple class that tests offloading for trivial applications.
 */
public class TestRemoteable extends Remoteable {
  private static final long serialVersionUID = 1L;

  public transient DFE dfe;

  public TestRemoteable(DFE dfe) {
    this.dfe = dfe;
  }

  public String cpuLoader1() {
    Method toExecute;
    String result = "";
    try {
      toExecute = this.getClass().getDeclaredMethod("localCpuLoader1", (Class[]) null);
      result = (String) dfe.execute(toExecute, this);
    } catch (SecurityException e) {
      e.printStackTrace();
      throw e;
    } catch (NoSuchMethodException e) {
      e.printStackTrace();
    } catch (Throwable e) {
      e.printStackTrace();
    }
    return result;
  }

  public String localCpuLoader1() {
    for (int i = 0; i < 10000; i++) {
    }
    return "cpuLoader1 finished";
  }

  public String cpuLoader2() {
    Method toExecute;
    String result = "";
    try {
      toExecute = this.getClass().getDeclaredMethod("localCpuLoader2", (Class[]) null);
      result = (String) dfe.execute(toExecute, this);
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

  public Long cpuLoader3(int seed) {
    Method toExecute;
    Class<?>[] paramTypes = {int.class};
    Object[] paramValues = {seed};
    Long result = null;
    try {
      toExecute = this.getClass().getDeclaredMethod("localCpuLoader3", paramTypes);
      result = (Long) dfe.execute(toExecute, paramValues, this);
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

  public String localCpuLoader2() {
    for (int i = 0; i < 500000; i++) {
    }
    return "cpuLoader2 finished";
  }

  public Long localCpuLoader3(int seed) {
    Random rand = new Random(seed);
    return rand.nextLong();
  }

  @Override
  public void copyState(Remoteable state) {
    // No fields to restore
  }
}
