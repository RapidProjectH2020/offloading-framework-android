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

import android.util.Log;
import eu.project.rapid.ac.DFE;
import eu.project.rapid.ac.Remoteable;

public class JniTest extends Remoteable {

  /**
   * 
   */
  private static final long serialVersionUID = 7407706990063388777L;

  private transient DFE dfe;

  public int temp = 0;

  public JniTest(DFE dfe) {
    this.dfe = dfe;
  }

  /*
   * A native method that is implemented by the 'hello-jni' native library, which is packaged with
   * this application.
   */
  public native String stringFromJNI();

  static {
    try {
      System.loadLibrary("hello-jni");
    } catch (UnsatisfiedLinkError e) {
      Log.i("JniTest", "Could not load native library, maybe this is running on the clone.");
    }
  }

  public String jniCaller() {
    Method toExecute;
    String result = null;
    try {
      toExecute = this.getClass().getDeclaredMethod("localjniCaller", (Class[]) null);
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

  public String localjniCaller() {
    return stringFromJNI();
  }

  @Override
  public void copyState(Remoteable arg0) {
    // TODO Auto-generated method stub
  }
}
