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
package eu.project.rapid.ac;

import java.io.File;
import java.io.Serializable;
import java.util.LinkedList;

import android.util.Log;

public abstract class Remoteable implements Serializable {

  private static final long serialVersionUID = 1L;

  public abstract void copyState(Remoteable state);

  /**
   * Load all provided shared libraries - used when an exception is thrown on the server-side,
   * meaning that the necessary libraries have not been loaded. x86 version of the libraries
   * included in the APK of the remote application are then loaded and the operation is re-executed.
   * 
   * @param libFiles
   */
  public void loadLibraries(LinkedList<File> libFiles) {
    for (File libFile : libFiles) {
      Log.d("Remoteable",
          "Loading library: " + libFile.getName() + " (" + libFile.getAbsolutePath() + ")");
      System.load(libFile.getAbsolutePath());
    }
  }

  /**
   * Override this method if you want to prepare the data before executing the method.
   * 
   * This can be useful in case the class contains data that are not serializable but are needed by
   * the method when offloaded. Use this method to convert the data to some serializable form before
   * the method is offloaded.
   * 
   * Do not explicitly call this method. It will be automatically called by the framework before
   * offloading.
   */
  public void prepareDataOnClient() {};

  /**
   * Override this method if you want to prepare the data before executing the method on the VM.
   * 
   * This can be useful in case the class contains data that are not serializable and were
   * serialized using the method prepareDataOnClient.
   * 
   * Do not explicitly call this method. It will be automatically called by the framework before
   * running the method on the VM.
   */
  public void prepareDataOnServer() {};
}
