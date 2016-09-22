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
package eu.project.rapid.as;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectStreamClass;

import dalvik.system.DexClassLoader;

/**
 * Custom object input stream to also deal with dynamically loaded classes. The classes can be
 * retrieved from Android Dex files, provided in Apk (android application) files.
 * 
 */
public class DynamicObjectInputStream extends ObjectInputStream {
  private static final String TAG = "DynamicObjectInputStream";

  private ClassLoader mCurrent = null;
  private DexClassLoader mCurrentDexLoader = null;

  public DynamicObjectInputStream(InputStream in) throws IOException {
    super(in);
  }

  /**
   * Server side only. Need to have the server's classloaders otherwise if the classloaders are
   * initialized here it doesn't work correctly.
   * 
   * @param classLoader
   * @param dexClassLoader
   */
  public void setClassLoaders(ClassLoader classLoader, DexClassLoader dexClassLoader) {
    mCurrent = classLoader;
    mCurrentDexLoader = dexClassLoader;
  }

  /**
   * Override the method resolving a class to also look into the constructed DexClassLoader
   */
  @Override
  protected Class<?> resolveClass(ObjectStreamClass desc)
      throws IOException, ClassNotFoundException {
    // Log.i(TAG, "Resolving class: " + desc.getName());

    try {
      try {
        return mCurrent.loadClass(desc.getName());
      } catch (ClassNotFoundException e) {
        return mCurrentDexLoader.loadClass(desc.getName());
      }
    } catch (ClassNotFoundException e) {
      return super.resolveClass(desc);
    } catch (NullPointerException e) {
      // Thrown when currentDexLoader is
      // not yet set up
      return super.resolveClass(desc);
    }
  }

  /**
   * Add a Dex file to the Class Loader for dynamic class loading for clients
   * 
   * @param apkFile the apk package
   */
  public DexClassLoader addDex(final File apkFile) {

    if (mCurrentDexLoader == null) {
      mCurrentDexLoader = new DexClassLoader(apkFile.getAbsolutePath(),
          apkFile.getParentFile().getAbsolutePath(), null, mCurrent);
    } else {
      mCurrentDexLoader = new DexClassLoader(apkFile.getAbsolutePath(),
          apkFile.getParentFile().getAbsolutePath(), null, mCurrentDexLoader);
    }

    return mCurrentDexLoader;
  }
}
