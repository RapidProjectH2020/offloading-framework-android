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
package eu.project.rapid.as;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

/**
 * Listens to android.intent.action.BOOT_COMPLETED (defined in AndroidManifest.xml) and starts the
 * execution server when the system has finished booting
 * 
 */
public class AccelerationServerAutoStarter extends BroadcastReceiver {
  private static final String TAG = AccelerationServerAutoStarter.class.getName();

  @Override
  public void onReceive(Context context, Intent intent) {
    Log.d(TAG, "onReceiveIntent: Start Execution Service");

    Intent serviceIntent = new Intent();
    serviceIntent.setAction("eu.project.rapid.as.AccelerationServer");
    context.startService(serviceIntent);
  }
}
