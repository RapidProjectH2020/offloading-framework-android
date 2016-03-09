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
package eu.project.rapid.ac.d2d;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

public class D2DService extends Service {

  private static final String TAG = "D2DService";

  @Override
  public void onCreate() {

    Log.i(TAG, "onCreate()");

  }

  @Override
  public int onStartCommand(Intent intent, int flags, int startId) {

    Log.i(TAG, "onStartCommand()");

    return START_STICKY;
  }

  @Override
  public void onDestroy() {

  }

  @Override
  public IBinder onBind(Intent intent) {
    return null;
  }

}
