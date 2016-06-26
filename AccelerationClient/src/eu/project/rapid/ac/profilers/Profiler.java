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
package eu.project.rapid.ac.profilers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import android.content.Context;
import android.util.Log;
import eu.project.rapid.ac.db.DatabaseQuery;
import eu.project.rapid.ac.profilers.phone.Phone;
import eu.project.rapid.ac.profilers.phone.PhoneHtcDream;
import eu.project.rapid.ac.utils.Constants;

public class Profiler {

  private static final String TAG = "Profiler";

  Phone phone;

  private ProgramProfiler progProfiler;
  private NetworkProfiler netProfiler;
  private DeviceProfiler devProfiler;
  private Context mContext;
  private int mRegime;
  public static final int REGIME_CLIENT = 1;
  public static final int REGIME_SERVER = 2;

  private static FileWriter logFileWriter;

  private String mLocation;

  public LogRecord lastLogRecord;


  // private final int MIN_FREQ = 480000; // The minimum frequency for HTC Hero CPU
  // private final int MAX_FREQ = 528000; // The maximum frequency for HTC Hero CPU

  public Profiler(int regime, Context context, ProgramProfiler progProfiler,
      NetworkProfiler netProfiler, DeviceProfiler devProfiler) {
    this.progProfiler = progProfiler;
    this.netProfiler = netProfiler;
    this.devProfiler = devProfiler;
    this.mContext = context;
    this.mRegime = regime;

    if (mRegime == REGIME_CLIENT) {
      // this.devProfiler.trackBatteryLevel();
    }
  }

  public void startExecutionInfoTracking() {
    if (netProfiler != null) {
      netProfiler.startTransmittedDataCounting();
      mLocation = "REMOTE";
    } else {
      mLocation = "LOCAL";
    }
    Log.d(TAG, mLocation + " " + progProfiler.methodName);
    progProfiler.startExecutionInfoTracking();

    if (mRegime == REGIME_CLIENT) {
      devProfiler.startDeviceProfiling();
    }
  }

  /**
   * Stop running profilers and log current information
   * 
   */
  public LogRecord stopAndLogExecutionInfoTracking(long prepareDataDuration, Long pureExecTime) {

    if (mRegime == REGIME_CLIENT) {
      devProfiler.stopAndCollectDeviceProfiling();
    }

    progProfiler.stopAndCollectExecutionInfoTracking();

    if (netProfiler != null) {
      netProfiler.stopAndCollectTransmittedData();
    }

    lastLogRecord = new LogRecord(progProfiler, netProfiler, devProfiler);
    lastLogRecord.prepareDataDuration = prepareDataDuration;
    lastLogRecord.pureDuration = pureExecTime;
    lastLogRecord.execLocation = mLocation;

    if (mRegime == REGIME_CLIENT) {
      // Energy model implemented for HTC G1 following PowertTutor
      if (android.os.Build.MODEL.equals(Constants.PHONE_NAME_HTC_G1)) {
        phone = new PhoneHtcDream(devProfiler, netProfiler, progProfiler);

        // Add here all the cases for the other phones.
      } else {
        // By default, if the current phone's call is not implemented yet, use the htc dream.
        phone = new PhoneHtcDream(devProfiler, netProfiler, progProfiler);
      }

      phone.estimateEnergyConsumption();
      lastLogRecord.energyConsumption = phone.getTotalEstimatedEnergy();
      lastLogRecord.cpuEnergy = phone.getEstimatedCpuEnergy();
      lastLogRecord.screenEnergy = phone.getEstimatedScreenEnergy();
      lastLogRecord.wifiEnergy = phone.getEstimatedWiFiEnergy();
      lastLogRecord.threeGEnergy = phone.getEstimated3GEnergy();

      Log.d(TAG, "Log record - " + lastLogRecord.toString());

      try {
        synchronized (this) {
          if (logFileWriter == null) {
            File logFile = new File(Constants.LOG_FILE_NAME);
            boolean logFileCreated = logFile.createNewFile(); // Try creating new, if doesn't exist
            logFileWriter = new FileWriter(logFile, true);
            if (logFileCreated) {
              logFileWriter.append(LogRecord.LOG_HEADERS + "\n");
            }
          }

          logFileWriter.append(lastLogRecord.toString() + "\n");
          logFileWriter.flush();
        }
      } catch (IOException e) {
        Log.w(TAG, "Not able to create the logFile " + Constants.LOG_FILE_NAME + ": " + e);
      }

      updateDB();
    }

    return lastLogRecord;
  }

  private void updateDB() {

    DatabaseQuery query = new DatabaseQuery(mContext);

    // Insert the new record in the DB
    query.appendData(DatabaseQuery.KEY_APP_NAME, lastLogRecord.appName);
    query.appendData(DatabaseQuery.KEY_METHOD_NAME, lastLogRecord.methodName);
    query.appendData(DatabaseQuery.KEY_EXEC_LOCATION, lastLogRecord.execLocation);
    query.appendData(DatabaseQuery.KEY_NETWORK_TYPE, lastLogRecord.networkType);
    query.appendData(DatabaseQuery.KEY_NETWORK_SUBTYPE, lastLogRecord.networkSubtype);
    query.appendData(DatabaseQuery.KEY_UL_RATE, Integer.toString(lastLogRecord.ulRate));
    query.appendData(DatabaseQuery.KEY_DL_RATE, Integer.toString(lastLogRecord.dlRate));
    query.appendData(DatabaseQuery.KEY_EXEC_DURATION, Long.toString(lastLogRecord.execDuration));
    query.appendData(DatabaseQuery.KEY_EXEC_ENERGY, Long.toString(lastLogRecord.energyConsumption));
    query.appendData(DatabaseQuery.KEY_TIMESTAMP, Long.toString(System.currentTimeMillis()));

    query.addRow();

    // Close the database
    try {
      query.destroy();
    } catch (Throwable e) {
      e.printStackTrace();
    }
  }
}
