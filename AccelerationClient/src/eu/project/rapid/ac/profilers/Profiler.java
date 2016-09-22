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

import android.util.Log;
import eu.project.rapid.ac.db.DBCache;
import eu.project.rapid.ac.db.DBEntry;
import eu.project.rapid.ac.profilers.phone.Phone;
import eu.project.rapid.ac.profilers.phone.PhoneFactory;
import eu.project.rapid.ac.utils.Constants;


public class Profiler {

  private static final String TAG = "Profiler";

  Phone phone;

  private ProgramProfiler progProfiler;
  private NetworkProfiler netProfiler;
  private DeviceProfiler devProfiler;
  private int mRegime;
  public static final int REGIME_CLIENT = 1;
  public static final int REGIME_SERVER = 2;

  private static FileWriter logFileWriter;

  private String mLocation;

  private LogRecord lastLogRecord;

  public Profiler(int regime, ProgramProfiler progProfiler, NetworkProfiler netProfiler,
      DeviceProfiler devProfiler) {
    this.progProfiler = progProfiler;
    this.netProfiler = netProfiler;
    this.devProfiler = devProfiler;
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

  private void stopProfilers() {

    if (mRegime == REGIME_CLIENT) {
      devProfiler.stopAndCollectDeviceProfiling();
    }

    progProfiler.stopAndCollectExecutionInfoTracking();

    if (netProfiler != null) {
      netProfiler.stopAndCollectTransmittedData();
    }
  }

  /**
   * Stop running profilers and discard current information
   * 
   */
  public void stopAndDiscardExecutionInfoTracking() {
    stopProfilers();
  }

  /**
   * Stop running profilers and log current information
   * 
   */
  public void stopAndLogExecutionInfoTracking(long prepareDataDuration, Long pureExecTime) {

    stopProfilers();

    lastLogRecord = new LogRecord(progProfiler, netProfiler, devProfiler);
    lastLogRecord.prepareDataDuration = prepareDataDuration;
    lastLogRecord.pureDuration = pureExecTime;
    lastLogRecord.execLocation = mLocation;

    if (mRegime == REGIME_CLIENT) {
      phone = PhoneFactory.getPhone(devProfiler, netProfiler, progProfiler);
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
            // Try creating new, if doesn't exist
            boolean logFileCreated = logFile.createNewFile();
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

      updateDbCache();
    }
  }

  private void updateDbCache() {
    DBCache dbCache = DBCache.getDbCache();
    // public DBEntry(String appName, String methodName, String execLocation, String networkType,
    // String networkSubType, int ulRate, int dlRate, long execDuration, long execEnergy)
    DBEntry dbEntry =
        new DBEntry(lastLogRecord.appName, lastLogRecord.methodName, lastLogRecord.execLocation,
            lastLogRecord.networkType, lastLogRecord.networkSubtype, lastLogRecord.ulRate,
            lastLogRecord.dlRate, lastLogRecord.execDuration, lastLogRecord.energyConsumption);
    dbCache.insertEntry(dbEntry);
  }
}
