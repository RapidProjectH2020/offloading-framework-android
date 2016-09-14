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

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.StringTokenizer;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.provider.Settings;
import android.util.Log;

/**
 * Device state profiler - currently only tracks battery state, listening to ACTION_BATTERY_CHANGED
 * intent to update the battery level and allows to track change in voltage between two points in a
 * program (where battery voltage readings are taken from /sys/class/power_supply, based on Android
 * OS source)
 */
public class DeviceProfiler {

  private static final String TAG = "DeviceProfiler";

  public static int batteryLevel;
  public static boolean batteryTrackingOn = false;
  private static Object batteryTrackingSyncObject = new Object();

  private Long mStartBatteryVoltage;
  public Long batteryVoltageDelta;

  private static Context context;

  /**
   * Variables for CPU Usage
   */
  private int PID;
  private boolean stopReadingFiles;
  private ArrayList<Long> pidCpuUsage;
  private ArrayList<Long> systemCpuUsage;
  private long uTime;
  private long sTime;
  private long pidTime;
  private long diffPidTime;
  private long prevPidTime;
  private long userMode;
  private long niceMode;
  private long systemMode;
  private long idleTask;
  private long ioWait;
  private long irq;
  private long softirq;
  private long runningTime;
  private long prevrunningTime;
  private long diffRunningTime;
  private final String pidStatFile;
  private final String statFile;
  private long diffIdleTask;
  private long prevIdleTask;
  private ArrayList<Long> idleSystem;
  private ArrayList<Integer> screenBrightness;

  private static BroadcastReceiver batteryLevelReceiver;

  /**
   * Variables for CPU frequency<br>
   * Obtained reading the files:<br>
   * /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq<br>
   * /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq<br>
   * /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
   */
  private int currentFreq; // The current frequency in KHz
  private ArrayList<Integer> frequence;
  private final String curFreqFile = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";

  public DeviceProfiler(Context context) {
    DeviceProfiler.context = context;
    batteryVoltageDelta = null;

    pidCpuUsage = new ArrayList<Long>();
    systemCpuUsage = new ArrayList<Long>();
    idleSystem = new ArrayList<Long>();
    frequence = new ArrayList<Integer>();
    screenBrightness = new ArrayList<Integer>();

    PID = android.os.Process.myPid();

    pidStatFile = "/proc/" + PID + "/stat";
    statFile = "/proc/stat";

    synchronized (this) {
      stopReadingFiles = false;
    }
  }

  /**
   * Start device information tracking from a certain point in a program (currently only battery
   * voltage)
   */
  public void startDeviceProfiling() {
    mStartBatteryVoltage = SysClassBattery.getCurrentVoltage();
    calculatePidCpuUsage();
    calculateScreenBrightness();
  }

  public static void onDestroy() {
    if (batteryLevelReceiver != null) {
      DeviceProfiler.context.unregisterReceiver(batteryLevelReceiver);
    }
    synchronized (batteryTrackingSyncObject) {
      batteryTrackingOn = false;
    }
  }

  /**
   * Stop device information tracking and store the data in the object
   */
  public void stopAndCollectDeviceProfiling() {
    batteryVoltageDelta = SysClassBattery.getCurrentVoltage() - mStartBatteryVoltage;

    synchronized (this) {
      // Log.i(TAG, "Flag to stopReadingFiles set to true");
      stopReadingFiles = true;
    }
  }

  /**
   * Computes the battery level by registering a receiver to the intent triggered by a battery
   * status/level change.
   */
  public static void trackBatteryLevel() {
    if (batteryTrackingOn == false) {
      batteryLevelReceiver = new BroadcastReceiver() {
        public void onReceive(Context context, Intent intent) {
          // context.unregisterReceiver(this);
          int rawlevel = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
          int scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
          int level = -1;
          if (rawlevel >= 0 && scale > 0) {
            level = (rawlevel * 100) / scale;
          }
          Log.d(TAG,
              "Battery level - " + level + ", voltage - " + SysClassBattery.getCurrentVoltage());
          batteryLevel = level;
        }
      };
      IntentFilter batteryLevelFilter = new IntentFilter(Intent.ACTION_BATTERY_CHANGED);
      context.registerReceiver(batteryLevelReceiver, batteryLevelFilter);
      synchronized (batteryTrackingSyncObject) {
        batteryTrackingOn = true;
      }
    }
  }

  /**
   * Class exposing battery information, based on battery service and Android OS implementation
   */
  private static class SysClassBattery {
    private final static String SYS_CLASS_POWER = "/sys/class/power_supply";
    private final static String BATTERY = "/battery";
    private final static String VOLTAGE = "/batt_vol";
    private final static String VOLTAGE_ALT = "/voltage_now";

    /**
     * Read current battery voltage from /sys/class/power_supply/battery/batt_vol or
     * /sys/class/power_supply/battery/voltage_now - try both files since it is done in the battery
     * service of Android, so must be model/version dependent
     */
    public static Long getCurrentVoltage() {
      StringBuilder sb = new StringBuilder();
      sb.append(SYS_CLASS_POWER).append(BATTERY).append(VOLTAGE);
      Long result = readLong(sb.toString());
      if (result != -1)
        return result;
      else {
        sb = new StringBuilder();
        sb.append(SYS_CLASS_POWER).append(BATTERY).append(VOLTAGE_ALT);
        result = readLong(sb.toString());
        return result;
      }

    }

    private static RandomAccessFile getFile(String filename) throws IOException {
      File f = new File(filename);
      return new RandomAccessFile(f, "r");
    }

    private static long readLong(String file) {
      RandomAccessFile raf = null;
      try {
        raf = getFile(file);
        return Long.valueOf(raf.readLine());
      } catch (Exception e) {
        // Log.d(TAG, "Could not read voltage: " + e);
        return -1;
      } finally {
        if (raf != null) {
          try {
            raf.close();
          } catch (IOException e) {
          }
        }
      }
    }
  }

  /**
   * Calculate the CPU usage of process every second<br>
   * These values are registered in the array <b>pidCpuUsage[]</b>
   */
  private void calculatePidCpuUsage() {
    Thread t = new Thread("PidCpuUsage") {
      public void run() {

        boolean firstTime = true;

        while (!stopReadingFiles) {

          try {
            calculateProcessExecutionTime();
            calculateSystemExecutionTime();

            try {
              getCurrentCpuFreq();
            } catch (Exception e) {
              // Do nothing. We know that in remote side this will throw an exception.
            }

            /**
             * To prevent errors from the first run don't consider it
             */
            if (!firstTime) {
              pidCpuUsage.add(diffPidTime);
              systemCpuUsage.add(diffRunningTime);
              frequence.add(currentFreq);
              idleSystem.add(diffIdleTask);

              Log.i("DeviceProfiler", "PidCpuTime: " + diffPidTime);
              Log.i("DeviceProfiler", "Frequence: " + currentFreq);
            }

            prevPidTime = pidTime;
            prevrunningTime = runningTime;
            prevIdleTask = idleTask;

            firstTime = false;

            try {
              Thread.sleep(1000);
            } catch (InterruptedException e) {
            }
          } catch (FileNotFoundException e1) {
            Log.e(TAG, "Stopping device profiling: FileNotFoundException - " + e1);
            stopReadingFiles = true;
          } catch (IOException e1) {
            Log.e(TAG, "Stopping device profiling: IOException - " + e1);
            stopReadingFiles = true;
          }
        }
      }
    };
    t.start();
  }

  /**
   * Open the file <code>/proc/$PID/stat</code> and read utime and stime<br>
   * <b>utime</b>: execution of process in user mode (in jiffies)<br>
   * <b>stime</b>: execution of process in kernel mode (in jiffies)<br>
   * These are 14th and 15th variables respectively in the file<br>
   * The sum <b>pidTime = utime + stime</b> gives the total running time of process<br>
   * <b>diffPidTime</b> is the running time of process during the last second<br>
   * 
   * @throws IOException
   */
  private void calculateProcessExecutionTime() throws FileNotFoundException, IOException {
    BufferedReader brPidStat = null;

    try {
      brPidStat = new BufferedReader(new FileReader(pidStatFile));

      String strLine = brPidStat.readLine();
      StringTokenizer st = new StringTokenizer(strLine);

      for (int i = 1; i < 14; i++)
        st.nextToken();

      uTime = Long.parseLong(st.nextToken());
      sTime = Long.parseLong(st.nextToken());
      pidTime = uTime + sTime;
      diffPidTime = pidTime - prevPidTime;
    } finally {
      try {
        if (brPidStat != null)
          brPidStat.close();
      } catch (IOException e) {
      }
    }
  }

  /**
   * Open the file "/proc/stat" and read information about system execution<br>
   * <b>userMode</b>: normal processes executing in user mode (in jiffies)<br>
   * <b>niceMode</b>: niced processes executing in user mode (in jiffies)<br>
   * <b>systemMode</b>: processes executing in kernel mode (in jiffies)<br>
   * <b>idleTask</b>: twiddling thumbs (in jiffies)<br>
   * <b>runningTime</b>: total time of execution (in jiffies)<br>
   * <b>ioWait</b>: waiting for I/O to complete (in jiffies)<br>
   * <b>irq</b>: servicing interrupts (in jiffies)<br>
   * <b>softirq</b>: servicing softirq (in jiffies)<br>
   * <b>diffRunningTime</b>: time of execution during the last second (in jiffies)<br>
   * 
   * @throws IOException
   */
  private void calculateSystemExecutionTime() throws FileNotFoundException, IOException {
    BufferedReader brStat = null;

    try {
      brStat = new BufferedReader(new FileReader(statFile));

      String strLine = brStat.readLine();
      StringTokenizer st = new StringTokenizer(strLine);
      st.nextToken();

      userMode = Long.parseLong(st.nextToken());
      niceMode = Long.parseLong(st.nextToken());
      systemMode = Long.parseLong(st.nextToken());
      idleTask = Long.parseLong(st.nextToken());
      ioWait = Long.parseLong(st.nextToken());
      irq = Long.parseLong(st.nextToken());
      softirq = Long.parseLong(st.nextToken());

      // runningTime = userMode + niceMode + systemMode + idleTask + ioWait + irq + softirq;
      idleTask += ioWait;
      runningTime = userMode + niceMode + systemMode + irq + softirq;
      diffRunningTime = runningTime - prevrunningTime;
      diffIdleTask = idleTask - prevIdleTask;
    } finally {
      try {
        if (brStat != null)
          brStat.close();
      } catch (IOException e) {
      }
    }
  }

  private void getCurrentCpuFreq() throws FileNotFoundException, IOException {
    BufferedReader brFreq = null;

    try {
      brFreq = new BufferedReader(new FileReader(curFreqFile));

      String strLine = brFreq.readLine();
      currentFreq = Integer.parseInt(strLine);
    } finally {
      try {
        if (brFreq != null)
          brFreq.close();
      } catch (IOException e) {
      }
    }
  }

  /**
   * For now the implementation is very dummy: is assumed that the screen is always ON during the
   * execution.
   * 
   * TODO: better implementation of this method, account also the fact that the screen can go off.
   * 
   */

  private void calculateScreenBrightness() {
    Thread t = new Thread("ScreenBrightness") {
      public void run() {

        while (!stopReadingFiles) {

          int brightness = Settings.System.getInt(context.getContentResolver(),
              Settings.System.SCREEN_BRIGHTNESS, -1);

          screenBrightness.add(brightness);

          // Log.i(TAG, "Screen brightness: " + brightness);

          try {
            Thread.sleep(1000);
          } catch (InterruptedException e) {
          }
        }
      }
    };
    t.start();
  }

  public int getSeconds() {
    return pidCpuUsage.size();
  }

  public long getSystemCpuUsage(int i) {
    return systemCpuUsage.get(i);
  }

  public long getPidCpuUsage(int i) {
    return pidCpuUsage.get(i);
  }

  public int getFrequence(int i) {
    return frequence.get(i);
  }

  public long getIdleSystem(int i) {
    return idleSystem.get(i);
  }

  public int getScreenBrightness(int i) {
    return screenBrightness.get(i);
  }
}
