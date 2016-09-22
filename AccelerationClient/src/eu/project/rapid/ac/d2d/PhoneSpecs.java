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
package eu.project.rapid.ac.d2d;

import java.io.Serializable;

import android.content.Context;
import android.util.Log;
import eu.project.rapid.ac.utils.Utils;

/**
 * This object contains the specifics of a phone. In a D2D communication scenario, we will receive
 * the specifics from the nearby phones and will sort them based on CPU, memory, GPU.<br>
 * 
 * @author sokol
 *
 */
public class PhoneSpecs implements Serializable, Comparable<PhoneSpecs> {
  private static final long serialVersionUID = -4918806738265004873L;

  private static final String TAG = "PhoneSpecs";

  private String phoneId;
  private long timestamp; // TimeStamp to be used by the clients as a mean to measure the freshness
                          // of this phone
  private String ip;
  private int nrCPUs; // number of CPU cores
  private int cpuFreqKHz; // CPU frequency in KHz
  private int ramMB; // Memory in MB
  private boolean hasGpu;

  private static PhoneSpecs phoneSpecs;

  /**
   * @param context Context of the application calling this method.
   */
  private PhoneSpecs(Context context) throws NullPointerException {
    if (context == null) {
      throw new NullPointerException("Context cannot be null");
    }

    // FIXME: On Android 6 we can't just read the ID directly, we need to ask for runtime
    // permission.
    phoneId = Utils.getDeviceIdHashHex(context);
    nrCPUs = Utils.getDeviceNrCPUs();
    cpuFreqKHz = Utils.getDeviceCPUFreq();
    try {
      ip = Utils.getIpAddress().getHostAddress();
    } catch (Exception e) {
      Log.w(TAG,
          "Error while getting the IP (most probably we are not connected to WiFi network): " + e);
    }
  }

  public static PhoneSpecs getPhoneSpecs(Context context) {
    if (phoneSpecs == null) {
      phoneSpecs = new PhoneSpecs(context);
    }

    return phoneSpecs;
  }

  /**
   * @return the phoneId
   */
  public String getPhoneId() {
    return phoneId;
  }

  /**
   * @param phoneId the phoneId to set
   */
  public void setPhoneId(String phoneId) {
    this.phoneId = phoneId;
  }

  /**
   * @return the timestamp
   */
  public long getTimestamp() {
    return timestamp;
  }

  /**
   * @param timestamp the timestamp to set
   */
  public void setTimestamp(long timestamp) {
    this.timestamp = timestamp;
  }

  /**
   * @return the ip
   */
  public String getIp() {
    return ip;
  }

  /**
   * @param ip the ip to set
   */
  public void setIp(String ip) {
    this.ip = ip;
  }

  /**
   * @return the nrCPUs
   */
  public int getNrCPUs() {
    return nrCPUs;
  }

  /**
   * @param nrCPUs the nrCPUs to set
   */
  public void setNrCPUs(int nrCPUs) {
    this.nrCPUs = nrCPUs;
  }

  /**
   * @return the cpuFreqKHz
   */
  public int getCpuPowerKHz() {
    return cpuFreqKHz;
  }

  /**
   * @param cpuFreqKHz the cpuFreqKHz to set
   */
  public void setCpuPowerKHz(int cpuPowerKHz) {
    this.cpuFreqKHz = cpuPowerKHz;
  }

  /**
   * @return the ramMB
   */
  public int getRamMB() {
    return ramMB;
  }

  /**
   * @param ramMB the ramMB to set
   */
  public void setRamMB(int ramMB) {
    this.ramMB = ramMB;
  }

  /**
   * @return the hasGpu
   */
  public boolean isHasGpu() {
    return hasGpu;
  }

  /**
   * @param hasGpu the hasGpu to set
   */
  public void setHasGpu(boolean hasGpu) {
    this.hasGpu = hasGpu;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = prime + ((phoneId == null) ? 0 : phoneId.hashCode());
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }

    final PhoneSpecs other = (PhoneSpecs) obj;
    if (phoneId == other.phoneId) {
      return true;
    }

    return false;
  }


  @Override
  public int compareTo(PhoneSpecs otherPhone) {
    if (otherPhone == null) {
      return 1;
    }

    if (this.phoneId == otherPhone.phoneId) {
      return 0;
    }

    if (this.nrCPUs > otherPhone.nrCPUs || this.cpuFreqKHz > otherPhone.cpuFreqKHz) {
      return 1;
    } else if (this.cpuFreqKHz < otherPhone.cpuFreqKHz) {
      return -1;
    } else {
      return this.ramMB - otherPhone.ramMB;
    }
  }

  @Override
  public String toString() {
    return "ID=" + this.phoneId + ", nrCPUs=" + this.nrCPUs + ", CPU=" + this.cpuFreqKHz + " KHz"
        + ", RAM=" + this.ramMB + " MB" + ", GPU=" + this.hasGpu + ", IP=" + this.ip;
  }
}
