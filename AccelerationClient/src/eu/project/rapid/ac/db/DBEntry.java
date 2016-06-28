package eu.project.rapid.ac.db;

import java.io.Serializable;

public class DBEntry implements Serializable {

  private static final long serialVersionUID = 1L;

  private String appName;
  private String methodName;
  private String execLocation;
  private String networkType;
  private String networkSubType;
  private int ulRate;
  private int dlRate;
  private long execDuration;
  private long execEnergy;
  private long timestamp;

  public DBEntry(String appName, String methodName, String execLocation, String networkType,
      String networkSubType, int ulRate, int dlRate, long execDuration, long execEnergy) {

    this.appName = appName;
    this.methodName = methodName;
    this.execLocation = execLocation;
    this.networkType = networkType;
    this.networkSubType = networkSubType;
    this.ulRate = ulRate;
    this.dlRate = dlRate;
    this.execDuration = execDuration;
    this.execEnergy = execEnergy;
    this.timestamp = System.currentTimeMillis();
  }

  /**
   * @return the appName
   */
  public String getAppName() {
    return appName;
  }

  /**
   * @param appName the appName to set
   */
  public void setAppName(String appName) {
    this.appName = appName;
  }

  /**
   * @return the methodName
   */
  public String getMethodName() {
    return methodName;
  }

  /**
   * @param methodName the methodName to set
   */
  public void setMethodName(String methodName) {
    this.methodName = methodName;
  }

  /**
   * @return the execLocation
   */
  public String getExecLocation() {
    return execLocation;
  }

  /**
   * @param execLocation the execLocation to set
   */
  public void setExecLocation(String execLocation) {
    this.execLocation = execLocation;
  }

  /**
   * @return the networkType
   */
  public String getNetworkType() {
    return networkType;
  }

  /**
   * @param networkType the networkType to set
   */
  public void setNetworkType(String networkType) {
    this.networkType = networkType;
  }

  /**
   * @return the networkSubType
   */
  public String getNetworkSubType() {
    return networkSubType;
  }

  /**
   * @param networkSubType the networkSubType to set
   */
  public void setNetworkSubType(String networkSubType) {
    this.networkSubType = networkSubType;
  }

  /**
   * @return the ulRate
   */
  public int getUlRate() {
    return ulRate;
  }

  /**
   * @param ulRate the ulRate to set
   */
  public void setUlRate(int ulRate) {
    this.ulRate = ulRate;
  }

  /**
   * @return the dlRate
   */
  public int getDlRate() {
    return dlRate;
  }

  /**
   * @param dlRate the dlRate to set
   */
  public void setDlRate(int dlRate) {
    this.dlRate = dlRate;
  }

  /**
   * @return the execDuration
   */
  public long getExecDuration() {
    return execDuration;
  }

  /**
   * @param execDuration the execDuration to set
   */
  public void setExecDuration(long execDuration) {
    this.execDuration = execDuration;
  }

  /**
   * @return the execEnergy
   */
  public long getExecEnergy() {
    return execEnergy;
  }

  /**
   * @param execEnergy the execEnergy to set
   */
  public void setExecEnergy(long execEnergy) {
    this.execEnergy = execEnergy;
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

}
