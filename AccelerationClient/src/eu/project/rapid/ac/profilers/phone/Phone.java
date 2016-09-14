package eu.project.rapid.ac.profilers.phone;

import android.util.Log;
import eu.project.rapid.ac.profilers.DeviceProfiler;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.ac.profilers.ProgramProfiler;

/**
 * This is the abstract class that contains the coefficients for the energy estimation model.
 * 
 * @author sokol
 *
 */
public abstract class Phone {

  private final String TAG = "Phone";

  private DeviceProfiler devProfiler;
  private NetworkProfiler netProfiler;
  private ProgramProfiler progProfiler;

  // CPU
  int MIN_FREQ; // The minimum frequency of the CPU
  int MAX_FREQ; // The maximum frequency of the CPU
  double betaUh;
  double betaUl;
  double betaCpu;

  // Screen
  double betaBrightness;

  // WiFi
  double betaWiFiHigh;
  int betaWiFiLow;

  // 3G
  int beta3GIdle;
  int beta3GFACH;
  int beta3GDCH;

  private long totalEstimatedEnergy;
  private double estimatedCpuEnergy;
  private double estimatedScreenEnergy;
  private double estimatedWiFiEnergy;
  private double estimated3GEnergy;

  public Phone(DeviceProfiler devProfiler, NetworkProfiler netProfiler,
      ProgramProfiler progProfiler) {

    this.devProfiler = devProfiler;
    this.netProfiler = netProfiler;
    this.progProfiler = progProfiler;
  }

  public void estimateEnergyConsumption() {
    int duration = devProfiler.getSeconds();

    estimatedCpuEnergy = estimateCpuEnergy(duration);
    Log.d(TAG, "CPU energy: " + estimatedCpuEnergy + " mJ");

    estimatedScreenEnergy = estimateScreenEnergy(duration);
    Log.d(TAG, "Screen energy: " + estimatedScreenEnergy + " mJ");

    // if (lastLogRecord.execLocation.equals("REMOTE") && lastLogRecord.networkType.equals("WIFI"))
    // {
    estimatedWiFiEnergy = estimateWiFiEnergy(duration);
    Log.d(TAG, "WiFi energy: " + estimatedWiFiEnergy + " mJ");
    // } else if (lastLogRecord.execLocation.equals("REMOTE") &&
    // lastLogRecord.networkType.equals("MOBILE")) {
    estimated3GEnergy = estimate3GEnergy(duration);
    Log.d(TAG, "3G energy: " + estimated3GEnergy + " mJ");
    // }

    totalEstimatedEnergy = (long) (estimatedCpuEnergy + estimatedScreenEnergy + estimatedWiFiEnergy
        + estimated3GEnergy);

    Log.d(TAG, "Total energy: " + totalEstimatedEnergy + " mJ");
    Log.d(TAG, "-------------------------------------------");
  }

  /**
   * Estimate the Power for the CPU every second: P0, P1, P2, ..., Pt<br>
   * where t is the execution time in seconds.<br>
   * If we calculate the average power Pm = (P0 + P1 + ... + Pt) / t and multiply<br>
   * by the execution time we obtain the Energy consumed by the CPU executing the method.<br>
   * This is: E_cpu = Pm * t which is equal to: E_cpu = P0 + P1 + ... + Pt<br>
   * NOTE: This is due to the fact that we measure every second.<br>
   * 
   * @param duration Duration of method execution
   * @return The estimated energy consumed by the CPU (mJ)
   * 
   */
  private double estimateCpuEnergy(int duration) {
    double estimatedCpuEnergy = 0;
    byte freqL = 0, freqH = 0;
    int util;
    byte cpuON;

    for (int i = 0; i < duration; i++) {
      util = calculateCpuUtil(i);

      if (devProfiler.getFrequence(i) == MAX_FREQ) {
        freqH = 1;
      } else {
        freqL = 1;
      }

      /**
       * If the CPU has been in idle state for more than 90 jiffies<br>
       * then decide to consider it in idle state for all the second (1 jiffie = 1/100 sec)
       */
      cpuON = (byte) ((devProfiler.getIdleSystem(i) < 90) ? 1 : 0);

      estimatedCpuEnergy += (betaUh * freqH + betaUl * freqL) * util + betaCpu * cpuON;

      // Log.d(TAG, "util freqH freqL cpuON power: " +
      // util + " " + freqH + " " + freqL + " " + cpuON +
      // " " + estimatedCpuEnergy + "mJ");

      // Log.d(TAG, "CPU Energy: " + estimatedCpuEnergy + "mJ");

      freqH = 0;
      freqL = 0;
    }

    return estimatedCpuEnergy;
  }

  private int calculateCpuUtil(int i) {
    return (int) Math.ceil(100 * devProfiler.getPidCpuUsage(i) / devProfiler.getSystemCpuUsage(i));
  }


  private double estimateScreenEnergy(int duration) {
    double estimatedScreenEnergy = 0;

    for (int i = 0; i < duration; i++) {
      estimatedScreenEnergy += betaBrightness * devProfiler.getScreenBrightness(i);
    }

    return estimatedScreenEnergy;
  }


  /**
   * The WiFi interface can be (mainly) in two states: high_state or low_state<br>
   * Transition from low_state to high_state happens when packet_rate > 15<br>
   * packet_rate = (nRxPackets + nTxPackets) / s<br>
   * RChannel: WiFi channel rate<br>
   * Rdata: WiFi data rate<br>
   * 
   * @param duration Duration of method execution
   * @return The estimated energy consumed by the WiFi interface (mJ)
   * 
   */
  private double estimateWiFiEnergy(int duration) {

    if (netProfiler == null) {
      return 0;
    }

    double estimatedWiFiEnergy = 0;
    boolean inHighPowerState = false;
    int nRxPackets, nTxPackets;
    double betaRChannel;
    // double betaWiFiHigh;
    double Rdata;

    for (int i = 0; i < duration; i++) {
      nRxPackets = netProfiler.getWiFiRxPacketRate(i);
      nTxPackets = netProfiler.getWiFiTxPacketRate(i);
      Rdata = (netProfiler.getUplinkDataRate(i) * 8) / 1000000; // Convert from B/s -> b/s -> Mb/s

      // The Wifi interface transits to the high-power state if the packet rate
      // is higher than 15 (according to the paper of PowerTutor)
      // Then the transition to the low-power state is done when the packet rate
      // is lower than 8
      if (!inHighPowerState) {
        if (nRxPackets + nTxPackets > 15) {
          inHighPowerState = true;
          betaRChannel = calculateBetaRChannel();
          betaWiFiHigh = 710 + betaRChannel * Rdata;
          estimatedWiFiEnergy += betaWiFiHigh;
        } else {
          estimatedWiFiEnergy += betaWiFiLow;
        }
      } else {
        if (nRxPackets + nTxPackets < 8) {
          inHighPowerState = false;
          estimatedWiFiEnergy += betaWiFiLow;
        } else {
          betaRChannel = calculateBetaRChannel();
          betaWiFiHigh = 710 + betaRChannel * Rdata;
          estimatedWiFiEnergy += betaWiFiHigh;
        }
      }

      // Log.d(TAG, "nRxPackets nTxPackets: " + nRxPackets + " " + nTxPackets);
      // Log.d(TAG, "Partial Wifi Energy: " + estimatedWiFiEnergy + " mJ");
    }

    return estimatedWiFiEnergy;
  }

  private double calculateBetaRChannel() {
    // The Channel Rate of WiFi connection (Mbps)
    int RChannel = netProfiler.getLinkSpeed();
    return 48 - 0.768 * RChannel;
  }

  /**
   * In the powerTutor paper the states of 3G interface are three: <b>idle, cell_fach</b> and
   * <b>cell_dch</b><br>
   * Transition from <b>idle</b> to <b>cell_fach</b> happens if there are data to send or receive
   * <br>
   * Transition from cell_fach to idle happens if no activity for 4 seconds<br>
   * Transition from cell_fach to cell_dch happens when uplink_buffer > uplink_queue ore d_b > d_q
   * <br>
   * Transition from cell_dch to cell_fach happens if no activity for 6 seconds.
   * 
   * @param duration Duration of method execution
   * @return The estimated energy consumed by the 3G interface (mJ)
   * 
   */
  private double estimate3GEnergy(int duration) {

    if (netProfiler == null) {
      return 0;
    }

    double estimated3GEnergy = 0;

    for (int i = 0; i < duration; i++) {
      switch (netProfiler.get3GActiveState(i)) {
        case NetworkProfiler.THREEG_IN_IDLE_STATE:
          estimated3GEnergy += beta3GIdle;
          break;

        case NetworkProfiler.THREEG_IN_FACH_STATE:
          estimated3GEnergy += beta3GFACH;
          break;

        case NetworkProfiler.THREEG_IN_DCH_STATE:
          estimated3GEnergy += beta3GDCH;
          break;
      }
    }

    return estimated3GEnergy;
  }

  /**
   * @return the totalEstimatedEnergy
   */
  public long getTotalEstimatedEnergy() {
    return totalEstimatedEnergy;
  }

  /**
   * @return the estimatedCpuEnergy
   */
  public double getEstimatedCpuEnergy() {
    return estimatedCpuEnergy;
  }

  /**
   * @return the estimatedScreenEnergy
   */
  public double getEstimatedScreenEnergy() {
    return estimatedScreenEnergy;
  }

  /**
   * @return the estimatedWiFiEnergy
   */
  public double getEstimatedWiFiEnergy() {
    return estimatedWiFiEnergy;
  }

  /**
   * @return the estimated3GEnergy
   */
  public double getEstimated3GEnergy() {
    return estimated3GEnergy;
  }
}
