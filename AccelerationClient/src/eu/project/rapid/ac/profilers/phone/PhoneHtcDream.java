package eu.project.rapid.ac.profilers.phone;

import eu.project.rapid.ac.profilers.DeviceProfiler;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.ac.profilers.ProgramProfiler;

public class PhoneHtcDream extends Phone {

  public PhoneHtcDream(DeviceProfiler devProfiler, NetworkProfiler netProfiler,
      ProgramProfiler progProfiler) {
    super(devProfiler, netProfiler, progProfiler);

    // CPU
    MIN_FREQ = 245760; // The minimum frequency for HTC Dream CPU
    MAX_FREQ = 352000; // The maximum frequency for HTC Dream CPU
    betaUh = 4.34;
    betaUl = 3.42;
    betaCpu = 121.46;

    // Screen
    betaBrightness = 2.4;

    // WiFi
    betaWiFiLow = 20;

    // 3G
    beta3GIdle = 10;
    beta3GFACH = 401;
    beta3GDCH = 570;
  }
}
