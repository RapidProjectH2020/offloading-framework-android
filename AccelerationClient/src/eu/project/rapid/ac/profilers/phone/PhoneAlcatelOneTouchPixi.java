package eu.project.rapid.ac.profilers.phone;

import eu.project.rapid.ac.profilers.DeviceProfiler;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.ac.profilers.ProgramProfiler;

public class PhoneAlcatelOneTouchPixi extends Phone {

  public PhoneAlcatelOneTouchPixi(DeviceProfiler devProfiler, NetworkProfiler netProfiler,
      ProgramProfiler progProfiler) {
    super(devProfiler, netProfiler, progProfiler);

    // CPU
    MIN_FREQ = 598000; // The minimum frequency (KHz)
    MAX_FREQ = 1000000; // The maximum frequency (KHz)
    betaUh = 0.1232; // (mW/MHz)
    betaUl = 0.094; // (mW/MHz)
    betaCpu = 18.438; // (mW)

    // Screen
    betaBrightness = 0.591; // (mW)

    // WiFi
    betaWiFiLow = 5; // (mW)
    betaWiFiHigh = 155; // (mW)

    // 3G
    beta3GIdle = 10; // (mW)
    beta3GFACH = 510; // (mW)
    beta3GDCH = 672; // (mW)
  }
}
