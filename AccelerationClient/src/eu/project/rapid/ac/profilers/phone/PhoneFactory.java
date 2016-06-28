package eu.project.rapid.ac.profilers.phone;

import eu.project.rapid.ac.profilers.DeviceProfiler;
import eu.project.rapid.ac.profilers.NetworkProfiler;
import eu.project.rapid.ac.profilers.ProgramProfiler;
import eu.project.rapid.ac.utils.Constants;

public class PhoneFactory {

  private PhoneFactory() {};

  public static Phone getPhone(DeviceProfiler devProfiler, NetworkProfiler netProfiler,
      ProgramProfiler progProfiler) {
    if (android.os.Build.MODEL.equals(Constants.PHONE_NAME_HTC_G1)) {
      return new PhoneHtcDream(devProfiler, netProfiler, progProfiler);
    }
    // Add here all the cases for the other phones.
    else {
      // By default, if the current phone's call is not implemented yet, use the htc dream.
      return new PhoneHtcDream(devProfiler, netProfiler, progProfiler);
    }
  }
}
