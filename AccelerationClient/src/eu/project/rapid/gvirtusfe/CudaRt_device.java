/*
 * To change this license header, choose License Headers in Project Properties. To change this
 * template file, choose Tools | Templates and open the template in the editor.
 */
package eu.project.rapid.gvirtusfe;

import java.io.IOException;

import eu.project.rapid.ac.DFE;


/**
 *
 * @author cferraro
 */
public class CudaRt_device {

  private DFE dfe;
  private GVirtusFrontend gvfe;

  public CudaRt_device(DFE dfe) {
    this.dfe = dfe;
    this.gvfe = dfe.getGvirtusFrontend();
  }

  public int cudaGetDeviceCount(Result res) throws IOException {

    Buffer b = new Buffer();
    b.AddPointer(0);
    String outputbuffer = "";
    gvfe.Execute("cudaGetDeviceCount", b, res);
    // fe.ExecuteMultiThread("cudaGetDeviceCount",b,res);
    int sizeType = res.getInput_stream().readByte();
    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    for (int i = 0; i < sizeType; i++) {
      if (i == 0) {
        byte bb = res.getInput_stream().readByte();
        outputbuffer += Integer.toHexString(bb & 0xFF);
      } else
        res.getInput_stream().readByte();
    }
    StringBuilder out2 = new StringBuilder();
    if (outputbuffer.length() > 2) {
      for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
        String str = outputbuffer.substring(i, i + 2);
        out2.insert(0, str);
      }
      outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
    }

    return Integer.valueOf(outputbuffer);
  }

  public int cudaDeviceCanAccessPeer(Result res, int device, int peers) throws IOException {
    Buffer b = new Buffer();
    b.AddPointer(0);
    b.AddInt(device);
    b.AddInt(peers);
    String outputbuffer = "";
    gvfe.Execute("cudaDeviceCanAccessPeer", b, res);
    // fe.ExecuteMultiThread("cudaDeviceCanAccessPeer",b,res);
    int sizeType = res.getInput_stream().readByte();
    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    for (int i = 0; i < sizeType; i++) {
      if (i == 0) {
        byte bb = res.getInput_stream().readByte();
        outputbuffer += Integer.toHexString(bb & 0xFF);
      } else
        res.getInput_stream().readByte();
    }
    StringBuilder out2 = new StringBuilder();
    if (outputbuffer.length() > 2) {
      for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
        String str = outputbuffer.substring(i, i + 2);
        out2.insert(0, str);
      }
      outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
    }
    return Integer.valueOf(outputbuffer);
  }

  public int cudaDriverGetVersion(Result res) throws IOException {


    Buffer b = new Buffer();
    b.AddPointer(0);
    String outputbuffer = "";
    gvfe.Execute("cudaDriverGetVersion", b, res);
    // fe.ExecuteMultiThread("cudaDriverGetVersion",b,res);
    int sizeType = res.getInput_stream().readByte();
    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    for (int i = 0; i < sizeType; i++) {
      if (i == 0 || i == 1) {
        byte bb = res.getInput_stream().readByte();
        outputbuffer += Integer.toHexString(bb & 0xFF);
      } else
        res.getInput_stream().readByte();
    }

    StringBuilder out2 = new StringBuilder();
    if (outputbuffer.length() > 2) {
      for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
        String str = outputbuffer.substring(i, i + 2);
        out2.insert(0, str);
      }
      outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
    }
    return Integer.valueOf(outputbuffer);
  }

  public int cudaRuntimeGetVersion(Result res) throws IOException {

    Buffer b = new Buffer();
    b.AddPointer(0);
    String outputbuffer = "";
    gvfe.Execute("cudaRuntimeGetVersion", b, res);
    // fe.ExecuteMultiThread("cudaRuntimeGetVersion",b,res);
    int sizeType = res.getInput_stream().readByte();
    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    for (int i = 0; i < sizeType; i++) {
      if (i == 0 || i == 1) {
        byte bb = res.getInput_stream().readByte();
        outputbuffer += Integer.toHexString(bb & 0xFF);
      } else
        res.getInput_stream().readByte();
    }
    StringBuilder out2 = new StringBuilder();
    if (outputbuffer.length() > 2) {
      for (int i = 0; i < outputbuffer.length() - 1; i += 2) {
        String str = outputbuffer.substring(i, i + 2);
        out2.insert(0, str);
      }
      outputbuffer = String.valueOf(Integer.parseInt(out2.toString(), 16));
    }
    return Integer.valueOf(outputbuffer);
  }


  public int cudaSetDevice(int device, Result res) throws IOException {

    Buffer b = new Buffer();
    b.Add(device);
    gvfe.Execute("cudaSetDevice", b, res);
    // fe.ExecuteMultiThread("cudaSetDevice",b,res);
    return 0;
  }

  public String cudaGetErrorString(int error, Result res) throws IOException {

    Buffer b = new Buffer();
    b.AddInt(error);
    String outbuffer = "";
    StringBuilder output = new StringBuilder();
    gvfe.Execute("cudaGetErrorString", b, res);
    int sizeType = res.getInput_stream().readByte();
    // System.out.print("sizeType " + sizeType);

    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();
    res.getInput_stream().readByte();
    // System.out.print("sizeType " + sizeType);

    for (int i = 0; i < 7; i++)
      res.getInput_stream().readByte();


    for (int i = 0; i < sizeType; i++) {
      byte bit = res.getInput_stream().readByte();
      outbuffer += Integer.toHexString(bit);
      // System.out.print(outbuffer.toString());
    }
    for (int i = 0; i < outbuffer.length() - 1; i += 2) {
      String str = outbuffer.substring(i, i + 2);
      output.append((char) Integer.parseInt(str, 16));

    }
    return output.toString();

  }

  public void cudaDeviceReset(Result res) throws IOException {
    Buffer b = new Buffer();
    gvfe.Execute("cudaDeviceReset", b, res);
    // fe.ExecuteMultiThread("cudaDeviceReset", b, res);
  }


  private int getInt(Result res) throws IOException {

    StringBuilder output = new StringBuilder();
    for (int i = 0; i < 4; i++) {
      byte bit = res.getInput_stream().readByte();
      int a = bit & 0xFF;
      if (a == 0) {
        output.insert(0, Integer.toHexString(a));
        output.insert(0, Integer.toHexString(a));
      } else {
        output.insert(0, Integer.toHexString(a));
      }
    }
    return Integer.parseInt(output.toString(), 16);

  }

  private long getLong(Result res) throws IOException {

    StringBuilder output = new StringBuilder();
    for (int i = 0; i < 8; i++) {
      byte bit = res.getInput_stream().readByte();
      int a = bit & 0xFF;
      if (a == 0) {
        output.insert(0, Integer.toHexString(a));
        output.insert(0, Integer.toHexString(a));
      } else {
        output.insert(0, Integer.toHexString(a));
      }
    }
    return Long.parseLong(output.toString(), 16);
  }

  public cudaDeviceProp cudaGetDeviceProperties(Result res, int device) throws IOException {
    Buffer b = new Buffer();
    String outbuffer = "";
    StringBuilder output = new StringBuilder();
    cudaDeviceProp struct = new cudaDeviceProp();

    b.AddStruct(struct);
    b.AddInt(device);
    gvfe.Execute("cudaGetDeviceProperties", b, res);
    // fe.ExecuteMultiThread("cudaGetDeviceProperties", b,res);
    int sizeType = 640;
    for (int i = 0; i < 8; i++) {
      res.getInput_stream().readByte();
    } // lettura size vettore buffer
    // lettura nome device
    for (int i = 0; i < 256; i++) {
      byte bit = res.getInput_stream().readByte();

      outbuffer += Integer.toHexString(bit);
    }
    for (int i = 0; i < outbuffer.length() - 1; i += 2) {
      String str = outbuffer.substring(i, i + 2);
      output.append((char) Integer.parseInt(str, 16));
    }
    struct.setName(output.toString());
    struct.setTotalGlobalMem(this.getLong(res));
    struct.setSharedMemPerBlock(this.getLong(res));
    struct.setRegsPerBlock(this.getInt(res));
    struct.setWarpSize(this.getInt(res));
    struct.setMemPitch(this.getLong(res));
    struct.setMaxThreadsPerBlock(this.getInt(res));
    struct.getMaxThreadsDim()[0] = this.getInt(res);
    struct.getMaxThreadsDim()[1] = this.getInt(res);
    struct.getMaxThreadsDim()[2] = this.getInt(res);
    struct.getMaxGridSize()[0] = this.getInt(res);
    struct.getMaxGridSize()[1] = this.getInt(res);
    struct.getMaxGridSize()[2] = this.getInt(res);
    struct.setClockRate(this.getInt(res)); // check
    struct.setTotalConstMem(this.getLong(res));
    struct.setMajor(this.getInt(res));
    struct.setMinor(this.getInt(res));
    struct.setTextureAlignment(this.getLong(res));
    struct.texturePitchAlignment = this.getLong(res); // check
    struct.setDeviceOverlap(this.getInt(res));
    struct.multiProcessorCount = this.getInt(res);
    struct.setKernelExecTimeoutEnabled(this.getInt(res));
    struct.integrated = this.getInt(res);
    struct.canMapHostMemory = this.getInt(res);
    struct.computeMode = this.getInt(res);
    struct.setMaxTexture1D(this.getInt(res));
    struct.maxTexture1DMipmap = this.getInt(res);
    struct.maxTexture1DLinear = this.getInt(res); // check
    struct.getMaxTexture2D()[0] = this.getInt(res);
    struct.getMaxTexture2D()[1] = this.getInt(res);
    struct.maxTexture2DMipmap[0] = this.getInt(res);
    struct.maxTexture2DMipmap[1] = this.getInt(res);
    struct.maxTexture2DLinear[0] = this.getInt(res);
    struct.maxTexture2DLinear[1] = this.getInt(res);
    struct.maxTexture2DLinear[2] = this.getInt(res);
    struct.maxTexture2DGather[0] = this.getInt(res);
    struct.maxTexture2DGather[1] = this.getInt(res);
    struct.getMaxTexture3D()[0] = this.getInt(res);
    struct.getMaxTexture3D()[1] = this.getInt(res);
    struct.getMaxTexture3D()[2] = this.getInt(res);
    struct.maxTexture3DAlt[0] = this.getInt(res);
    struct.maxTexture3DAlt[1] = this.getInt(res);
    struct.maxTexture3DAlt[2] = this.getInt(res);
    struct.maxTextureCubemap = this.getInt(res);
    struct.getMaxTexture1DLayered()[0] = this.getInt(res);
    struct.getMaxTexture1DLayered()[1] = this.getInt(res);
    struct.getMaxTexture2DLayered()[0] = this.getInt(res);
    struct.getMaxTexture2DLayered()[1] = this.getInt(res);
    struct.getMaxTexture2DLayered()[2] = this.getInt(res);
    struct.maxTextureCubemapLayered[0] = this.getInt(res);
    struct.maxTextureCubemapLayered[1] = this.getInt(res);
    struct.maxSurface1D = this.getInt(res);
    struct.maxSurface2D[0] = this.getInt(res);
    struct.maxSurface2D[1] = this.getInt(res);
    struct.maxSurface3D[0] = this.getInt(res);
    struct.maxSurface3D[1] = this.getInt(res);
    struct.maxSurface3D[2] = this.getInt(res);
    struct.maxSurface1DLayered[0] = this.getInt(res);
    struct.maxSurface1DLayered[1] = this.getInt(res);
    struct.maxSurface2DLayered[0] = this.getInt(res);
    struct.maxSurface2DLayered[1] = this.getInt(res);
    struct.maxSurface2DLayered[2] = this.getInt(res);
    struct.maxSurfaceCubemap = this.getInt(res);
    struct.maxSurfaceCubemapLayered[0] = this.getInt(res);
    struct.maxSurfaceCubemapLayered[1] = this.getInt(res);
    struct.surfaceAlignment = this.getLong(res);
    struct.concurrentKernels = this.getInt(res);
    struct.ECCEnabled = this.getInt(res);
    struct.pciBusID = this.getInt(res);
    struct.pciDeviceID = this.getInt(res);
    struct.pciDomainID = this.getInt(res);
    struct.tccDriver = this.getInt(res);
    struct.setAsyncEngineCount(this.getInt(res));
    struct.unifiedAddressing = this.getInt(res);
    struct.setMemoryClockRate(this.getInt(res));
    struct.setMemoryBusWidth(this.getInt(res));
    struct.setL2CacheSize(this.getInt(res));
    struct.setMaxThreadsPerMultiProcessor(this.getInt(res));
    struct.streamPrioritiesSupported = this.getInt(res);
    struct.globalL1CacheSupported = this.getInt(res);
    struct.localL1CacheSupported = this.getInt(res);
    struct.sharedMemPerMultiprocessor = this.getLong(res);
    struct.regsPerMultiprocessor = this.getInt(res);
    struct.managedMemory = this.getInt(res);
    struct.isMultiGpuBoard = this.getInt(res);
    struct.multiGpuBoardGroupID = this.getInt(res);
    this.getInt(res); // è in più da capire il perchè
    return struct;
  }

  public class cudaDeviceProp {

    private String name;
    private long totalGlobalMem = 0;
    /** < Global memory available on device in bytes */
    private long sharedMemPerBlock = 0;
    /** < Shared memory available per block in bytes */
    private int regsPerBlock = 0;
    /** < 32-bit registers available per block */
    private int warpSize = 0;
    /** < Warp size in threads */
    private long memPitch = 0;
    /** < Maximum pitch in bytes allowed by memory copies */
    private int maxThreadsPerBlock = 0;
    /** < Maximum number of threads per block */
    private int[] maxThreadsDim = new int[3];
    /** < Maximum size of each dimension of a block */
    private int[] maxGridSize = new int[3];
    /** < Maximum size of each dimension of a grid */
    private int clockRate = 0;
    /** < Clock frequency in kilohertz */
    private long totalConstMem = 0;
    /** < Constant memory available on device in bytes */
    private int major = 0;
    /** < Major compute capability */
    private int minor = 0;
    /** < Minor compute capability */
    private long textureAlignment = 0;
    /** < Alignment requirement for textures */
    long texturePitchAlignment = 0;
    /** < Pitch alignment requirement for texture references bound to pitched memory */
    private int deviceOverlap = 0;
    /**
     * < Device can concurrently copy memory and execute a kernel. Deprecated. Use instead
     * asyncEngineCount.
     */
    int multiProcessorCount = 0;
    /** < Number of multiprocessors on device */
    private int kernelExecTimeoutEnabled = 0;
    /** < Specified whether there is a run time limit on kernels */
    int integrated = 0;
    /** < Device is integrated as opposed to discrete */
    int canMapHostMemory = 0;
    /** < Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int computeMode = 0;
    /** < Compute mode (See ::cudaComputeMode) */
    private int maxTexture1D = 0;
    /** < Maximum 1D texture size */
    int maxTexture1DMipmap = 0;
    /** < Maximum 1D mipmapped texture size */
    int maxTexture1DLinear = 0;
    /** < Maximum size for 1D textures bound to linear memory */
    private int[] maxTexture2D = new int[2];
    /** < Maximum 2D texture dimensions */
    int[] maxTexture2DMipmap = new int[2];
    /** < Maximum 2D mipmapped texture dimensions */
    int[] maxTexture2DLinear = new int[3];
    /** < Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int[] maxTexture2DGather = new int[2];
    /** < Maximum 2D texture dimensions if texture gather operations have to be performed */
    private int[] maxTexture3D = new int[3];
    /** < Maximum 3D texture dimensions */
    int[] maxTexture3DAlt = new int[3];
    /** < Maximum alternate 3D texture dimensions */
    int maxTextureCubemap = 0;
    /** < Maximum Cubemap texture dimensions */
    private int[] maxTexture1DLayered = new int[2];
    /** < Maximum 1D layered texture dimensions */
    private int[] maxTexture2DLayered = new int[3];
    /** < Maximum 2D layered texture dimensions */
    int[] maxTextureCubemapLayered = new int[2];
    /** < Maximum Cubemap layered texture dimensions */
    int maxSurface1D = 0;
    /** < Maximum 1D surface size */
    int[] maxSurface2D = new int[2];
    /** < Maximum 2D surface dimensions */
    int[] maxSurface3D = new int[3];
    /** < Maximum 3D surface dimensions */
    int[] maxSurface1DLayered = new int[2];
    /** < Maximum 1D layered surface dimensions */
    int[] maxSurface2DLayered = new int[3];
    /** < Maximum 2D layered surface dimensions */
    int maxSurfaceCubemap = 0;
    /** < Maximum Cubemap surface dimensions */
    int[] maxSurfaceCubemapLayered = new int[2];
    /** < Maximum Cubemap layered surface dimensions */
    long surfaceAlignment = 0;
    /** < Alignment requirements for surfaces */
    int concurrentKernels = 0;
    /** < Device can possibly execute multiple kernels concurrently */
    int ECCEnabled = 0;
    /** < Device has ECC support enabled */
    int pciBusID = 0;
    /** < PCI bus ID of the device */
    int pciDeviceID = 0;
    /** < PCI device ID of the device */
    int pciDomainID = 0;
    /** < PCI domain ID of the device */
    int tccDriver = 0;
    /** < 1 if device is a Tesla device using TCC driver, 0 otherwise */
    private int asyncEngineCount = 0;
    /** < Number of asynchronous engines */
    int unifiedAddressing = 0;
    /** < Device shares a unified address space with the host */
    private int memoryClockRate = 0;
    /** < Peak memory clock frequency in kilohertz */
    private int memoryBusWidth = 0;
    /** < Global memory bus width in bits */
    private int l2CacheSize = 0;
    /** < Size of L2 cache in bytes */
    private int maxThreadsPerMultiProcessor = 0;
    /** < Maximum resident threads per multiprocessor */
    int streamPrioritiesSupported = 0;
    /** < Device supports stream priorities */
    int globalL1CacheSupported = 0;
    /** < Device supports caching globals in L1 */
    int localL1CacheSupported = 0;
    /** < Device supports caching locals in L1 */
    long sharedMemPerMultiprocessor = 0;
    /** < Shared memory available per multiprocessor in bytes */
    int regsPerMultiprocessor = 0;
    /** < 32-bit registers available per multiprocessor */
    int managedMemory = 0;
    /** < Device supports allocating managed memory on this system */
    int isMultiGpuBoard = 0;
    /** < Device is on a multi-GPU board */
    int multiGpuBoardGroupID = 0;

    /** < Unique identifier for a group of devices on the same multi-GPU board */

    public cudaDeviceProp() {}

    /**
     * @return the major
     */
    public int getMajor() {
      return major;
    }

    /**
     * @param major the major to set
     */
    public void setMajor(int major) {
      this.major = major;
    }

    /**
     * @return the minor
     */
    public int getMinor() {
      return minor;
    }

    /**
     * @param minor the minor to set
     */
    public void setMinor(int minor) {
      this.minor = minor;
    }

    /**
     * @return the totalGlobalMem
     */
    public long getTotalGlobalMem() {
      return totalGlobalMem;
    }

    /**
     * @param totalGlobalMem the totalGlobalMem to set
     */
    public void setTotalGlobalMem(long totalGlobalMem) {
      this.totalGlobalMem = totalGlobalMem;
    }

    /**
     * @return the clockRate
     */
    public int getClockRate() {
      return clockRate;
    }

    /**
     * @param clockRate the clockRate to set
     */
    public void setClockRate(int clockRate) {
      this.clockRate = clockRate;
    }

    /**
     * @return the memoryClockRate
     */
    public int getMemoryClockRate() {
      return memoryClockRate;
    }

    /**
     * @param memoryClockRate the memoryClockRate to set
     */
    public void setMemoryClockRate(int memoryClockRate) {
      this.memoryClockRate = memoryClockRate;
    }

    /**
     * @return the memoryBusWidth
     */
    public int getMemoryBusWidth() {
      return memoryBusWidth;
    }

    /**
     * @param memoryBusWidth the memoryBusWidth to set
     */
    public void setMemoryBusWidth(int memoryBusWidth) {
      this.memoryBusWidth = memoryBusWidth;
    }

    /**
     * @return the l2CacheSize
     */
    public int getL2CacheSize() {
      return l2CacheSize;
    }

    /**
     * @param l2CacheSize the l2CacheSize to set
     */
    public void setL2CacheSize(int l2CacheSize) {
      this.l2CacheSize = l2CacheSize;
    }

    /**
     * @return the name
     */
    public String getName() {
      return name;
    }

    /**
     * @param name the name to set
     */
    public void setName(String name) {
      this.name = name;
    }

    /**
     * @return the maxTexture1D
     */
    public int getMaxTexture1D() {
      return maxTexture1D;
    }

    /**
     * @param maxTexture1D the maxTexture1D to set
     */
    public void setMaxTexture1D(int maxTexture1D) {
      this.maxTexture1D = maxTexture1D;
    }

    /**
     * @return the maxTexture2D
     */
    public int[] getMaxTexture2D() {
      return maxTexture2D;
    }

    /**
     * @param maxTexture2D the maxTexture2D to set
     */
    public void setMaxTexture2D(int[] maxTexture2D) {
      this.maxTexture2D = maxTexture2D;
    }

    /**
     * @return the maxTexture3D
     */
    public int[] getMaxTexture3D() {
      return maxTexture3D;
    }

    /**
     * @param maxTexture3D the maxTexture3D to set
     */
    public void setMaxTexture3D(int[] maxTexture3D) {
      this.maxTexture3D = maxTexture3D;
    }

    /**
     * @return the maxTexture1DLayered
     */
    public int[] getMaxTexture1DLayered() {
      return maxTexture1DLayered;
    }

    /**
     * @param maxTexture1DLayered the maxTexture1DLayered to set
     */
    public void setMaxTexture1DLayered(int[] maxTexture1DLayered) {
      this.maxTexture1DLayered = maxTexture1DLayered;
    }

    /**
     * @return the maxTexture2DLayered
     */
    public int[] getMaxTexture2DLayered() {
      return maxTexture2DLayered;
    }

    /**
     * @param maxTexture2DLayered the maxTexture2DLayered to set
     */
    public void setMaxTexture2DLayered(int[] maxTexture2DLayered) {
      this.maxTexture2DLayered = maxTexture2DLayered;
    }

    /**
     * @return the totalConstMem
     */
    public long getTotalConstMem() {
      return totalConstMem;
    }

    /**
     * @param totalConstMem the totalConstMem to set
     */
    public void setTotalConstMem(long totalConstMem) {
      this.totalConstMem = totalConstMem;
    }

    /**
     * @return the sharedMemPerBlock
     */
    public long getSharedMemPerBlock() {
      return sharedMemPerBlock;
    }

    /**
     * @param sharedMemPerBlock the sharedMemPerBlock to set
     */
    public void setSharedMemPerBlock(long sharedMemPerBlock) {
      this.sharedMemPerBlock = sharedMemPerBlock;
    }

    /**
     * @return the regsPerBlock
     */
    public int getRegsPerBlock() {
      return regsPerBlock;
    }

    /**
     * @param regsPerBlock the regsPerBlock to set
     */
    public void setRegsPerBlock(int regsPerBlock) {
      this.regsPerBlock = regsPerBlock;
    }

    /**
     * @return the warpSize
     */
    public int getWarpSize() {
      return warpSize;
    }

    /**
     * @param warpSize the warpSize to set
     */
    public void setWarpSize(int warpSize) {
      this.warpSize = warpSize;
    }

    /**
     * @return the maxThreadsPerMultiProcessor
     */
    public int getMaxThreadsPerMultiProcessor() {
      return maxThreadsPerMultiProcessor;
    }

    /**
     * @param maxThreadsPerMultiProcessor the maxThreadsPerMultiProcessor to set
     */
    public void setMaxThreadsPerMultiProcessor(int maxThreadsPerMultiProcessor) {
      this.maxThreadsPerMultiProcessor = maxThreadsPerMultiProcessor;
    }

    /**
     * @return the maxThreadsPerBlock
     */
    public int getMaxThreadsPerBlock() {
      return maxThreadsPerBlock;
    }

    /**
     * @param maxThreadsPerBlock the maxThreadsPerBlock to set
     */
    public void setMaxThreadsPerBlock(int maxThreadsPerBlock) {
      this.maxThreadsPerBlock = maxThreadsPerBlock;
    }

    /**
     * @return the maxThreadsDim
     */
    public int[] getMaxThreadsDim() {
      return maxThreadsDim;
    }

    /**
     * @param maxThreadsDim the maxThreadsDim to set
     */
    public void setMaxThreadsDim(int[] maxThreadsDim) {
      this.maxThreadsDim = maxThreadsDim;
    }

    /**
     * @return the maxGridSize
     */
    public int[] getMaxGridSize() {
      return maxGridSize;
    }

    /**
     * @param maxGridSize the maxGridSize to set
     */
    public void setMaxGridSize(int[] maxGridSize) {
      this.maxGridSize = maxGridSize;
    }

    /**
     * @return the memPitch
     */
    public long getMemPitch() {
      return memPitch;
    }

    /**
     * @param memPitch the memPitch to set
     */
    public void setMemPitch(long memPitch) {
      this.memPitch = memPitch;
    }

    /**
     * @return the textureAlignment
     */
    public long getTextureAlignment() {
      return textureAlignment;
    }

    /**
     * @param textureAlignment the textureAlignment to set
     */
    public void setTextureAlignment(long textureAlignment) {
      this.textureAlignment = textureAlignment;
    }

    /**
     * @return the deviceOverlap
     */
    public int getDeviceOverlap() {
      return deviceOverlap;
    }

    /**
     * @param deviceOverlap the deviceOverlap to set
     */
    public void setDeviceOverlap(int deviceOverlap) {
      this.deviceOverlap = deviceOverlap;
    }

    /**
     * @return the asyncEngineCount
     */
    public int getAsyncEngineCount() {
      return asyncEngineCount;
    }

    /**
     * @param asyncEngineCount the asyncEngineCount to set
     */
    public void setAsyncEngineCount(int asyncEngineCount) {
      this.asyncEngineCount = asyncEngineCount;
    }

    /**
     * @return the kernelExecTimeoutEnabled
     */
    public int getKernelExecTimeoutEnabled() {
      return kernelExecTimeoutEnabled;
    }

    /**
     * @param kernelExecTimeoutEnabled the kernelExecTimeoutEnabled to set
     */
    public void setKernelExecTimeoutEnabled(int kernelExecTimeoutEnabled) {
      this.kernelExecTimeoutEnabled = kernelExecTimeoutEnabled;
    }
  }
}
