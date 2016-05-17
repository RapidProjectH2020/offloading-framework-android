/*
 * To change this license header, choose License Headers in Project Properties. To change this
 * template file, choose Tools | Templates and open the template in the editor.
 */
package eu.project.rapid.gvirtusfe;

import java.io.IOException;


/**
 *
 * @author cferraro
 */
public class CudaRt_device {

  public CudaRt_device() {}

  public int cudaGetDeviceCount(Frontend fe, Result res) throws IOException {

    Buffer b = new Buffer();
    b.AddPointer(0);
    String outputbuffer = "";
    fe.Execute("cudaGetDeviceCount", b, res);
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

  public int cudaDeviceCanAccessPeer(Frontend fe, Result res, int device, int peers)
      throws IOException {
    Buffer b = new Buffer();
    b.AddPointer(0);
    b.AddInt(device);
    b.AddInt(peers);
    String outputbuffer = "";
    fe.Execute("cudaDeviceCanAccessPeer", b, res);
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

  public int cudaDriverGetVersion(Frontend fe, Result res) throws IOException {


    Buffer b = new Buffer();
    b.AddPointer(0);
    String outputbuffer = "";
    fe.Execute("cudaDriverGetVersion", b, res);
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

  public int cudaRuntimeGetVersion(Frontend fe, Result res) throws IOException {

    Buffer b = new Buffer();
    b.AddPointer(0);
    String outputbuffer = "";
    fe.Execute("cudaRuntimeGetVersion", b, res);
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


  public int cudaSetDevice(Frontend fe, int device, Result res) throws IOException {

    Buffer b = new Buffer();
    b.Add(device);
    fe.Execute("cudaSetDevice", b, res);
    // fe.ExecuteMultiThread("cudaSetDevice",b,res);
    return 0;
  }

  public String cudaGetErrorString(Frontend fe, int error, Result res) throws IOException {

    Buffer b = new Buffer();
    b.AddInt(error);
    String outbuffer = "";
    StringBuilder output = new StringBuilder();
    fe.Execute("cudaGetErrorString", b, res);
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

  public void cudaDeviceReset(Frontend fe, Result res) throws IOException {
    Buffer b = new Buffer();
    fe.Execute("cudaDeviceReset", b, res);
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

  public cudaDeviceProp cudaGetDeviceProperties(Frontend fe, Result res, int device)
      throws IOException {
    Buffer b = new Buffer();
    String outbuffer = "";
    StringBuilder output = new StringBuilder();
    cudaDeviceProp struct = new cudaDeviceProp();

    b.AddStruct(struct);
    b.AddInt(device);
    fe.Execute("cudaGetDeviceProperties", b, res);
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
    struct.name = output.toString();
    struct.totalGlobalMem = this.getLong(res);
    struct.sharedMemPerBlock = this.getLong(res);
    struct.regsPerBlock = this.getInt(res);
    struct.warpSize = this.getInt(res);
    struct.memPitch = this.getLong(res);
    struct.maxThreadsPerBlock = this.getInt(res);
    struct.maxThreadsDim[0] = this.getInt(res);
    struct.maxThreadsDim[1] = this.getInt(res);
    struct.maxThreadsDim[2] = this.getInt(res);
    struct.maxGridSize[0] = this.getInt(res);
    struct.maxGridSize[1] = this.getInt(res);
    struct.maxGridSize[2] = this.getInt(res);
    struct.clockRate = this.getInt(res); // check
    struct.totalConstMem = this.getLong(res);
    struct.major = this.getInt(res);
    struct.minor = this.getInt(res);
    struct.textureAlignment = this.getLong(res);
    struct.texturePitchAlignment = this.getLong(res); // check
    struct.deviceOverlap = this.getInt(res);
    struct.multiProcessorCount = this.getInt(res);
    struct.kernelExecTimeoutEnabled = this.getInt(res);
    struct.integrated = this.getInt(res);
    struct.canMapHostMemory = this.getInt(res);
    struct.computeMode = this.getInt(res);
    struct.maxTexture1D = this.getInt(res);
    struct.maxTexture1DMipmap = this.getInt(res);
    struct.maxTexture1DLinear = this.getInt(res); // check
    struct.maxTexture2D[0] = this.getInt(res);
    struct.maxTexture2D[1] = this.getInt(res);
    struct.maxTexture2DMipmap[0] = this.getInt(res);
    struct.maxTexture2DMipmap[1] = this.getInt(res);
    struct.maxTexture2DLinear[0] = this.getInt(res);
    struct.maxTexture2DLinear[1] = this.getInt(res);
    struct.maxTexture2DLinear[2] = this.getInt(res);
    struct.maxTexture2DGather[0] = this.getInt(res);
    struct.maxTexture2DGather[1] = this.getInt(res);
    struct.maxTexture3D[0] = this.getInt(res);
    struct.maxTexture3D[1] = this.getInt(res);
    struct.maxTexture3D[2] = this.getInt(res);
    struct.maxTexture3DAlt[0] = this.getInt(res);
    struct.maxTexture3DAlt[1] = this.getInt(res);
    struct.maxTexture3DAlt[2] = this.getInt(res);
    struct.maxTextureCubemap = this.getInt(res);
    struct.maxTexture1DLayered[0] = this.getInt(res);
    struct.maxTexture1DLayered[1] = this.getInt(res);
    struct.maxTexture2DLayered[0] = this.getInt(res);
    struct.maxTexture2DLayered[1] = this.getInt(res);
    struct.maxTexture2DLayered[2] = this.getInt(res);
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
    struct.asyncEngineCount = this.getInt(res);
    struct.unifiedAddressing = this.getInt(res);
    struct.memoryClockRate = this.getInt(res);
    struct.memoryBusWidth = this.getInt(res);
    struct.l2CacheSize = this.getInt(res);
    struct.maxThreadsPerMultiProcessor = this.getInt(res);
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

    String name;
    long totalGlobalMem = 0;
    /** < Global memory available on device in bytes */
    long sharedMemPerBlock = 0;
    /** < Shared memory available per block in bytes */
    int regsPerBlock = 0;
    /** < 32-bit registers available per block */
    int warpSize = 0;
    /** < Warp size in threads */
    long memPitch = 0;
    /** < Maximum pitch in bytes allowed by memory copies */
    int maxThreadsPerBlock = 0;
    /** < Maximum number of threads per block */
    int[] maxThreadsDim = new int[3];
    /** < Maximum size of each dimension of a block */
    int[] maxGridSize = new int[3];
    /** < Maximum size of each dimension of a grid */
    int clockRate = 0;
    /** < Clock frequency in kilohertz */
    long totalConstMem = 0;
    /** < Constant memory available on device in bytes */
    int major = 0;
    /** < Major compute capability */
    int minor = 0;
    /** < Minor compute capability */
    long textureAlignment = 0;
    /** < Alignment requirement for textures */
    long texturePitchAlignment = 0;
    /** < Pitch alignment requirement for texture references bound to pitched memory */
    int deviceOverlap = 0;
    /**
     * < Device can concurrently copy memory and execute a kernel. Deprecated. Use instead
     * asyncEngineCount.
     */
    int multiProcessorCount = 0;
    /** < Number of multiprocessors on device */
    int kernelExecTimeoutEnabled = 0;
    /** < Specified whether there is a run time limit on kernels */
    int integrated = 0;
    /** < Device is integrated as opposed to discrete */
    int canMapHostMemory = 0;
    /** < Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int computeMode = 0;
    /** < Compute mode (See ::cudaComputeMode) */
    int maxTexture1D = 0;
    /** < Maximum 1D texture size */
    int maxTexture1DMipmap = 0;
    /** < Maximum 1D mipmapped texture size */
    int maxTexture1DLinear = 0;
    /** < Maximum size for 1D textures bound to linear memory */
    int[] maxTexture2D = new int[2];
    /** < Maximum 2D texture dimensions */
    int[] maxTexture2DMipmap = new int[2];
    /** < Maximum 2D mipmapped texture dimensions */
    int[] maxTexture2DLinear = new int[3];
    /** < Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int[] maxTexture2DGather = new int[2];
    /** < Maximum 2D texture dimensions if texture gather operations have to be performed */
    int[] maxTexture3D = new int[3];
    /** < Maximum 3D texture dimensions */
    int[] maxTexture3DAlt = new int[3];
    /** < Maximum alternate 3D texture dimensions */
    int maxTextureCubemap = 0;
    /** < Maximum Cubemap texture dimensions */
    int[] maxTexture1DLayered = new int[2];
    /** < Maximum 1D layered texture dimensions */
    int[] maxTexture2DLayered = new int[3];
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
    int asyncEngineCount = 0;
    /** < Number of asynchronous engines */
    int unifiedAddressing = 0;
    /** < Device shares a unified address space with the host */
    int memoryClockRate = 0;
    /** < Peak memory clock frequency in kilohertz */
    int memoryBusWidth = 0;
    /** < Global memory bus width in bits */
    int l2CacheSize = 0;
    /** < Size of L2 cache in bytes */
    int maxThreadsPerMultiProcessor = 0;
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

    public String getName() {
      return name;
    }

    public void setName(String name) {
      this.name = name;
    }

    public long getTotalGlobalMem() {
      return totalGlobalMem;
    }

    public void setTotalGlobalMem(long totalGlobalMem) {
      this.totalGlobalMem = totalGlobalMem;
    }

    public long getSharedMemPerBlock() {
      return sharedMemPerBlock;
    }

    public void setSharedMemPerBlock(long sharedMemPerBlock) {
      this.sharedMemPerBlock = sharedMemPerBlock;
    }

    public int getRegsPerBlock() {
      return regsPerBlock;
    }

    public void setRegsPerBlock(int regsPerBlock) {
      this.regsPerBlock = regsPerBlock;
    }

    public int getWarpSize() {
      return warpSize;
    }

    public void setWarpSize(int warpSize) {
      this.warpSize = warpSize;
    }

    public long getMemPitch() {
      return memPitch;
    }

    public void setMemPitch(long memPitch) {
      this.memPitch = memPitch;
    }

    public int getMaxThreadsPerBlock() {
      return maxThreadsPerBlock;
    }

    public void setMaxThreadsPerBlock(int maxThreadsPerBlock) {
      this.maxThreadsPerBlock = maxThreadsPerBlock;
    }

    public int[] getMaxThreadsDim() {
      return maxThreadsDim;
    }

    public void setMaxThreadsDim(int[] maxThreadsDim) {
      this.maxThreadsDim = maxThreadsDim;
    }

    public int[] getMaxGridSize() {
      return maxGridSize;
    }

    public void setMaxGridSize(int[] maxGridSize) {
      this.maxGridSize = maxGridSize;
    }

    public int getClockRate() {
      return clockRate;
    }

    public void setClockRate(int clockRate) {
      this.clockRate = clockRate;
    }

    public long getTotalConstMem() {
      return totalConstMem;
    }

    public void setTotalConstMem(long totalConstMem) {
      this.totalConstMem = totalConstMem;
    }

    public int getMajor() {
      return major;
    }

    public void setMajor(int major) {
      this.major = major;
    }

    public int getMinor() {
      return minor;
    }

    public void setMinor(int minor) {
      this.minor = minor;
    }

    public long getTextureAlignment() {
      return textureAlignment;
    }

    public void setTextureAlignment(long textureAlignment) {
      this.textureAlignment = textureAlignment;
    }

    public long getTexturePitchAlignment() {
      return texturePitchAlignment;
    }

    public void setTexturePitchAlignment(long texturePitchAlignment) {
      this.texturePitchAlignment = texturePitchAlignment;
    }

    public int getDeviceOverlap() {
      return deviceOverlap;
    }

    public void setDeviceOverlap(int deviceOverlap) {
      this.deviceOverlap = deviceOverlap;
    }

    public int getMultiProcessorCount() {
      return multiProcessorCount;
    }

    public void setMultiProcessorCount(int multiProcessorCount) {
      this.multiProcessorCount = multiProcessorCount;
    }

    public int getKernelExecTimeoutEnabled() {
      return kernelExecTimeoutEnabled;
    }

    public void setKernelExecTimeoutEnabled(int kernelExecTimeoutEnabled) {
      this.kernelExecTimeoutEnabled = kernelExecTimeoutEnabled;
    }

    public int getIntegrated() {
      return integrated;
    }

    public void setIntegrated(int integrated) {
      this.integrated = integrated;
    }

    public int getCanMapHostMemory() {
      return canMapHostMemory;
    }

    public void setCanMapHostMemory(int canMapHostMemory) {
      this.canMapHostMemory = canMapHostMemory;
    }

    public int getComputeMode() {
      return computeMode;
    }

    public void setComputeMode(int computeMode) {
      this.computeMode = computeMode;
    }

    public int getMaxTexture1D() {
      return maxTexture1D;
    }

    public void setMaxTexture1D(int maxTexture1D) {
      this.maxTexture1D = maxTexture1D;
    }

    public int getMaxTexture1DMipmap() {
      return maxTexture1DMipmap;
    }

    public void setMaxTexture1DMipmap(int maxTexture1DMipmap) {
      this.maxTexture1DMipmap = maxTexture1DMipmap;
    }

    public int getMaxTexture1DLinear() {
      return maxTexture1DLinear;
    }

    public void setMaxTexture1DLinear(int maxTexture1DLinear) {
      this.maxTexture1DLinear = maxTexture1DLinear;
    }

    public int[] getMaxTexture2D() {
      return maxTexture2D;
    }

    public void setMaxTexture2D(int[] maxTexture2D) {
      this.maxTexture2D = maxTexture2D;
    }

    public int[] getMaxTexture2DMipmap() {
      return maxTexture2DMipmap;
    }

    public void setMaxTexture2DMipmap(int[] maxTexture2DMipmap) {
      this.maxTexture2DMipmap = maxTexture2DMipmap;
    }

    public int[] getMaxTexture2DLinear() {
      return maxTexture2DLinear;
    }

    public void setMaxTexture2DLinear(int[] maxTexture2DLinear) {
      this.maxTexture2DLinear = maxTexture2DLinear;
    }

    public int[] getMaxTexture2DGather() {
      return maxTexture2DGather;
    }

    public void setMaxTexture2DGather(int[] maxTexture2DGather) {
      this.maxTexture2DGather = maxTexture2DGather;
    }

    public int[] getMaxTexture3D() {
      return maxTexture3D;
    }

    public void setMaxTexture3D(int[] maxTexture3D) {
      this.maxTexture3D = maxTexture3D;
    }

    public int[] getMaxTexture3DAlt() {
      return maxTexture3DAlt;
    }

    public void setMaxTexture3DAlt(int[] maxTexture3DAlt) {
      this.maxTexture3DAlt = maxTexture3DAlt;
    }

    public int getMaxTextureCubemap() {
      return maxTextureCubemap;
    }

    public void setMaxTextureCubemap(int maxTextureCubemap) {
      this.maxTextureCubemap = maxTextureCubemap;
    }

    public int[] getMaxTexture1DLayered() {
      return maxTexture1DLayered;
    }

    public void setMaxTexture1DLayered(int[] maxTexture1DLayered) {
      this.maxTexture1DLayered = maxTexture1DLayered;
    }

    public int[] getMaxTexture2DLayered() {
      return maxTexture2DLayered;
    }

    public void setMaxTexture2DLayered(int[] maxTexture2DLayered) {
      this.maxTexture2DLayered = maxTexture2DLayered;
    }

    public int[] getMaxTextureCubemapLayered() {
      return maxTextureCubemapLayered;
    }

    public void setMaxTextureCubemapLayered(int[] maxTextureCubemapLayered) {
      this.maxTextureCubemapLayered = maxTextureCubemapLayered;
    }

    public int getMaxSurface1D() {
      return maxSurface1D;
    }

    public void setMaxSurface1D(int maxSurface1D) {
      this.maxSurface1D = maxSurface1D;
    }

    public int[] getMaxSurface2D() {
      return maxSurface2D;
    }

    public void setMaxSurface2D(int[] maxSurface2D) {
      this.maxSurface2D = maxSurface2D;
    }

    public int[] getMaxSurface3D() {
      return maxSurface3D;
    }

    public void setMaxSurface3D(int[] maxSurface3D) {
      this.maxSurface3D = maxSurface3D;
    }

    public int[] getMaxSurface1DLayered() {
      return maxSurface1DLayered;
    }

    public void setMaxSurface1DLayered(int[] maxSurface1DLayered) {
      this.maxSurface1DLayered = maxSurface1DLayered;
    }

    public int[] getMaxSurface2DLayered() {
      return maxSurface2DLayered;
    }

    public void setMaxSurface2DLayered(int[] maxSurface2DLayered) {
      this.maxSurface2DLayered = maxSurface2DLayered;
    }

    public int getMaxSurfaceCubemap() {
      return maxSurfaceCubemap;
    }

    public void setMaxSurfaceCubemap(int maxSurfaceCubemap) {
      this.maxSurfaceCubemap = maxSurfaceCubemap;
    }

    public int[] getMaxSurfaceCubemapLayered() {
      return maxSurfaceCubemapLayered;
    }

    public void setMaxSurfaceCubemapLayered(int[] maxSurfaceCubemapLayered) {
      this.maxSurfaceCubemapLayered = maxSurfaceCubemapLayered;
    }

    public long getSurfaceAlignment() {
      return surfaceAlignment;
    }

    public void setSurfaceAlignment(long surfaceAlignment) {
      this.surfaceAlignment = surfaceAlignment;
    }

    public int getConcurrentKernels() {
      return concurrentKernels;
    }

    public void setConcurrentKernels(int concurrentKernels) {
      this.concurrentKernels = concurrentKernels;
    }

    public int getECCEnabled() {
      return ECCEnabled;
    }

    public void setECCEnabled(int ECCEnabled) {
      this.ECCEnabled = ECCEnabled;
    }

    public int getPciBusID() {
      return pciBusID;
    }

    public void setPciBusID(int pciBusID) {
      this.pciBusID = pciBusID;
    }

    public int getPciDeviceID() {
      return pciDeviceID;
    }

    public void setPciDeviceID(int pciDeviceID) {
      this.pciDeviceID = pciDeviceID;
    }

    public int getPciDomainID() {
      return pciDomainID;
    }

    public void setPciDomainID(int pciDomainID) {
      this.pciDomainID = pciDomainID;
    }

    public int getTccDriver() {
      return tccDriver;
    }

    public void setTccDriver(int tccDriver) {
      this.tccDriver = tccDriver;
    }

    public int getAsyncEngineCount() {
      return asyncEngineCount;
    }

    public void setAsyncEngineCount(int asyncEngineCount) {
      this.asyncEngineCount = asyncEngineCount;
    }

    public int getUnifiedAddressing() {
      return unifiedAddressing;
    }

    public void setUnifiedAddressing(int unifiedAddressing) {
      this.unifiedAddressing = unifiedAddressing;
    }

    public int getMemoryClockRate() {
      return memoryClockRate;
    }

    public void setMemoryClockRate(int memoryClockRate) {
      this.memoryClockRate = memoryClockRate;
    }

    public int getMemoryBusWidth() {
      return memoryBusWidth;
    }

    public void setMemoryBusWidth(int memoryBusWidth) {
      this.memoryBusWidth = memoryBusWidth;
    }

    public int getL2CacheSize() {
      return l2CacheSize;
    }

    public void setL2CacheSize(int l2CacheSize) {
      this.l2CacheSize = l2CacheSize;
    }

    public int getMaxThreadsPerMultiProcessor() {
      return maxThreadsPerMultiProcessor;
    }

    public void setMaxThreadsPerMultiProcessor(int maxThreadsPerMultiProcessor) {
      this.maxThreadsPerMultiProcessor = maxThreadsPerMultiProcessor;
    }

    public int getStreamPrioritiesSupported() {
      return streamPrioritiesSupported;
    }

    public void setStreamPrioritiesSupported(int streamPrioritiesSupported) {
      this.streamPrioritiesSupported = streamPrioritiesSupported;
    }

    public int getGlobalL1CacheSupported() {
      return globalL1CacheSupported;
    }

    public void setGlobalL1CacheSupported(int globalL1CacheSupported) {
      this.globalL1CacheSupported = globalL1CacheSupported;
    }

    public int getLocalL1CacheSupported() {
      return localL1CacheSupported;
    }

    public void setLocalL1CacheSupported(int localL1CacheSupported) {
      this.localL1CacheSupported = localL1CacheSupported;
    }

    public long getSharedMemPerMultiprocessor() {
      return sharedMemPerMultiprocessor;
    }

    public void setSharedMemPerMultiprocessor(long sharedMemPerMultiprocessor) {
      this.sharedMemPerMultiprocessor = sharedMemPerMultiprocessor;
    }

    public int getRegsPerMultiprocessor() {
      return regsPerMultiprocessor;
    }

    public void setRegsPerMultiprocessor(int regsPerMultiprocessor) {
      this.regsPerMultiprocessor = regsPerMultiprocessor;
    }

    public int getManagedMemory() {
      return managedMemory;
    }

    public void setManagedMemory(int managedMemory) {
      this.managedMemory = managedMemory;
    }

    public int getIsMultiGpuBoard() {
      return isMultiGpuBoard;
    }

    public void setIsMultiGpuBoard(int isMultiGpuBoard) {
      this.isMultiGpuBoard = isMultiGpuBoard;
    }

    public int getMultiGpuBoardGroupID() {
      return multiGpuBoardGroupID;
    }

    public void setMultiGpuBoardGroupID(int multiGpuBoardGroupID) {
      this.multiGpuBoardGroupID = multiGpuBoardGroupID;
    }



  }



}
