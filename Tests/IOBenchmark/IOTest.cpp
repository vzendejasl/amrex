// -------------------------------------------------------------
// IOTest.cpp
// -------------------------------------------------------------
#include <Array.H>
#include <IntVect.H>
#include <Box.H>
#include <BoxArray.H>
#include <FArrayBox.H>
#include <MultiFab.H>
#include <VisMF.H>
#include <ParallelDescriptor.H>
#include <Utility.H>
#include <NFiles.H>

#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <cerrno>

#include <unistd.h>
#include <string.h>
#include <sys/stat.h>

using std::cout;
using std::endl;
using std::ends;
using std::ofstream;
using std::streamoff;

const int XDIR(0);
const int YDIR(1);
const int ZDIR(2);
Real bytesPerMB(1.0e+06);
const bool verboseDir(true);


// -------------------------------------------------------------
void DirectoryTests() {
    int ndirs(256), nlevels(4);

    if(ParallelDescriptor::IOProcessor()) { 
      errno = 0;
      mkdir("testdir", 0755);
      std::cout << "_here 0:  errno = " << strerror(errno) << std::endl;
      errno = 0;
      rmdir("testdir");
      std::cout << "_here 1:  errno = " << strerror(errno) << std::endl;
      errno = 0;
      mkdir("testnest/n0/n1", 0755);
      std::cout << "_here 2:  errno = " << strerror(errno) << std::endl;
      errno = 0;
    }

    BL_PROFILE_VAR("mkdirs", mkdirs);
    for(int i(0); i < ndirs; ++i) {
      std::stringstream dirname;
      dirname << "dir" << i;
      if(ParallelDescriptor::IOProcessor()) {
        if( ! BoxLib::UtilCreateDirectory(dirname.str(), 0755, verboseDir)) {
          BoxLib::CreateDirectoryFailed(dirname.str());
        }
        for(int level(0); level < nlevels; ++level) {
          std::stringstream dirname;
          dirname << "dir" << i << "/Level_" << level;
          if( ! BoxLib::UtilCreateDirectory(dirname.str(), 0755, verboseDir)) {
            BoxLib::CreateDirectoryFailed(dirname.str());
          }
        }
      }
    }
    ParallelDescriptor::Barrier("waitfordir");
    BL_PROFILE_VAR_STOP(mkdirs);

    BL_PROFILE_VAR("renamedirs", renamedirs);
    for(int i(0); i < ndirs; ++i) {
      if(ParallelDescriptor::IOProcessor()) {
        std::stringstream dirname;
        dirname << "dir" << i;
        std::string newdirname;
        newdirname = dirname.str() + ".old";
        std::rename(dirname.str().c_str(), newdirname.c_str());
      }
    }
    ParallelDescriptor::Barrier("renamedirs");
    BL_PROFILE_VAR_STOP(renamedirs);
}


// -------------------------------------------------------------
void NFileTests(int nOutFiles, const std::string &filePrefix) {
  int myProc(ParallelDescriptor::MyProc());
  Array<int> data(32);

  for(int i(0); i < data.size(); ++i) {
    data[i] = (100 * myProc) + i;
  }

  bool groupSets(false), setBuf(true);
  for(NFilesIter nfi(nOutFiles, filePrefix, groupSets, setBuf); nfi.ReadyToWrite(); ++nfi) {
    nfi.Stream().write((const char *) data.dataPtr(), data.size() * sizeof(int));
  }
}



// -------------------------------------------------------------
void FileTests() {
  Array<int> myInts(4096 * 4096);
  for(int i(0); i < myInts.size(); ++i) {
    myInts[i] = i;
  }

  std::fstream myFile;

  BL_PROFILE_VAR("makeafile", makeafile);
  myFile.open("myFile", std::ios::out|std::ios::trunc|std::ios::binary);
  myFile.write((const char *) myInts.dataPtr(), myInts.size() * sizeof(int));
  myFile.close();
  BL_PROFILE_VAR_STOP(makeafile);

  BL_PROFILE_VAR_NS("seektests", seektests);
  myFile.open("myFile", std::ios::in|std::ios::binary);
  myFile.seekg(0, std::ios::end);
  myFile.seekg(0, std::ios::beg);
  for(int i(0); i < myInts.size()/10; ++i) {
    BL_PROFILE_VAR_START(seektests);
    myFile.seekg(1, std::ios::cur);
    BL_PROFILE_VAR_STOP(seektests);
  }
  myFile.close();


  std::string dirname("/home/vince/Development/BoxLib/Tests/IOBenchmark/a/b/c/d");
  if(ParallelDescriptor::IOProcessor()) {
    if( ! BoxLib::UtilCreateDirectory(dirname, 0755, verboseDir)) {
      BoxLib::CreateDirectoryFailed(dirname);
    }
  }
  std::string rdirname("relative/e/f/g");
  if(ParallelDescriptor::IOProcessor()) {
    if( ! BoxLib::UtilCreateDirectory(rdirname, 0755, verboseDir)) {
      BoxLib::CreateDirectoryFailed(rdirname);
    }
  }
  std::string nsdirname("noslash");
  if(ParallelDescriptor::IOProcessor()) {
    if( ! BoxLib::UtilCreateDirectory(nsdirname, 0755, verboseDir)) {
      BoxLib::CreateDirectoryFailed(nsdirname);
    }
  }

}


// -------------------------------------------------------------
BoxArray MakeBoxArray(int maxgrid,  int nboxes) {
#if (BL_SPACEDIM == 2)
  IntVect ivlo(0, 0);
  IntVect ivhi(maxgrid - 1, maxgrid - 1);
#else
  IntVect ivlo(0, 0, 0);
  IntVect ivhi(maxgrid - 1, maxgrid - 1, maxgrid - 1);
#endif
  int iSide(pow(static_cast<Real>(nboxes), 1.0/3.0));
  Box tempBox(ivlo, ivhi);
  BoxArray bArray(nboxes);
  int ix(0), iy(0), iz(0);
  for(int ibox(0); ibox < nboxes; ++ibox) {
    Box sBox(tempBox);
    sBox.shift(XDIR, ix * maxgrid);
    sBox.shift(YDIR, iy * maxgrid);
#if (BL_SPACEDIM == 3)
    sBox.shift(ZDIR, iz * maxgrid);
#endif
    bArray.set(ibox, sBox);
    ++ix;
    if(ix > iSide) {
      ix = 0;
      ++iy;
    }
    if(iy > iSide) {
      iy = 0;
      ++iz;
    }
  }
  return bArray;
}


// -------------------------------------------------------------
void TestWriteNFiles(int nfiles, int maxgrid, int ncomps, int nboxes,
                     bool raninit, bool mb2)
{
  VisMF::SetNOutFiles(nfiles);
  if(mb2) {
    bytesPerMB = pow(2.0, 20);
  }

  BoxArray bArray(MakeBoxArray(maxgrid, nboxes));
  if(ParallelDescriptor::IOProcessor()) {
    cout << "  Timings for writing to " << nfiles << " files:" << endl;
  }

  // make a MultiFab
  MultiFab mfout(bArray, ncomps, 0);
  for(MFIter mfiset(mfout); mfiset.isValid(); ++mfiset) {
    for(int invar(0); invar < ncomps; ++invar) {
      if(raninit) {
        Real *dp = mfout[mfiset].dataPtr(invar);
	for(int i(0); i < mfout[mfiset].box().numPts(); ++i) {
	  dp[i] = BoxLib::Random() + (1.0 + static_cast<Real> (invar));
	}
      } else {
        mfout[mfiset].setVal((100.0 * mfiset.index()) + invar, invar);
      }
    }
  }

  long npts(bArray[0].numPts());
  long totalNBytes(npts * ncomps * nboxes *sizeof(Real));
  std::string mfName("TestMF");

  VisMF::RemoveFiles(mfName, true);

  ParallelDescriptor::Barrier();
  double wallTimeStart(ParallelDescriptor::second());

  VisMF::Write(mfout, mfName); 

  double wallTime(ParallelDescriptor::second() - wallTimeStart);

  double wallTimeMax(wallTime);
  double wallTimeMin(wallTime);

  ParallelDescriptor::ReduceRealMin(wallTimeMin);
  ParallelDescriptor::ReduceRealMax(wallTimeMax);
  Real megabytes((static_cast<Real> (totalNBytes)) / bytesPerMB);

  if(ParallelDescriptor::IOProcessor()) {
    cout << std::setprecision(5);
    cout << "------------------------------------------" << endl;
    cout << "  Total megabytes = " << megabytes << endl;
    cout << "  Write:  Megabytes/sec   = " << megabytes/wallTimeMax << endl;
    cout << "  Wall clock time = " << wallTimeMax << endl;
    cout << "  Min wall clock time = " << wallTimeMin << endl;
    cout << "  Max wall clock time = " << wallTimeMax << endl;
    cout << "------------------------------------------" << endl;
  }
}


// -------------------------------------------------------------
void TestWriteNFilesNoFabHeader(int nfiles, int maxgrid, int ncomps,
                                int nboxes, bool raninit, bool mb2,
			        VisMF::Header::Version whichVersion, bool groupSets,
			        bool setBuf)
{
  VisMF::SetNOutFiles(nfiles);
  if(mb2) {
    bytesPerMB = pow(2.0, 20);
  }

  BoxArray bArray(MakeBoxArray(maxgrid, nboxes));
  if(ParallelDescriptor::IOProcessor()) {
    cout << "  Timings for writing to " << nfiles << " files:" << endl;
  }

  // make a MultiFab
  MultiFab mfout(bArray, ncomps, 0);
  for(MFIter mfiset(mfout); mfiset.isValid(); ++mfiset) {
    for(int invar(0); invar < ncomps; ++invar) {
      if(raninit) {
        Real *dp = mfout[mfiset].dataPtr(invar);
	for(int i(0); i < mfout[mfiset].box().numPts(); ++i) {
	  dp[i] = BoxLib::Random() + (1.0 + static_cast<Real> (invar));
	}
      } else {
        mfout[mfiset].setVal((100.0 * mfiset.index()) + invar, invar);
      }
    }
  }

  long npts(bArray[0].numPts());
  long totalNBytes(npts * ncomps * nboxes *sizeof(Real));
  std::string mfName;
  switch(whichVersion) {
    case VisMF::Header::NoFabHeader_v1:
      mfName = "TestMFNoFabHeader";
    break;
    case VisMF::Header::NoFabHeaderMinMax_v1:
      mfName = "TestMFNoFabHeaderMinMax";
    break;
    case VisMF::Header::NoFabHeaderFAMinMax_v1:
      mfName = "TestMFNoFabHeaderFAMinMax";
    break;
    default:
      BoxLib::Abort("**** Error in TestWriteNFilesNoFabHeader:: bad version.");
  }

  VisMF::RemoveFiles(mfName, true);

  ParallelDescriptor::Barrier();
  double wallTimeStart(ParallelDescriptor::second());

  VisMF::Header::Version currentVersion(VisMF::GetHeaderVersion());
  VisMF::SetHeaderVersion(whichVersion);
  VisMF::WriteNoFabHeader(mfout, mfName, whichVersion, groupSets, setBuf); 
  VisMF::SetHeaderVersion(currentVersion);

  double wallTime(ParallelDescriptor::second() - wallTimeStart);

  double wallTimeMax(wallTime);
  double wallTimeMin(wallTime);

  ParallelDescriptor::ReduceRealMin(wallTimeMin);
  ParallelDescriptor::ReduceRealMax(wallTimeMax);
  Real megabytes((static_cast<Real> (totalNBytes)) / bytesPerMB);

  if(ParallelDescriptor::IOProcessor()) {
    cout << std::setprecision(5);
    cout << "------------------------------------------" << endl;
    cout << "  Total megabytes = " << megabytes << endl;
    cout << "  Write:  Megabytes/sec   = " << megabytes/wallTimeMax << endl;
    cout << "  Wall clock time = " << wallTimeMax << endl;
    cout << "  Min wall clock time = " << wallTimeMin << endl;
    cout << "  Max wall clock time = " << wallTimeMax << endl;
    cout << "------------------------------------------" << endl;
  }
}


// -------------------------------------------------------------
void TestReadMF(const std::string &mfName) {
  MultiFab mfin;

  ParallelDescriptor::Barrier();
  double wallTimeStart(ParallelDescriptor::second());

  VisMF::Read(mfin, mfName); 

  for(int i(0); i < mfin.nComp(); ++i) {
    Real mfMin = mfin.min(i);
    Real mfMax = mfin.max(i);
    if(ParallelDescriptor::IOProcessor()) {
      std::cout << "MMMMMMMM:  i mfMin mfMax = " << i << "  " << mfMin << "  " << mfMax << std::endl;
    }
  }


  double wallTime(ParallelDescriptor::second() - wallTimeStart);

  double wallTimeMax(wallTime);
  double wallTimeMin(wallTime);

  ParallelDescriptor::ReduceRealMin(wallTimeMin);
  ParallelDescriptor::ReduceRealMax(wallTimeMax);

  long npts(mfin.boxArray()[0].numPts());
  int  ncomps(mfin.nComp());
  int  nboxes(mfin.boxArray().size());
  long totalNBytes(npts * ncomps * nboxes *sizeof(Real));
  Real megabytes((static_cast<Real> (totalNBytes)) / bytesPerMB);

  if(ParallelDescriptor::IOProcessor()) {
    cout << std::setprecision(5);
    cout << "------------------------------------------" << endl;
    cout << "  ncomps = " << ncomps << endl;
    cout << "  nboxes = " << nboxes << endl;
    cout << "  Total megabytes = " << megabytes << endl;
    cout << "  Read:  Megabytes/sec   = " << megabytes/wallTimeMax << endl;
    cout << "  Wall clock time = " << wallTimeMax << endl;
    cout << "  Min wall clock time = " << wallTimeMin << endl;
    cout << "  Max wall clock time = " << wallTimeMax << endl;
    cout << "------------------------------------------" << endl;
  }
}
// -------------------------------------------------------------
// -------------------------------------------------------------


