/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


#include <mpi.h>

// nalu
#include <NaluEnv.h>
#include <Overset.h>

// util
#include <stk_util/environment/CPUTime.hpp>
#include <stk_util/environment/perf_util.hpp>

// boost for input params
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>


int main( int argc, char ** argv )
{

  // start up MPI
  if ( MPI_SUCCESS != MPI_Init( &argc , &argv ) ) {
    throw std::runtime_error("MPI_Init failed");
  }

  // NaluEnv singleton
  sierra::naluUnit::NaluEnv &naluEnv = sierra::naluUnit::NaluEnv::self();
  
  // command line options.
  std::string inputFileName, logFileName;
 
  boost::program_options::options_description desc("Nalu Supported Options");
  desc.add_options()
    ("help,h","Help message")
    ("version,v", "Code Version 1.0")
    ("log-file,o", boost::program_options::value<std::string>(&logFileName),
        "Analysis log file");

  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

  boost::program_options::notify(vm);

  // deal with some default parameters
  if ( vm.count("help") ) {
    if (!naluEnv.parallel_rank())
      std::cerr << desc << std::endl;
    return 0;
  }

  if (vm.count("version")) {
    if (!naluEnv.parallel_rank())
      std::cerr << "Version: Nalu1.0" << std::endl;
    return 0;
  }

  // deal with logfile name; if none supplied, go with naluInit.log
  if (!vm.count("log-file")) {
    logFileName = "naluUnit.log";
  }

  // deal with log file stream
  naluEnv.set_log_file_stream(logFileName);  
  
  naluEnv.naluOutputP0() << "NaluUnit Shall Commence" << std::endl;
 
  // create overset unit test
  sierra::naluUnit::Overset *overset = new sierra::naluUnit::Overset();
  
  // execute it
  overset->execute();

  // delete it
  delete overset;
  
  // all done  
  return 0;
}
