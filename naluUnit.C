/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <NaluEnv.h>
#include <element_promotion/PromoteElementTest.h>
#include <mpi.h>
#include <overset/Overset.h>
#include <surfaceFields/SurfaceFields.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <boost/program_options.hpp>

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

  //==============================
  // create; execute; delete
  //==============================

  // overset
  const bool doOverset = false;
  if ( doOverset ) {
    sierra::naluUnit::Overset *overset = new sierra::naluUnit::Overset();
    overset->execute();
    delete overset;
  }

  // surface
  const bool doSurfaceFields = false;
  if ( doSurfaceFields ) {
    sierra::naluUnit::SurfaceFields *surfaceFields = new sierra::naluUnit::SurfaceFields();
    surfaceFields->execute();
    delete surfaceFields;
  }

  //std::string quadMesh = "test_meshes/1x1_tquad4_R0.g";
  std::string quadMesh = "test_meshes/quad4_64.g";
  const bool doPromotionQuad9 = true;
  if ( doPromotionQuad9 ) {
    auto promoteTest = new sierra::naluUnit::PromoteElementTest("Quad9", quadMesh);
    promoteTest->execute();
    delete promoteTest;
  }

  const bool doPromotionQuad16 = true;
  if ( doPromotionQuad16 ) {
    auto promoteTest = new sierra::naluUnit::PromoteElementTest("Quad16", quadMesh);
    promoteTest->execute();
    delete promoteTest;
  }

  const bool doPromotionQuad25 = true;
  if ( doPromotionQuad25 ) {
    auto promoteTest = new sierra::naluUnit::PromoteElementTest("Quad25", quadMesh);
    promoteTest->execute();
    delete promoteTest;
  }

  const bool doPromotionQuad36 = true;
  if ( doPromotionQuad36 ) {
    auto promoteTest = new sierra::naluUnit::PromoteElementTest("Quad36", quadMesh);
    promoteTest->execute();
    delete promoteTest;
  }

  const bool doPromotionHex27 = true;
  if ( doPromotionHex27 ) {
    auto promoteTest = new sierra::naluUnit::PromoteElementTest("Hex27", "test_meshes/hex8_32.g");
    promoteTest->execute();
    delete promoteTest;
  }

  // all done
  return 0;
}
