/*------------------------------------------------------------------------*/
/*  Copyright 2014 Sandia Corporation.                                    */
/*  This software is released under the license detailed                  */
/*  in the file, LICENSE, which is located in the top-level Nalu          */
/*  directory structure                                                   */
/*------------------------------------------------------------------------*/


// nalu
#include <NaluEnv.h>
#include <element_promotion/PromoteElementTest.h>
#include <element_promotion/QuadratureRuleTest.h>
#include <element_promotion/MasterElementHOTest.h>
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

  //==============================
  // Elem promotion unit test options

  const bool doQuadrature = true;
  const bool doMasterElementQuad = true;
  const bool doMasterElementHex= true;
  const bool doPromotionQuadGaussLegendre = true;
  const bool doPromotionQuadSGL = true;
  const bool doPromotionHexGaussLegendre = true;
  const bool doPromotionHexSGL = true;


  const int maxQuadOrder = 7;
  const int maxHexOrder = 7;

  //std::string quadMesh = "test_meshes/2d_1m_P1.g";
  //std::string quadMesh = "test_meshes/1x1_tquad4_R0.g";
  std::string quadMesh = "test_meshes/quad4_64.g";


  //std::string hexMesh = "test_meshes/thex8_8.g";
  //std::string hexMesh = "test_meshes/2cm_ped_35K_mks.g";
  //std::string hexMesh = "test_meshes/1cm_ped_35KR.g";
  //std::string hexMesh = "test_meshes/100cm_13K_S_R1.g";
  //std::string hexMesh = "test_meshes/hexLdomain.g";
  std::string hexMesh = "test_meshes/hex8_4.g";
  //==============================


  if ( doQuadrature ) {
    sierra::naluUnit::QuadratureRuleTest().execute();
  }

  if ( doMasterElementQuad ) {
    for (int j = 2; j <= maxQuadOrder; ++j) {
      sierra::naluUnit::MasterElementHOTest(2, j).execute();
    }
  }

  if (doMasterElementHex ) {
    for (int j = 2; j <= maxHexOrder; ++j) {
      sierra::naluUnit::MasterElementHOTest(3, j).execute();
    }
  }

  if (doPromotionQuadGaussLegendre) {
    for (int j = 2; j <= maxQuadOrder; ++j) {
      sierra::naluUnit::PromoteElementTest(2, j, quadMesh, "GaussLegendre").execute();
    }
  }

  if (doPromotionQuadSGL) {
    for (int j = 2; j <= maxQuadOrder; ++j) {
      sierra::naluUnit::PromoteElementTest(2, j, quadMesh, "SGL").execute();
    }
  }

  if (doPromotionHexGaussLegendre) {
    for (int j = 2; j <= maxHexOrder; ++j) {
      sierra::naluUnit::PromoteElementTest(3, j, hexMesh, "GaussLegendre").execute();
    }
  }

  if (doPromotionHexSGL) {
    for (int j = 2; j <= maxHexOrder; ++j) {
      sierra::naluUnit::PromoteElementTest(3, j, hexMesh, "SGL").execute();
    }
  }

  // all done
  return 0;
}
