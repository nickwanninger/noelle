# Project
cmake_minimum_required(VERSION 3.13)
project(PDG)

# Dependences
include(${CMAKE_CURRENT_SOURCE_DIR}/../../scripts/DependencesCMake.txt)

# Pass
add_subdirectory(src)

# Install
install(
  FILES
  include/noelle/core/PDG.hpp
  include/noelle/core/SCC.hpp
  include/noelle/core/SCCDAG.hpp
  include/noelle/core/PDGPrinter.hpp
  DESTINATION 
  include/noelle/core
  )
