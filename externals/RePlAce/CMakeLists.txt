cmake_minimum_required(VERSION 3.5)

project(replace)

set(CMAKE_CXX_STANDARD 11)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dcimg_use_jpeg=1 -DUSE_FFT2D_PTHREADS=1 -DFFT_2D_MAX_THREADS=16 -O3 ${OpenMP_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Dcimg_display=0 -Dcimg_use_jpeg=0 -O3 -g -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "-g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(THREADS_PREFER_PTHREAD_FLAG ON)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(REPLACE_HOME ${PROJECT_SOURCE_DIR} )
set(DEFLIB_HOME
  ${PROJECT_SOURCE_DIR}/module/def
)
set(LEFLIB_HOME
  ${PROJECT_SOURCE_DIR}/module/lef
)
set(FLUTE_HOME
  ${PROJECT_SOURCE_DIR}/module/flute
)
set(OPENSTA_HOME
  ${PROJECT_SOURCE_DIR}/module/OpenSTA
)
set(CIMG_HOME
  ${PROJECT_SOURCE_DIR}/module/CImg
)
set(EIGEN_HOME
  ${PROJECT_SOURCE_DIR}/module/eigen-git-mirror 
)

add_subdirectory(${DEFLIB_HOME})
add_subdirectory(${LEFLIB_HOME})
add_subdirectory(${FLUTE_HOME})
# add_subdirectory(${OPENSTA_HOME})

# find_package(GoogleHash) # Optional
find_package(ZLIB REQUIRED)
# find_package(X11) # Disabled
# find_package(JPEG) # Disabled
find_package(Threads REQUIRED)

set (REPLACE_SRC
  src/argument.cpp
  src/bin.cpp
  src/bookShelfIO.cpp
  src/charge.cpp
  src/defParser.cpp
  src/detailPlace.cpp
  src/fft.cpp
  src/fftsg2d.cpp
  src/fftsg3d.cpp
  src/fftsg.cpp
  src/gcell.cpp
  src/initPlacement.cpp
  src/lefdefIO.cpp
  src/lefParser.cpp
  src/macro.cpp
  src/main.cpp
  src/util.cpp
  src/ns.cpp
  src/opt.cpp
  src/plot.cpp
  src/routeOpt.cpp
  src/tier.cpp
  src/timing.cpp
  # src/timingSta.cpp
  src/trial.cpp
  src/wlen.cpp
  src/replace_external.cpp
  src/replace_private.cpp
  # src/replace_wrap.cpp
  )


set (REPLACE_HEADER
  src/bin.h
  src/bookShelfIO.h
  src/charge.h
  src/fft.h
  src/gcell.h
  src/replace_private.h
  src/replace_external.h
  src/initPlacement.h
  src/ns.h
  src/opt.h
  src/plot.h
  src/routeOpt.h
  src/util.h
  src/wlen.h
  src/timing.h
  src/timingSta.h
  src/macro.h
  src/lefdefIO.h
  )

include_directories(
  .
  src
  /opt/homebrew/include
  ${EIGEN_HOME}
  ${CIMG_HOME}
  ${FLUTE_HOME}
  ${DEFLIB_HOME}/def
  ${DEFLIB_HOME}/cdef
  ${LEFLIB_HOME}/lef
  ${LEFLIB_HOME}/clef
  # ${OPENSTA_HOME}/app
  # ${OPENSTA_HOME}/util
  # ${OPENSTA_HOME}/graph
  # ${OPENSTA_HOME}/db
  # ${OPENSTA_HOME}/lib
  # ${OPENSTA_HOME}/search
  # ${OPENSTA_HOME}/parasitics
  # ${OPENSTA_HOME}/sdc
  # ${OPENSTA_HOME}/sta
  )


add_executable( replace
  ${REPLACE_SRC}
)

target_link_libraries( replace
  # OpenSTA
  flute
  def
  defzlib
  cdef
  cdefzlib
  lef
  lefzlib
  clef
  clefzlib

  ${ZLIB_LIBRARIES}
  # ${X11_LIBRARIES}
  # ${JPEG_LIBRARIES}
  # ${TCL_LIB} # Removed TCL

  Threads::Threads
  )

install(FILES ${REPLACE_HEADER} DESTINATION include)
install(TARGETS replace DESTINATION bin)
