project(flightros)

cmake_minimum_required(VERSION 3.0.0)

find_package(catkin_simple REQUIRED)

find_package(OpenCV REQUIRED)

option(BUILD_MP "Build Motion Planning" ON)

catkin_simple()

# Setup Default Build Type as Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif ()

# Setup Architcture-specific Flags
if ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
    message(STATUS "Using ARMv7 optimized flags!")
    set(CMAKE_CXX_ARCH_FLAGS " -Wno-psabi -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
elseif ("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    message(STATUS "Using ARM aarch64 optimized flags!")
    set(CMAKE_CXX_ARCH_FLAGS " -Wno-psabi -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
else ()
    set(CMAKE_CXX_ARCH_FLAGS " -march=native")
endif ()

# Setup General C++ Flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_STACK_ALLOCATION_LIMIT=1048576")
# otherwise double free or corruption (out) error when running racing or motion_planning example
add_compile_options(-O3)

# Setup Release and Debug flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_ARCH_FLAGS} -Wall -DNDEBUG -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -g")

# #pilot

# cs_add_library(flight_pilot
#     src/pilot/flight_pilot.cpp
# )

# target_link_libraries(flight_pilot
#   ${catkin_LIBRARIES}
#   ${OpenCV_LIBRARIES}
#   stdc++fs
# )

# cs_add_executable(flight_pilot_node
#    src/pilot/flight_pilot_node.cpp
# )

# target_link_libraries(flight_pilot_node
#   flight_pilot
#   ${OpenCV_LIBRARIES}
#   stdc++fs
# )

if(BUILD_MP)

    # motion_planning
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        message("Found OpenMP ${OpenMP_CXX_FOUND}  ${OpenMP_VERSION} ${OpenMP_CXX_VERSION_MAJOR} ${Open3D_VERSION} OpenMP::OpenMP_CXX")
        get_cmake_property(_variableNames VARIABLES)
        list (SORT _variableNames)
        foreach (_variableName ${_variableNames})
            message(STATUS "${_variableName}=${${_variableName}}")
        endforeach()
    else ()
        message("OpenMP not found")
    endif ()

    list(APPEND CMAKE_PREFIX_PATH "/home/kblee/open3d_install")

    # Open3D
    find_package(Open3D)
    if (Open3D_FOUND)
        message("Found Open3D ${Open3D_VERSION}")
        list(APPEND Open3D_LIBRARIES dl)
        # link_directories must be before add_executable
        link_directories(${Open3D_LIBRARY_DIRS})
    else ()
        message("Open3D not found")
    endif ()

    find_package(ompl REQUIRED)
    if (OMPL_FOUND)
        message("Found OMPL ${OMPL_VERSION}")
        include_directories(${OMPL_INCLUDE_DIRS})

        else ()
        message("OMPL not found")
    endif ()

    if (NOT OpenMP_CXX_FOUND)
        message("Failed to find OpenMP")
    endif ()
    if (NOT Open3D_FOUND)
        message("Failed to find Open3D")
    endif ()
    if (NOT OMPL_FOUND)
        message("Failed to find OMPL")
    endif ()

    if (OpenMP_CXX_FOUND AND Open3D_FOUND AND OMPL_FOUND)
        cs_add_executable(motion_planning
            src/motion_planning/motion_planning.cpp
        )
        target_include_directories(motion_planning PUBLIC ${Open3D_INCLUDE_DIRS})

        target_link_libraries(motion_planning
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        stdc++fs
        ompl
        ${Open3D_LIBRARIES}
        OpenMP::OpenMP_CXX
        zmq
        zmqpp
        )
    else ()
        message("Failed to build motion planning")
    endif ()

endif()

# racing

catkin_package(
LIBRARIES
CATKIN_DEPENDS
)

cs_add_executable(racing
    src/racing/racing.cpp
)

target_link_libraries(racing
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
stdc++fs
zmq
zmqpp
)

# camera

cs_add_executable(camera
    src/camera/camera.cpp
)

target_link_libraries(camera
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
stdc++fs
zmq
zmqpp
)

# image_data_generator

cs_add_executable(image_data_generator
    src/test/image_data_generator.cpp
)

target_link_libraries(image_data_generator
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
stdc++fs
zmq
zmqpp
)

# object_detection

cs_add_executable(object_detection
    src/test/object_detection.cpp
)

target_link_libraries(object_detection
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
stdc++fs
zmq
zmqpp
)

# trajectory

cs_add_executable(trajectory
    src/test/trajectory.cpp
)

target_link_libraries(trajectory
# ${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
stdc++fs
zmq
zmqpp
)

# planning

if(BUILD_MP)

    # planning
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        message("Found OpenMP ${OpenMP_CXX_FOUND}  ${OpenMP_VERSION} ${OpenMP_CXX_VERSION_MAJOR} ${Open3D_VERSION} OpenMP::OpenMP_CXX")
        get_cmake_property(_variableNames VARIABLES)
        list (SORT _variableNames)
        foreach (_variableName ${_variableNames})
            message(STATUS "${_variableName}=${${_variableName}}")
        endforeach()
    else ()
        message("OpenMP not found")
    endif ()

    list(APPEND CMAKE_PREFIX_PATH "/home/kblee/open3d_install")

    # Open3D
    find_package(Open3D)
    if (Open3D_FOUND)
        message("Found Open3D ${Open3D_VERSION}")
        list(APPEND Open3D_LIBRARIES dl)
        # link_directories must be before add_executable
        link_directories(${Open3D_LIBRARY_DIRS})
    else ()
        message("Open3D not found")
    endif ()

    find_package(ompl REQUIRED)
    if (OMPL_FOUND)
        message("Found OMPL ${OMPL_VERSION}")
        include_directories(${OMPL_INCLUDE_DIRS})

        else ()
        message("OMPL not found")
    endif ()

    message("==============================================================")
    message("====================== Check message!!! ======================")
    message("==============================================================")

    if (NOT OpenMP_CXX_FOUND)
        message("Failed to find OpenMP")
    endif ()
    if (NOT Open3D_FOUND)
        message("Failed to find Open3D")
    endif ()
    if (NOT OMPL_FOUND)
        message("Failed to find OMPL")
    endif ()

    if (OpenMP_CXX_FOUND AND Open3D_FOUND AND OMPL_FOUND)

        message("=====================================================================")
        message("====================== Start building planning ======================")
        message("=====================================================================")

        cs_add_executable(planning
            src/test/planning.cpp
        )
        target_include_directories(planning PUBLIC ${Open3D_INCLUDE_DIRS})

        target_link_libraries(planning
        ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES}
        stdc++fs
        ompl
        ${Open3D_LIBRARIES}
        OpenMP::OpenMP_CXX
        zmq
        zmqpp
        )
    else ()
        message("Failed to build planning")
    endif ()

endif()

# point_cloud

cs_add_executable(point_cloud
    src/test/point_cloud.cpp
)

target_link_libraries(point_cloud
${catkin_LIBRARIES}
${OpenCV_LIBRARIES}
stdc++fs
zmq
zmqpp
)

# Finish
cs_install()
cs_export()