IF (TENGINE_ENABLE_BANG)
    IF (${CMAKE_MINOR_VERSION} LESS 17)
        MESSAGE (FATAL_ERROR "Tengine: Backend needs CMake version >= 3.17.")
    ENDIF()

    SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/modules)
    FIND_PACKAGE(CNRT REQUIRED)
    FIND_PACKAGE(CNNL REQUIRED)
ENDIF()