find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# FFmpeg comes from pkg-config on the maintainer platform and from vcpkg on Windows (vcpkg.json),
# where no system package exists. Consumers link one interface target so no target below carries a
# platform branch.
if(WIN32)
  # vcpkg's ffmpeg config exposes variables, not imported targets.
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg INTERFACE)
  target_include_directories(ninfer_ffmpeg INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg INTERFACE ${FFMPEG_LIBRARIES})
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale)
  add_library(ninfer_ffmpeg INTERFACE)
  target_link_libraries(ninfer_ffmpeg INTERFACE PkgConfig::FFMPEG)
endif()
add_library(ninfer::ffmpeg ALIAS ninfer_ffmpeg)

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

# CPU grammar source base; consumers will link it explicitly.
add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment), so the
  # constraint travels with the discovery on both platforms.
  if(WIN32)
    find_package(CURL 7.85 REQUIRED)
    add_library(ninfer_curl INTERFACE)
    target_link_libraries(ninfer_curl INTERFACE CURL::libcurl)
  else()
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    add_library(ninfer_curl INTERFACE)
    target_link_libraries(ninfer_curl INTERFACE PkgConfig::LIBCURL)
  endif()
  add_library(ninfer::curl ALIAS ninfer_curl)
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
