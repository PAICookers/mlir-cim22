add_mlir_library(MLIRCIM22Artifact
  ProgramImage.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIM22/Artifact

  DEPENDS
  MLIRCIM22ArtifactSchemaIncGen

  LINK_LIBS PUBLIC
  ${MLIRCIM22_FLATBUFFERS_TARGET}
  LLVMSupport
)
