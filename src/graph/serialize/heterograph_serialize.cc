/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/serialize/heterograph_serialize.cc
 * \brief DGLHeteroGraph serialization implementation
 *
 * The storage structure is
 * {
 *   // MetaData Section
 *   uint64_t kDGLSerializeMagic
 *   uint64_t kVersion = 2
 *   uint64_t GraphType = kDGLHeteroGraph
 *   dgl_id_t num_graphs
 *   ** Reserved Area till 4kB **
 *
 *   uint64_t gdata_start_pos (This stores the start position of graph_data,
 * which is used to skip label dict part if unnecessary)
 *   vector<pair<string, NDArray>> label_dict (To store the dict[str, NDArray])
 *
 *   vector<HeteroGraphData> graph_datas;
 *   vector<dgl_id_t> graph_indices (start address of each graph)
 *   uint64_t size_of_graph_indices_vector (Used to seek to graph_indices
 * vector)
 *
 * }
 *
 * Storage of HeteroGraphData is
 * {
 *   HeteroGraphPtr ptr;
 *   vector<vector<pair<string, NDArray>>> node_tensors;
 *   vector<vector<pair<string, NDArray>>> edge_tensors;
 *   vector<string> ntype_name;
 *   vector<string> etype_name;
 * }
 *
 */

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/LogLevel.h>
#include <aws/core/utils/stream/PreallocatedStreamBuf.h>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/model/GetObjectRequest.h>
#include <aws/s3-crt/model/PutObjectRequest.h>
#include <aws/s3-crt/model/GetBucketLocationRequest.h>
#include <aws/s3-crt/model/HeadObjectRequest.h>

#include <dgl/graph_op.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "../heterograph.h"
#include "./graph_serialize.h"
#include "./streamwithcount.h"
#include "dmlc/memory_io.h"

namespace dgl {
namespace serialize {

using namespace dgl::runtime;
using dmlc::SeekStream;
using dmlc::Stream;
using dmlc::io::FileSystem;
using dmlc::io::URI;

bool SaveHeteroGraphs(std::string filename, List<HeteroGraphData> hdata,
                      const std::vector<NamedTensor> &nd_list) {
  auto fs = std::unique_ptr<StreamWithCount>(
    StreamWithCount::Create(filename.c_str(), "w", false));
  CHECK(fs->IsValid()) << "File name " << filename << " is not a valid name";

  // Write DGL MetaData
  const uint64_t kVersion = 2;
  std::array<char, 4096> meta_buffer;

  // Write metadata into char buffer with size 4096
  dmlc::MemoryFixedSizeStream meta_fs_(meta_buffer.data(), 4096);
  auto meta_fs = static_cast<Stream *>(&meta_fs_);
  meta_fs->Write(kDGLSerializeMagic);
  meta_fs->Write(kVersion);
  meta_fs->Write(GraphType::kHeteroGraph);
  uint64_t num_graph = hdata.size();
  meta_fs->Write(num_graph);

  // Write metadata into files
  fs->Write(meta_buffer.data(), 4096);

  // Calculate label dict binary size
  std::string labels_blob;
  dmlc::MemoryStringStream label_fs_(&labels_blob);
  auto label_fs = static_cast<Stream *>(&label_fs_);
  label_fs->Write(nd_list);

  uint64_t gdata_start_pos =
    fs->Count() + sizeof(uint64_t) + labels_blob.size();

  // Write start position of gdata, which can be skipped when only reading gdata
  // And label dict
  fs->Write(gdata_start_pos);
  fs->Write(labels_blob.c_str(), labels_blob.size());

  std::vector<uint64_t> graph_indices(num_graph);

  // Write HeteroGraphData
  for (uint64_t i = 0; i < num_graph; ++i) {
    graph_indices[i] = fs->Count();
    auto gdata = hdata[i].sptr();
    fs->Write(gdata);
  }

  // Write indptr into string to count size
  std::string indptr_blob;
  dmlc::MemoryStringStream indptr_fs_(&indptr_blob);
  auto indptr_fs = static_cast<Stream *>(&indptr_fs_);
  indptr_fs->Write(graph_indices);

  uint64_t indptr_buffer_size = indptr_blob.size();
  fs->Write(indptr_blob);
  fs->Write(indptr_buffer_size);

  return true;
}

void parseS3Path(const std::string &fname, std::string *bucket,
                 std::string *object) {
  if (fname.empty()) {
    throw std::invalid_argument{"The filename cannot be an empty string."};
  }

  if (fname.size() < 5 || fname.substr(0, 5) != "s3://") {
    throw std::invalid_argument{
            "The filename must start with the S3 scheme."};
  }

  std::string path = fname.substr(5);

  if (path.empty()) {
    throw std::invalid_argument{"The filename cannot be an empty string."};
  }

  auto pos = path.find_first_of('/');
  if (pos == 0) {
    throw std::invalid_argument{
            "The filename does not contain a bucket name."};
  }

  *bucket = path.substr(0, pos);
  *object = path.substr(pos + 1);
  if (pos == std::string::npos) {
    *object = "";
  }
}

std::shared_ptr<Aws::S3Crt::S3CrtClient> getS3Client() {
  Aws::SDKOptions options;
  Aws::InitAPI(options);

  Aws::S3Crt::ClientConfiguration cfg;

  // S3 Streaming Constants.
  // Defaults for p3.8xlarge in us-east-1. needs to be adapted by python client
  cfg.throughputTargetGbps = 5; // Set for p3.8xlarge with 10Gbps card
  cfg.partSize = 16 * 1024 * 1024;  // 16 MB.
  cfg.region = "us-east-1";

  return std::shared_ptr<Aws::S3Crt::S3CrtClient>(new Aws::S3Crt::S3CrtClient(cfg));
}

std::istream& s3_read_crt(const std::string &file_url, std::uint64_t &elapsed_seconds) {

  std::string bucket, object;
  parseS3Path(file_url, &bucket, &object);

  Aws::S3Crt::Model::GetObjectRequest request;
  request.SetBucket(std::move(bucket));
  request.SetKey(std::move(object));

  auto start = std::chrono::steady_clock::now();

  Aws::S3Crt::Model::GetObjectOutcome getObjectOutcome = getS3Client()->GetObject(request);

  auto finish = std::chrono::steady_clock::now();
  elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();

  if (getObjectOutcome.IsSuccess()) {
    auto outcome = getObjectOutcome.GetResultWithOwnership();
    return outcome.GetBody();
  }
  else {
    return NULL;
  }
}

std::uint64_t get_object_size(const std::string &bucket,
                              const std::string &object) {
  Aws::S3Crt::Model::HeadObjectRequest headObjectRequest;
  headObjectRequest.WithBucket(bucket.c_str()).WithKey(object.c_str());
  auto headObjectOutcome = getS3Client()->HeadObject(headObjectRequest);
  if (headObjectOutcome.IsSuccess()) {
    return headObjectOutcome.GetResult().GetContentLength();
  }
  Aws::String const &error_aws = headObjectOutcome.GetError().GetMessage();
  std::string error_str(error_aws.c_str(), error_aws.size());
  throw std::invalid_argument(error_str);
  return 0;
}

std::uint64_t get_object_size(const std::string &file_url){
  std::string bucket, object;
  parseS3Path(file_url, &bucket, &object);
  return get_object_size(bucket, object);
}

std::vector<HeteroGraphData> LoadHeteroGraphs(const std::string &filename,
                                              std::vector<dgl_id_t> idx_list) {
  std::uint64_t elapsed_seconds;
  auto ret = s3_read_crt(filename, elapsed_seconds);
//  auto fs = std::unique_ptr<SeekStream>(
//    SeekStream::CreateForRead(filename.c_str(), false));

  CHECK(fs) << "File name " << filename << " is not a valid name";
  // Read DGL MetaData
  uint64_t magicNum, graphType, version, num_graph;
  fs->Read(&magicNum);
  fs->Read(&version);
  fs->Read(&graphType);
  CHECK(fs->Read(&num_graph)) << "Invalid num of graph";
  fs->Seek(4096);

  CHECK_EQ(magicNum, kDGLSerializeMagic) << "Invalid DGL files";
  CHECK_EQ(version, 2) << "Invalid GraphType";
  CHECK_EQ(graphType, GraphType::kHeteroGraph) << "Invalid GraphType";

  uint64_t gdata_start_pos;
  fs->Read(&gdata_start_pos);
  // Skip labels part
  fs->Seek(gdata_start_pos);

  std::vector<HeteroGraphData> gdata_refs;
  if (idx_list.empty()) {
    // Read All Graphs
    gdata_refs.reserve(num_graph);
    for (uint64_t i = 0; i < num_graph; ++i) {
      HeteroGraphData gdata = HeteroGraphData::Create();
      auto hetero_data = gdata.sptr();
      fs->Read(&hetero_data);
      gdata_refs.push_back(gdata);
    }
  } else {
    uint64_t gdata_start_pos = fs->Tell();
    // Read Selected Graphss
    gdata_refs.reserve(idx_list.size());
    URI uri(filename.c_str());
    uint64_t filesize = FileSystem::GetInstance(uri)->GetPathInfo(uri).size;
    fs->Seek(filesize - sizeof(uint64_t));
    uint64_t indptr_buffer_size;
    fs->Read(&indptr_buffer_size);

    std::vector<uint64_t> graph_indices(num_graph);
    fs->Seek(filesize - sizeof(uint64_t) - indptr_buffer_size);
    fs->Read(&graph_indices);

    fs->Seek(gdata_start_pos);
    // Would be better if idx_list is sorted. However the returned the graphs
    // should be the same order as the idx_list
    for (uint64_t i = 0; i < idx_list.size(); ++i) {
      auto gid = idx_list[i];
      CHECK((gid < graph_indices.size()) && (gid >= 0))
        << "ID " << gid
        << " in idx_list is out of bound. Please check your idx_list.";
      fs->Seek(graph_indices[gid]);
      HeteroGraphData gdata = HeteroGraphData::Create();
      auto hetero_data = gdata.sptr();
      fs->Read(&hetero_data);
      gdata_refs.push_back(gdata);
    }
  }

  return gdata_refs;
}

std::vector<NamedTensor> LoadLabels_V2(const std::string &filename) {
  auto fs = std::unique_ptr<SeekStream>(
    SeekStream::CreateForRead(filename.c_str(), false));
  CHECK(fs) << "File name " << filename << " is not a valid name";
  // Read DGL MetaData
  uint64_t magicNum, graphType, version, num_graph;
  fs->Read(&magicNum);
  fs->Read(&version);
  fs->Read(&graphType);
  CHECK(fs->Read(&num_graph)) << "Invalid num of graph";
  fs->Seek(4096);

  uint64_t gdata_start_pos;
  fs->Read(&gdata_start_pos);

  std::vector<NamedTensor> labels_list;
  fs->Read(&labels_list);

  return labels_list;
}

DGL_REGISTER_GLOBAL("data.heterograph_serialize._CAPI_MakeHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    List<Map<std::string, Value>> ndata = args[1];
    List<Map<std::string, Value>> edata = args[2];
    List<Value> ntype_names = args[3];
    List<Value> etype_names = args[4];
    *rv = HeteroGraphData::Create(hg.sptr(), ndata, edata, ntype_names,
                                  etype_names);
  });

DGL_REGISTER_GLOBAL("data.heterograph_serialize._CAPI_SaveHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    List<HeteroGraphData> hgdata = args[1];
    Map<std::string, Value> nd_map = args[2];
    std::vector<NamedTensor> nd_list;
    for (auto kv : nd_map) {
      NDArray ndarray = static_cast<NDArray>(kv.second->data);
      nd_list.emplace_back(kv.first, ndarray);
    }
    *rv = dgl::serialize::SaveHeteroGraphs(filename, hgdata, nd_list);
  });

DGL_REGISTER_GLOBAL(
  "data.heterograph_serialize._CAPI_GetGindexFromHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphData hdata = args[0];
    *rv = HeteroGraphRef(hdata->gptr);
  });

DGL_REGISTER_GLOBAL(
  "data.heterograph_serialize._CAPI_GetEtypesFromHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphData hdata = args[0];
    List<Value> etype_names;
    for (const auto &name : hdata->etype_names) {
      etype_names.push_back(Value(MakeValue(name)));
    }
    *rv = etype_names;
  });

DGL_REGISTER_GLOBAL(
  "data.heterograph_serialize._CAPI_GetNtypesFromHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphData hdata = args[0];
    List<Value> ntype_names;
    for (auto name : hdata->ntype_names) {
      ntype_names.push_back(Value(MakeValue(name)));
    }
    *rv = ntype_names;
  });

DGL_REGISTER_GLOBAL(
  "data.heterograph_serialize._CAPI_GetNDataFromHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphData hdata = args[0];
    List<List<Value>> ntensors;
    for (auto tensor_list : hdata->node_tensors) {
      List<Value> nlist;
      for (const auto &kv : tensor_list) {
        nlist.push_back(Value(MakeValue(kv.first)));
        nlist.push_back(Value(MakeValue(kv.second)));
      }
      ntensors.push_back(nlist);
    }
    *rv = ntensors;
  });

DGL_REGISTER_GLOBAL(
  "data.heterograph_serialize._CAPI_GetEDataFromHeteroGraphData")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    HeteroGraphData hdata = args[0];
    List<List<Value>> etensors;
    for (auto tensor_list : hdata->edge_tensors) {
      List<Value> elist;
      for (const auto &kv : tensor_list) {
        elist.push_back(Value(MakeValue(kv.first)));
        elist.push_back(Value(MakeValue(kv.second)));
      }
      etensors.push_back(elist);
    }
    *rv = etensors;
  });

DGL_REGISTER_GLOBAL("data.graph_serialize._CAPI_LoadLabels_V2")
  .set_body([](DGLArgs args, DGLRetValue *rv) {
    std::string filename = args[0];
    auto labels_list = LoadLabels_V2(filename);
    Map<std::string, Value> rvmap;
    for (auto kv : labels_list) {
      rvmap.Set(kv.first, Value(MakeValue(kv.second)));
    }
    *rv = rvmap;
  });

}  // namespace serialize
}  // namespace dgl
