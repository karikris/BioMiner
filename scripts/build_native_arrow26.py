#!/usr/bin/env python3
"""Build one pinned, host-specific Arrow 26 candidate; never admit a runtime.

First use --configure-only. Review the actual dependency selection, then resume
with --configured-inputs-sha256. Compilation receipts remain unqualified. This
independent recipe deliberately leaves the retained Arrow 25 fallback unchanged.
"""
from __future__ import annotations

import argparse
import base64
import email
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import zipfile

ARROW_COMMIT = "446167169a0dc547e00e1bc29f417a40c0ca267a"
ARROW_ARCHIVE_SHA256 = "764adbb5601a4d0d9d67a10cf04892d33cfac9fd43012f08a2d7c9956a7a90d2"
INTEGRATED_SOURCES = {
    "cpp/src/arrow/compute/kernels/copy_data_internal.h": "21b1f8acce63e0ca301a968339d731eddb5efd8e8eeb1a55638ffe665a046621",
    "cpp/src/arrow/compute/kernels/vector_replace.cc": "12cd5bd3ab8e487aba96743c92cea9f234e3efba279f1200126029f84717d9c1",
    "cpp/src/arrow/compute/kernels/vector_replace_test.cc": "c087392bbd8114e903608b36c1e4e88a087677cbcfda484cb270d227b385788e",
    "cpp/src/arrow/compute/kernels/scalar_if_else.cc": "87f5ce8b675c9dcbaf1f45b25e2515a161fe3bf4b1b567acba0ecea287510fa8",
}
FEATURES = ("ACERO", "AZURE", "COMPUTE", "CSV", "DATASET", "FILESYSTEM", "FLIGHT", "GCS", "HDFS", "JSON", "ORC", "PARQUET", "S3", "SUBSTRAIT")
CODECS = ("BROTLI", "BZ2", "LZ4", "SNAPPY", "ZLIB", "ZSTD")
PYTHON_FEATURES = ("ACERO", "AZURE", "DATASET", "FLIGHT", "GCS", "HDFS", "ORC", "PARQUET", "S3", "SUBSTRAIT", "PARQUET_ENCRYPTION")
REQUIRED_ARROW_LIBRARIES = tuple(f"lib{name}.so.2600" for name in (
    "arrow", "arrow_compute", "arrow_acero", "arrow_dataset", "arrow_flight",
    "arrow_s3", "arrow_substrait", "parquet"))
BUILDER_PINS = {
    "scikit-build-core": "1.1.1", "cython": "3.3.0", "numpy": "2.5.4", "libcst": "1.9.0",
    "setuptools-scm": "10.3.4", "setuptools": "84.2.1", "build": "1.6.1",
    "packaging": "26.3", "pyproject-hooks": "1.3.3", "pathspec": "1.1.1",
    "pyyaml": "6.0.3", "vcs-versioning": "2.6.0",
}
# Finite reviewed source cohort. Local archive paths come only from explicit CLI
# arguments; these immutable upstream identities also enter build-inputs.json.
SDK_COHORT = {
    "aws_sdk_cpp": {
        "variable": "AWSSDK",
        "version": "1.11.914",
        "build_version": "d207a37ecb8dc33b5e131d0d59843d264b7f7e86",
        "commit": "d207a37ecb8dc33b5e131d0d59843d264b7f7e86",
        "sha256": "108382fed71e70ada73d4aaa45e0ba54777f7eca6bbd44ac24b9af9ef54f069e",
        "filename": "aws-sdk-cpp-d207a37ecb8dc33b5e131d0d59843d264b7f7e86.tar.gz",
        "env": "ARROW_AWSSDK_URL",
        "old_version": "1.11.778",
        "old_sha256": "06371849792f1a8633c3e0592a17b4700f39c748ad374f4e9a6592739ecc4d9c",
        "url": "https://github.com/aws/aws-sdk-cpp/archive/d207a37ecb8dc33b5e131d0d59843d264b7f7e86.tar.gz"
    },
    "aws_crt_cpp": {
        "variable": "AWS_CRT_CPP",
        "version": "0.43.8",
        "build_version": "9c319786d1d83d5ace6a9de817dcfe1861003609",
        "commit": "9c319786d1d83d5ace6a9de817dcfe1861003609",
        "sha256": "f107357efa7d6b9be783d96d49cdaf000fe881417c431ae9fe27faf1bb55f431",
        "filename": "aws-crt-cpp-9c319786d1d83d5ace6a9de817dcfe1861003609.tar.gz",
        "env": "ARROW_AWS_CRT_CPP_URL",
        "old_version": "v0.38.0",
        "old_sha256": "f57b82f0a9e6890e4d47aa1c48b9324b5beb5b58000715c4c6ef9f147be13891",
        "url": "https://github.com/awslabs/aws-crt-cpp/archive/9c319786d1d83d5ace6a9de817dcfe1861003609.tar.gz"
    },
    "aws_c_auth": {
        "variable": "AWS_C_AUTH",
        "version": "4b5d524bf1a511b05e0fffe5bdc51800770b9427",
        "build_version": "4b5d524bf1a511b05e0fffe5bdc51800770b9427",
        "commit": "4b5d524bf1a511b05e0fffe5bdc51800770b9427",
        "sha256": "b4f6690b382abe049d543804f40230e667a95a18f64506ef7647925e8c6b9b67",
        "filename": "aws-c-auth-4b5d524bf1a511b05e0fffe5bdc51800770b9427.tar.gz",
        "env": "ARROW_AWS_C_AUTH_URL",
        "old_version": "v0.10.1",
        "old_sha256": "85d737f0f735256f1931e85e4cadbe228d88698f7b59a9b390b49ef5d0778a43",
        "url": "https://github.com/awslabs/aws-c-auth/archive/4b5d524bf1a511b05e0fffe5bdc51800770b9427.tar.gz"
    },
    "aws_c_cal": {
        "variable": "AWS_C_CAL",
        "version": "8aa2a48a09f93c65d4cf06388e143a6584de6321",
        "build_version": "8aa2a48a09f93c65d4cf06388e143a6584de6321",
        "commit": "8aa2a48a09f93c65d4cf06388e143a6584de6321",
        "sha256": "29706ab10a4d95a74db278925b82c85077c08224c0efe837841d2354555c8817",
        "filename": "aws-c-cal-8aa2a48a09f93c65d4cf06388e143a6584de6321.tar.gz",
        "env": "ARROW_AWS_C_CAL_URL",
        "old_version": "v0.9.13",
        "old_sha256": "80b7c6087b0af461b4483e4c9483aea2e0dac5d9fb2289b057159ea6032409e1",
        "url": "https://github.com/awslabs/aws-c-cal/archive/8aa2a48a09f93c65d4cf06388e143a6584de6321.tar.gz"
    },
    "aws_c_common": {
        "variable": "AWS_C_COMMON",
        "version": "f7d471b1feaea9a310851672580f9ac7e031cfc0",
        "build_version": "f7d471b1feaea9a310851672580f9ac7e031cfc0",
        "commit": "f7d471b1feaea9a310851672580f9ac7e031cfc0",
        "sha256": "a87a89fa6678fda59dd66ea9246e70d837bf71ecec39829ec712e6863d1453e4",
        "filename": "aws-c-common-f7d471b1feaea9a310851672580f9ac7e031cfc0.tar.gz",
        "env": "ARROW_AWS_C_COMMON_URL",
        "old_version": "v0.12.6",
        "old_sha256": "138822ecdcaff1d702f37d4751f245847d088592724921cc6bf61c232b198d6b",
        "url": "https://github.com/awslabs/aws-c-common/archive/f7d471b1feaea9a310851672580f9ac7e031cfc0.tar.gz"
    },
    "aws_c_compression": {
        "variable": "AWS_C_COMPRESSION",
        "version": "d8264e64f698341eb03039b96b4f44702a9b3f83",
        "build_version": "d8264e64f698341eb03039b96b4f44702a9b3f83",
        "commit": "d8264e64f698341eb03039b96b4f44702a9b3f83",
        "sha256": "537971fe42e989b638e0f436683357ed3d07f07b06b910b1c37ab3a8094bb8fd",
        "filename": "aws-c-compression-d8264e64f698341eb03039b96b4f44702a9b3f83.tar.gz",
        "env": "ARROW_AWS_C_COMPRESSION_URL",
        "old_version": "v0.3.2",
        "old_sha256": "f93f5a5d8b3fee3a6d97b14ba279efacd4d4016ef9cc7dc4be7d43519ecfbe93",
        "url": "https://github.com/awslabs/aws-c-compression/archive/d8264e64f698341eb03039b96b4f44702a9b3f83.tar.gz"
    },
    "aws_c_event_stream": {
        "variable": "AWS_C_EVENT_STREAM",
        "version": "51bef3c44e1058b1689751539170b2e0f589ccdb",
        "build_version": "51bef3c44e1058b1689751539170b2e0f589ccdb",
        "commit": "51bef3c44e1058b1689751539170b2e0f589ccdb",
        "sha256": "b890d4940957d1ddade1a9d7de4adc37b7cb98dfdfdfec7436527fec8cb558df",
        "filename": "aws-c-event-stream-51bef3c44e1058b1689751539170b2e0f589ccdb.tar.gz",
        "env": "ARROW_AWS_C_EVENT_STREAM_URL",
        "old_version": "v0.7.0",
        "old_sha256": "88835b4c78462547917f622fd9dda45c991b7e356d9c07e2f0537d4d97fbd4fb",
        "url": "https://github.com/awslabs/aws-c-event-stream/archive/51bef3c44e1058b1689751539170b2e0f589ccdb.tar.gz"
    },
    "aws_c_http": {
        "variable": "AWS_C_HTTP",
        "version": "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4",
        "build_version": "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4",
        "commit": "8aefd899fc3210bfd0e3fd414011a3cb708bf6e4",
        "sha256": "61b1770e2e3a2bf046005cbfa8e26d621d8f266f05d03589ebee36bb4a0ac39f",
        "filename": "aws-c-http-8aefd899fc3210bfd0e3fd414011a3cb708bf6e4.tar.gz",
        "env": "ARROW_AWS_C_HTTP_URL",
        "old_version": "v0.10.13",
        "old_sha256": "d8352e7a1fb1996694a4dc31219ce03452882abf8d0858c104727f975e11b9c7",
        "url": "https://github.com/awslabs/aws-c-http/archive/8aefd899fc3210bfd0e3fd414011a3cb708bf6e4.tar.gz"
    },
    "aws_c_io": {
        "variable": "AWS_C_IO",
        "version": "28e4eb351a536daf1627bf69279e50671f7729b5",
        "build_version": "28e4eb351a536daf1627bf69279e50671f7729b5",
        "commit": "28e4eb351a536daf1627bf69279e50671f7729b5",
        "sha256": "ea51cbe20981ed34b7a8c01a60c848e4ccb716a29cd740ead07ffd61be5169c3",
        "filename": "aws-c-io-28e4eb351a536daf1627bf69279e50671f7729b5.tar.gz",
        "env": "ARROW_AWS_C_IO_URL",
        "old_version": "v0.26.3",
        "old_sha256": "521fd0848fca661130bbb7278a414d7a38bdcb9bc8ffa89f6660d84e5838a303",
        "url": "https://github.com/awslabs/aws-c-io/archive/28e4eb351a536daf1627bf69279e50671f7729b5.tar.gz"
    },
    "aws_c_mqtt": {
        "variable": "AWS_C_MQTT",
        "version": "e35b9ca3f9fcbf1a972c831c5e79046ce56959d1",
        "build_version": "e35b9ca3f9fcbf1a972c831c5e79046ce56959d1",
        "commit": "e35b9ca3f9fcbf1a972c831c5e79046ce56959d1",
        "sha256": "54d1987ce06eb7b39faace854818e9322d3df79cb310ada350beca0b12cb89cc",
        "filename": "aws-c-mqtt-e35b9ca3f9fcbf1a972c831c5e79046ce56959d1.tar.gz",
        "env": "ARROW_AWS_C_MQTT_URL",
        "old_version": "v0.15.2",
        "old_sha256": "66f3f5edff4ad1f765a86d3342b6017d0f29f950c1c24f8c1edacdc895202edc",
        "url": "https://github.com/awslabs/aws-c-mqtt/archive/e35b9ca3f9fcbf1a972c831c5e79046ce56959d1.tar.gz"
    },
    "aws_c_s3": {
        "variable": "AWS_C_S3",
        "version": "226c3e6937e3d5f7db9e10651f7bd20575e98187",
        "build_version": "226c3e6937e3d5f7db9e10651f7bd20575e98187",
        "commit": "226c3e6937e3d5f7db9e10651f7bd20575e98187",
        "sha256": "a8f33816bf7aff38467fb15ca233a33a9113be78dad47c8a7ee30d323b42ba74",
        "filename": "aws-c-s3-226c3e6937e3d5f7db9e10651f7bd20575e98187.tar.gz",
        "env": "ARROW_AWS_C_S3_URL",
        "old_version": "v0.12.0",
        "old_sha256": "1a8a8ceda0585d52028a1f3daa5861f924e7d8d2f6a17bec05813dc0b74d6eed",
        "url": "https://github.com/awslabs/aws-c-s3/archive/226c3e6937e3d5f7db9e10651f7bd20575e98187.tar.gz"
    },
    "aws_c_sdkutils": {
        "variable": "AWS_C_SDKUTILS",
        "version": "a1cc19f53b63658f1b1400b36f199eafeeb895a6",
        "build_version": "a1cc19f53b63658f1b1400b36f199eafeeb895a6",
        "commit": "a1cc19f53b63658f1b1400b36f199eafeeb895a6",
        "sha256": "f8ef9e93b30f9833d9c9ea474110c8fa2d29fd78b885546b07e68558ece3bb1c",
        "filename": "aws-c-sdkutils-a1cc19f53b63658f1b1400b36f199eafeeb895a6.tar.gz",
        "env": "ARROW_AWS_C_SDKUTILS_URL",
        "old_version": "v0.2.4",
        "old_sha256": "493cbed4fa57e0d4622fcff044e11305eb4fc12445f32c8861025597939175fc",
        "url": "https://github.com/awslabs/aws-c-sdkutils/archive/a1cc19f53b63658f1b1400b36f199eafeeb895a6.tar.gz"
    },
    "aws_checksums": {
        "variable": "AWS_CHECKSUMS",
        "version": "1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e",
        "build_version": "1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e",
        "commit": "1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e",
        "sha256": "c58d8c5319319dcade0926da44ad12effecb21372983e22835a657a32f03c9c9",
        "filename": "aws-checksums-1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e.tar.gz",
        "env": "ARROW_AWS_CHECKSUMS_URL",
        "old_version": "v0.2.10",
        "old_sha256": "cb6509f75e42ee25c372a6d379e8582ce5179e5335183842e808f7d8abb0c314",
        "url": "https://github.com/awslabs/aws-checksums/archive/1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e.tar.gz"
    },
    "s2n_tls": {
        "variable": "S2N_TLS",
        "version": "1.7.12",
        "build_version": "6459002cd61e4bf3bbc53db23f068b21da9ff6bd",
        "commit": "6459002cd61e4bf3bbc53db23f068b21da9ff6bd",
        "sha256": "94a57c4f4713cc20088b723bcdf0bb254e32a559ffcbed583e67fe3d8ddec442",
        "filename": "s2n-tls-6459002cd61e4bf3bbc53db23f068b21da9ff6bd.tar.gz",
        "env": "ARROW_S2N_TLS_URL",
        "old_version": "v1.7.0",
        "old_sha256": "a6e8228e238239bb3c17b1eda3ed702bcbb2eaebc792eac4d754cc5619b0ea06",
        "url": "https://github.com/aws/s2n-tls/archive/6459002cd61e4bf3bbc53db23f068b21da9ff6bd.tar.gz"
    },
    "thrift": {
        "variable": "THRIFT",
        "version": "0.25.0",
        "build_version": "0.25.0",
        "commit": "27e8a425ffb498e190df3a12e239326bf5ba9ed6",
        "sha256": "66da4707214c54c94bac082103dc67adaf9e08925662700f269170a7b534b214",
        "filename": "thrift-27e8a425ffb498e190df3a12e239326bf5ba9ed6.tar.gz",
        "env": "ARROW_THRIFT_URL",
        "old_version": "0.22.0",
        "old_sha256": "794a0e455787960d9f27ab92c38e34da27e8deeda7a5db0e59dc64a00df8a1e5",
        "url": "https://www.apache.org/dyn/closer.lua/thrift/0.25.0/thrift-0.25.0.tar.gz?action=download"
    },
    "grpc": {
        "variable": "GRPC",
        "version": "1.83.1",
        "build_version": "v1.83.1",
        "commit": "aae267021b1ac256f8b9038d0ef528c3798cc137",
        "sha256": "60caa8397426d8a500e3e57ab6a49d1cb5aa62a36f2591eb9da3d77fa38ad8c9",
        "filename": "grpc-v1.83.1.tar.gz",
        "env": "ARROW_GRPC_URL",
        "old_version": "v1.76.0",
        "old_sha256": "0af37b800953130b47c075b56683ee60bdc3eda3c37fc6004193f5b569758204",
        "url": "https://github.com/grpc/grpc/archive/v1.83.1.tar.gz"
    },
    "protobuf": {
        "variable": "PROTOBUF",
        "version": "35.1",
        "build_version": "v35.1",
        "commit": "35cd01f9fe9afbeea38cc7b979a3b6bfcde82c03",
        "sha256": "f0b6838e7522a8da96126d487068c959bc624926368f3024ac8fd03abd0a1ac4",
        "filename": "protobuf-35.1.tar.gz",
        "env": "ARROW_PROTOBUF_URL",
        "old_version": "v31.1",
        "old_sha256": "12bfd76d27b9ac3d65c00966901609e020481b9474ef75c7ff4601ac06fa0b82",
        "url": "https://github.com/protocolbuffers/protobuf/releases/download/v35.1/protobuf-35.1.tar.gz"
    },
    "absl": {
        "variable": "ABSL",
        "version": "20250512.1",
        "build_version": "20250512.1",
        "commit": "76bb24329e8bf5f39704eb10d21b9a80befa7c81",
        "sha256": "9b7a064305e9fd94d124ffa6cc358592eb42b5da588fb4e07d09254aa40086db",
        "filename": "absl-20250512.1.tar.gz",
        "env": "ARROW_ABSL_URL",
        "old_version": "20250127.0",
        "old_sha256": "16242f394245627e508ec6bb296b433c90f8d914f73b9c026fddb905e27276e8",
        "url": "https://github.com/abseil/abseil-cpp/archive/20250512.1.tar.gz"
    },
    "cares": {
        "variable": "CARES",
        "version": "1.34.9",
        "build_version": "1.34.9",
        "commit": "bc91829cce983a35ec01f5dbc05465858a3df469",
        "sha256": "9364b70a12eb59a06830fb488d9357f90100dff0d98b23ea3593a183d43e6f2f",
        "filename": "c-ares-1.34.9.tar.gz",
        "env": "ARROW_CARES_URL",
        "old_version": "1.34.6",
        "old_sha256": "912dd7cc3b3e8a79c52fd7fb9c0f4ecf0aaa73e45efda880266a2d6e26b84ef5",
        "url": "https://github.com/c-ares/c-ares/releases/download/v1.34.9/c-ares-1.34.9.tar.gz"
    },
    "rapidjson": {
        "variable": "RAPIDJSON",
        "version": "24b5e7a8b27f42fa16b96fc70aade9106cf7102f",
        "build_version": "24b5e7a8b27f42fa16b96fc70aade9106cf7102f",
        "commit": "24b5e7a8b27f42fa16b96fc70aade9106cf7102f",
        "sha256": "2d2601a82d2d3b7e143a3c8d43ef616671391034bc46891a9816b79cf2d3e7a8",
        "filename": "rapidjson-24b5e7a8b27f42fa16b96fc70aade9106cf7102f.tar.gz",
        "env": "ARROW_RAPIDJSON_URL",
        "old_version": "232389d4f1012dddec4ef84861face2d2ba85709",
        "old_sha256": "b9290a9a6d444c8e049bd589ab804e0ccf2b05dc5984a19ed5ae75d090064806",
        "url": "https://github.com/Tencent/rapidjson/archive/24b5e7a8b27f42fa16b96fc70aade9106cf7102f.tar.gz"
    },
    "jemalloc": {
        "variable": "JEMALLOC",
        "version": "5.3.1",
        "build_version": "5.3.1",
        "commit": "81034ce1f1373e37dc865038e1bc8eeecf559ce8",
        "sha256": "3826bc80232f22ed5c4662f3034f799ca316e819103bdc7bb99018a421706f92",
        "filename": "jemalloc-5.3.1.tar.bz2",
        "env": "ARROW_JEMALLOC_URL",
        "old_version": "5.3.0",
        "old_sha256": "2db82d1e7119df3e71b7640219b6dfe84789bc0537983c3b7ac4f7189aecfeaa",
        "url": "https://github.com/jemalloc/jemalloc/releases/download/5.3.1/jemalloc-5.3.1.tar.bz2"
    }
}
DEPENDENCIES = {
    "simdjson": {"version": "4.6.11", "commit": "f5de14f09256982933af2849beb43778bd421ca7",
                 "sha256": "9157ef4acb368d727873b4dd592f7b1d67d19fcb9201e8bc5a6b093e54e5171a",
                 "filename": "simdjson-f5de14f09256982933af2849beb43778bd421ca7.tar.gz", "env": "ARROW_SIMDJSON_URL"},
    "uriparser": {"version": "1.0.2", "commit": "9b2bed92f5deecf740819f9bf27724bee2fe9c12",
                  "sha256": "dd2e4843f43de6f5aedf430e530f1e2d159eba8ca4d64c2787af1c20f707a9e4",
                  "filename": "uriparser-1.0.2.tar.bz2", "env": "ARROW_URIPARSER_URL"},
    "boost": {"version": "1.92.0", "sha256": "f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984",
              "filename": "boost-1.92.0-cmake.tar.gz", "env": "ARROW_BOOST_URL"},
    "zlib": {"version": "1.3.2", "sha256": "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16",
             "filename": "zlib-1.3.2.tar.gz", "env": "ARROW_ZLIB_URL"},
    **{name: dict(dependency) for name, dependency in SDK_COHORT.items()},
}
SOURCE_POLICY = "arrow26-integrated-private-network-v3"
# Actual configuration remains closed until the reviewed PIC-provider helper
# has a committed source pin. Synthetic tests substitute their exact helper.
NETWORK_HELPER_SHA256 = "a8cb0355e3abfa4415093268c17678d63ea1e745a48f4a5dd750a6a01d037414"
NETWORK_ARCHIVES = ("libcurl.a", "libxml2.a", "libiconv.a", "libcharset.a", "libz.a", "libnghttp2.a")
NETWORK_HEADERS = ("curl/curl.h", "libxml2/libxml/parser.h", "iconv.h", "zlib.h", "nghttp2/nghttp2.h")
NETWORK_COMPONENTS = {
    "libcurl": ("curl", "8.23.0"), "libxml2": ("libxml2", "2.15.4"),
    "libiconv": ("libiconv", "1.19"), "libcharset": ("libiconv", "1.19"),
    "zlib": ("zlib", "1.3.2"), "libnghttp2": ("nghttp2", "1.68.0-2ubuntu0.2"),
}
TOOLCHAIN_PATCHES = (
    ("list(APPEND AWSSDK_PRODUCTS aws-lc s2n-tls)",
     "list(APPEND AWSSDK_PRODUCTS s2n-tls)\n    add_library(crypto ALIAS OpenSSL::Crypto)"),
    ("  fetchcontent_makeavailable(cares)",
     "  fetchcontent_makeavailable(cares)\n  set_target_properties(c-ares PROPERTIES UNITY_BUILD OFF)"),
    ("  fetchcontent_makeavailable(uriparser)",
     "  fetchcontent_makeavailable(uriparser)\n  set_target_properties(uriparser PROPERTIES UNITY_BUILD OFF)"),
    ("  fetchcontent_makeavailable(grpc)",
     "  set(CMAKE_UNITY_BUILD OFF)\n  fetchcontent_makeavailable(grpc)"),
    ('  add_library(RapidJSON INTERFACE IMPORTED)\n'
     '  target_include_directories(RapidJSON INTERFACE "${rapidjson_SOURCE_DIR}/include")\n'
     '  add_dependencies(RapidJSON rapidjson)',
     '  if(NOT TARGET RapidJSON)\n'
     '    add_library(RapidJSON INTERFACE IMPORTED)\n'
     '  endif()\n'
     '  target_include_directories(RapidJSON INTERFACE "$<BUILD_INTERFACE:${rapidjson_SOURCE_DIR}/include>")\n'
     '  add_dependencies(RapidJSON rapidjson)'),
    ('  fetchcontent_makeavailable(azure_sdk)\n'
     '  if(CMAKE_VERSION VERSION_LESS 3.28)',
     '  fetchcontent_makeavailable(azure_sdk)\n'
     '  if(NOT WIN32)\n'
     '    if(NOT TARGET azure-storage-common OR NOT TARGET LibXml2::LibXml2)\n'
     '      message(FATAL_ERROR "private Azure XML dependency requires expected targets")\n'
     '    endif()\n'
     '    get_target_property(_biominer_azure_alias azure-storage-common ALIASED_TARGET)\n'
     '    get_target_property(_biominer_azure_imported azure-storage-common IMPORTED)\n'
     '    get_target_property(_biominer_azure_type azure-storage-common TYPE)\n'
     '    get_target_property(_biominer_azure_source azure-storage-common SOURCE_DIR)\n'
     '    if(_biominer_azure_alias OR _biominer_azure_imported\n'
     '       OR NOT "${_biominer_azure_type}" STREQUAL "STATIC_LIBRARY"\n'
     '       OR NOT "${_biominer_azure_source}" STREQUAL "${azure_sdk_SOURCE_DIR}/sdk/storage/azure-storage-common")\n'
     '      message(FATAL_ERROR "private Azure XML dependency requires the owned static Azure target")\n'
     '    endif()\n'
     '    target_link_libraries(azure-storage-common PRIVATE LibXml2::LibXml2)\n'
     '  endif()\n'
     '  if(CMAKE_VERSION VERSION_LESS 3.28)'),
    ('set(MAKE_BUILD_ARGS "-j${NPROC}")', 'set(MAKE_BUILD_ARGS "-j1")'),
)
VERSION_PATCHES = (
    ("ARROW_SIMDJSON_BUILD_VERSION=v4.6.4", "ARROW_SIMDJSON_BUILD_VERSION=v4.6.11"),
    ("ARROW_SIMDJSON_BUILD_SHA256_CHECKSUM=b091107844fe928158c5c2265c20360fff312889ddf7ebc4528a0f0f8f2ff9cd",
     "ARROW_SIMDJSON_BUILD_SHA256_CHECKSUM=" + DEPENDENCIES["simdjson"]["sha256"]),
) + tuple(change for dependency in SDK_COHORT.values() for change in (
    (f"ARROW_{dependency['variable']}_BUILD_VERSION={dependency['old_version']}",
     f"ARROW_{dependency['variable']}_BUILD_VERSION={dependency['build_version']}"),
    (f"ARROW_{dependency['variable']}_BUILD_SHA256_CHECKSUM={dependency['old_sha256']}",
     f"ARROW_{dependency['variable']}_BUILD_SHA256_CHECKSUM={dependency['sha256']}"),
))
BOOST_SELECTION_ANCHOR = "  fetchcontent_makeavailable(boost)"

def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.absolute()), "sha256": sha256(path)}


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def check_record(record: dict, label: str = "input") -> Path:
    path = Path(record["path"])
    if not path.is_absolute() or not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"{label} checksum mismatch: {path}")
    return path


def preserve(path: Path, data: bytes, *, resume: bool, label: str) -> dict:
    if resume:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"{label} changed since preparation")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    return file_record(path)


def preserve_recipe(work: Path, source_bytes: bytes, *, resume: bool) -> dict:
    return preserve(work / "build-recipe.py", source_bytes, resume=resume, label="recipe")


def preserve_build_inputs(work: Path, inputs: dict, *, resume: bool) -> tuple:
    record = preserve(work / "build-inputs.json", canonical_json(inputs), resume=resume, label="build inputs")
    return record, "26.0.0+biominer." + record["sha256"][:12]


def archive_members(bundle: tarfile.TarFile) -> tuple:
    members, root, names = bundle.getmembers(), None, set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or member.name in names:
            raise ValueError("unsafe or duplicate archive path")
        names.add(member.name)
        root = root or path.parts[0]
        if path.parts[0] != root or not (member.isfile() or member.isdir() or member.issym()):
            raise ValueError("unsupported archive layout or member type")
        if member.issym():
            target = PurePosixPath(member.linkname)
            if target.is_absolute():
                raise ValueError("absolute archive symlink")
            normalized = os.path.normpath(str(path.parent / target))
            if normalized != root and not normalized.startswith(root + "/"):
                raise ValueError("escaping archive symlink")
    symlinks = {PurePosixPath(m.name) for m in members if m.issym()}
    if any(any(parent in symlinks for parent in PurePosixPath(m.name).parents) for m in members):
        raise ValueError("archive member traverses a symlink")
    if root is None:
        raise ValueError("empty source archive")
    return root, members


def tree_paths(root: Path) -> dict:
    result = {}
    def unreadable(error):
        raise ValueError(f"unreadable source/build tree: {error.filename}") from error
    for directory, folders, files in os.walk(root, followlinks=False, onerror=unreadable):
        for name in folders + files:
            path = Path(directory) / name
            result[path.relative_to(root).as_posix()] = path
    return result


def verify_archive_tree(bundle: tarfile.TarFile, source: Path, replacements: dict | None = None,
                        generated_files: dict | None = None) -> list:
    archive_root, members = archive_members(bundle)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source directory missing or symlink")
    expected, replacements = {}, replacements or {}
    for member in members:
        relative = PurePosixPath(member.name).relative_to(archive_root)
        if str(relative) == ".":
            continue
        expected[str(relative)] = member
        for parent in relative.parents:
            if str(parent) != ".":
                expected.setdefault(str(parent), None)
    actual = tree_paths(source)
    generated_files = generated_files or {}
    if set(generated_files) & set(expected) or set(actual) != set(expected) | set(generated_files):
        raise ValueError("source layout changed since extraction")
    evidence = []
    # Insertion order follows tar members, keeping compressed data reads forward.
    for relative, member in expected.items():
        path = actual[relative]
        if member is not None and member.issym():
            if not path.is_symlink() or os.readlink(path) != member.linkname:
                raise ValueError(f"source symlink layout changed: {relative}")
            evidence.append({"relative": relative, "type": "symlink", "target": member.linkname})
        elif member is None or member.isdir():
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"source directory layout changed: {relative}")
        else:
            stream = bundle.extractfile(member)
            original = stream.read()
            expected_bytes = replacements.get(relative, original)
            if path.is_symlink() or not path.is_file() or sha256(path) != hashlib.sha256(expected_bytes).hexdigest():
                raise ValueError(f"source changed since extraction: {relative}")
            evidence.append({"relative": relative, "type": "file", "sha256": sha256(path)})
    # Preserve canonical archive evidence and the existing generated-row suffix.
    evidence.sort(key=lambda row: row["relative"])
    for relative, data in sorted(generated_files.items()):
        path = actual[relative]
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"generated source differs from frozen builder template: {relative}")
        evidence.append({"relative": relative, "type": "generated", "sha256": sha256(path)})
    return evidence


def exact_replace(data: bytes, before: str, after: str) -> bytes:
    if data.count(before.encode()) != 1:
        raise ValueError("Arrow source transformation anchor must occur exactly once")
    return data.replace(before.encode(), after.encode())


def exact_assignment_replace(data: bytes, before: str, after: str) -> bytes:
    """Reject changed or repeated assignments, including misleading substrings."""
    name = before.split("=", 1)[0].encode()
    pattern = re.compile(rb"^" + re.escape(name) + rb"=.*$", re.MULTILINE)
    assignments = list(pattern.finditer(data))
    if len(assignments) != 1 or assignments[0].group() != before.encode():
        raise ValueError("Arrow version assignment anchor must occur exactly once and match the reviewed input")
    match = assignments[0]
    return data[:match.start()] + after.encode() + data[match.end():]


def source_transformations(bundle: tarfile.TarFile, root: str) -> dict:
    toolchain_path = "cpp/cmake_modules/ThirdpartyToolchain.cmake"
    toolchain = bundle.extractfile(root + "/" + toolchain_path).read()
    for before, after in TOOLCHAIN_PATCHES:
        toolchain = exact_replace(toolchain, before, after)
    toolchain = exact_replace(toolchain, BOOST_SELECTION_ANCHOR, BOOST_SELECTION_ANCHOR +
        '\n  file(WRITE "${CMAKE_BINARY_DIR}/biominer-boost-selection.txt" "${BOOST_INCLUDE_LIBRARIES}\\n")')
    versions_path = "cpp/thirdparty/versions.txt"
    versions = bundle.extractfile(root + "/" + versions_path).read()
    for before, after in VERSION_PATCHES:
        versions = exact_assignment_replace(versions, before, after)
    return {toolchain_path: toolchain, versions_path: versions}


def prepare_source(archive: Path, work: Path, *, resume: bool) -> tuple:
    if sha256(archive) != ARROW_ARCHIVE_SHA256:
        raise ValueError("Arrow source archive checksum mismatch")
    work = work.absolute()
    if any(path.is_symlink() for path in (work, *work.parents)):
        raise ValueError("work path contains a symlink")
    if work.exists() and (not work.is_dir() or (not resume and any(work.iterdir()))):
        raise ValueError("work directory already exists and is not empty")
    if resume:
        for name in ("cpp-build", "python-build", "arrow-prefix", "wheelhouse", "wheel-inventory",
                     "inputs", "build-receipt.json", "source-verification.json", "configured-inputs.json",
                     "resume-state.json", "logs", "private-network-helper.py", "private-network.cmake"):
            if (work / name).is_symlink():
                raise ValueError(f"resume output path is a symlink: {name}")
        receipt = work / "build-receipt.json"
        if receipt.exists():
            previous = json.loads(receipt.read_bytes())
            if previous.get("status") not in {"building", "failed", "configured_unverified"}:
                raise ValueError("resume cannot modify completed or unrecognized receipt")
        if list((work / "wheelhouse").glob("*.whl")):
            raise ValueError("resume cannot modify completed wheel output")
    source = work / f"arrow-{ARROW_COMMIT}"
    with tarfile.open(archive) as bundle:
        root, _ = archive_members(bundle)
        if root != source.name:
            raise ValueError("Arrow source archive root mismatch")
        transformed = source_transformations(bundle, root)
        for relative, expected in INTEGRATED_SOURCES.items():
            if hashlib.sha256(bundle.extractfile(root + "/" + relative).read()).hexdigest() != expected:
                raise ValueError(f"integrated Arrow source checksum mismatch: {relative}")
        if not resume:
            work.mkdir(parents=True, exist_ok=True)
            bundle.extractall(work, filter="data")
            verify_archive_tree(bundle, source)
            for relative, data in transformed.items():
                (source / relative).write_bytes(data)
        inventory = verify_archive_tree(bundle, source, transformed)
        evidence = {
            "schema_version": 1, "source_revision": ARROW_COMMIT, "archive": file_record(archive),
            "integrated_sources": [{"relative": name, "sha256": value} for name, value in sorted(INTEGRATED_SOURCES.items())],
            "transformations": [{"relative": name,
                "original_sha256": hashlib.sha256(bundle.extractfile(root + "/" + name).read()).hexdigest(),
                "sha256": hashlib.sha256(data).hexdigest()} for name, data in transformed.items()],
            "files": inventory,
        }
    preserve(work / "source-verification.json", canonical_json(evidence), resume=resume, label="source receipt")
    return source, evidence


def validate_builder(receipt_path: Path, receipt_sha256: str, *, python: str, cc: str, cxx: str) -> dict:
    check_record({"path": str(receipt_path), "sha256": receipt_sha256}, "builder receipt")
    receipt = json.loads(receipt_path.read_bytes())
    if (receipt.get("schema_version"), receipt.get("kind"), receipt.get("status")) != (
            1, "arrow26_python_builder", "frozen_verified"):
        raise ValueError("unapproved builder receipt")
    if python != receipt["python"]["path"]:
        raise ValueError("builder Python path differs from approved receipt")
    for name, selected in (("cc", cc), ("cxx", cxx)):
        if selected != receipt["toolchain"][name]["path"]:
            raise ValueError(f"compiler path differs from builder receipt: {name}")
    for name in ("python", "base_python", "requirements", "inventory", "verification"):
        check_record(receipt[name], "builder " + name)
    for name in ("cc", "cxx", "cmake", "ninja", "ld"):
        check_record(receipt["toolchain"][name], "builder toolchain " + name)
    inventory = json.loads(Path(receipt["inventory"]["path"]).read_bytes())
    root = Path(inventory["root"])
    if inventory.get("schema_version") != 1 or not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("invalid builder inventory root")
    if Path(python) != root / "bin/python" or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("builder Python is outside the approved inventory root")
    rows = {}
    for kind in ("files", "directories", "symlinks"):
        for row in inventory[kind]:
            relative = PurePosixPath(row["relative"])
            if relative.is_absolute() or ".." in relative.parts or str(relative) in rows:
                raise ValueError("invalid builder inventory path")
            rows[str(relative)] = (kind, row)
    actual = tree_paths(root) | {".": root}
    if set(actual) != set(rows):
        raise ValueError("builder inventory layout mismatch")
    base_python = Path(receipt["base_python"]["path"]).resolve()
    if Path(python).resolve() != base_python:
        raise ValueError("builder Python symlink does not resolve to approved base executable")
    for relative, (kind, row) in rows.items():
        path = actual[relative]
        if kind == "symlinks":
            if not path.is_symlink() or os.readlink(path) != row["target"]:
                raise ValueError(f"builder symlink changed: {relative}")
            resolved = path.resolve()
            if not resolved.is_relative_to(root) and not (
                    relative.startswith("bin/python") and resolved == base_python):
                raise ValueError(f"builder symlink escapes approved environment: {relative}")
        else:
            mode = stat.S_IMODE(path.lstat().st_mode)
            if path.is_symlink() or mode != row["mode"] or mode & 0o222:
                raise ValueError(f"builder inventory mode changed or writable: {relative}")
            if kind == "files":
                if not path.is_file() or sha256(path) != row["sha256"]:
                    raise ValueError(f"builder inventory file changed: {relative}")
            elif not path.is_dir():
                raise ValueError(f"builder inventory directory changed: {relative}")
    distributions = receipt["distributions"]
    actual_pins = {row["name"]: row["version"] for row in distributions}
    if actual_pins != BUILDER_PINS or len(distributions) != len(BUILDER_PINS):
        raise ValueError("builder distributions do not match exact approved pins")
    for row in distributions:
        for field in ("wheel", "metadata", "record"):
            path = check_record(row[field], "builder distribution " + field)
            if field != "wheel" and not path.resolve().is_relative_to(root):
                raise ValueError("builder distribution metadata outside frozen inventory")
        metadata = email.message_from_bytes(Path(row["metadata"]["path"]).read_bytes())
        name = re.sub(r"[-_.]+", "-", metadata["Name"].lower())
        if name != row["name"] or metadata["Version"] != row["version"]:
            raise ValueError("builder distribution metadata identity mismatch")
    return {**receipt, "receipt": file_record(receipt_path)}


def probe_builder(builder: dict) -> dict:
    probe = (
        "import sys,sysconfig,ssl,importlib.metadata as m,json; "
        "print(json.dumps({'executable':sys.executable,'base_executable':sys._base_executable,"
        "'version':list(sys.version_info[:3]),'implementation':sys.implementation.name,"
        "'gil':sys._is_gil_enabled(),'platform':sysconfig.get_platform(),'openssl':ssl.OPENSSL_VERSION,"
        "'distributions':[{'name':d.metadata['Name'],'version':d.version,'metadata':str(d._path/'METADATA')}"
        " for d in m.distributions(path=list(set([sysconfig.get_path('purelib'),sysconfig.get_path('platlib')])))]},sort_keys=True))")
    observed = json.loads(subprocess.check_output([builder["python"]["path"], "-I", "-B", "-c", probe], text=True))
    pins = {re.sub(r"[-_.]+", "-", d["name"].lower()): d["version"] for d in observed["distributions"]}
    if (observed["version"] != [3, 14, 7] or observed["implementation"] != "cpython"
            or not observed["gil"] or observed["platform"] != "linux-x86_64"
            or observed["executable"] != builder["python"]["path"]
            or Path(observed["base_executable"]).resolve() != Path(builder["base_python"]["path"]).resolve()
            or pins != BUILDER_PINS or len(observed["distributions"]) != len(BUILDER_PINS)):
        raise ValueError("observed builder Python or distribution identity mismatch")
    expected_metadata = {row["metadata"]["path"] for row in builder["distributions"]}
    if {row["metadata"] for row in observed["distributions"]} != expected_metadata:
        raise ValueError("observed builder distributions came from unexpected paths")
    return observed


def validate_network(receipt_path: Path, receipt_sha256: str, helper_path: Path, *, openssl_prefix: Path) -> dict:
    """Load only approved helper source and retain its complete verified closure."""
    helper_path, receipt_path = Path(helper_path).absolute(), Path(receipt_path).absolute()
    if any(path.is_symlink() for path in (helper_path, *helper_path.parents)):
        raise ValueError("network helper checksum path contains a symlink")
    helper_bytes = helper_path.read_bytes()
    helper_hash = hashlib.sha256(helper_bytes).hexdigest()
    if not isinstance(NETWORK_HELPER_SHA256, str) or helper_hash != NETWORK_HELPER_SHA256:
        raise ValueError("network helper checksum mismatch or committed helper pin unavailable")
    reference = {"path": str(receipt_path), "sha256": receipt_sha256}
    check_record(reference, "network receipt")
    namespace = {"__name__": "biominer_private_network_helper", "__file__": str(helper_path)}
    # Execute exactly the bytes just hashed, never a separately resolved module
    # or a timestamp-valid pyc from the mutable source directory.
    exec(compile(helper_bytes, str(helper_path), "exec"), namespace)
    verified = namespace["verify_completed_prefix"](receipt_path, receipt_sha256)
    if (set(verified) != {"receipt", "prefix", "build", "files", "private_files"}
            or verified["receipt"] != reference):
        raise ValueError("network descriptor does not bind the approved receipt")
    raw, prefix = verified["build"], Path(verified["prefix"])
    if raw != json.loads(receipt_path.read_bytes()):
        raise ValueError("network descriptor raw build differs from the approved receipt")
    if (raw.get("schema_version") != 1 or raw.get("kind") != "private_network_static_build"
            or raw.get("status") != "compiled_unverified" or raw.get("ready") is not False
            or raw.get("prefix") != str(prefix) or not prefix.is_absolute()
            or prefix != receipt_path.parent / "prefix" or prefix.resolve() != prefix):
        raise ValueError("network descriptor has an unapproved private prefix or build identity")
    if raw.get("openssl", {}).get("prefix") != str(Path(openssl_prefix).absolute()):
        raise ValueError("network descriptor selected different private OpenSSL inputs")
    files = {}
    for row in verified["files"]:
        path = check_record(row, "network closure")
        if str(path) in files:
            raise ValueError("network closure has duplicate file identities")
        files[str(path)] = row["sha256"]
    if files.get(str(receipt_path)) != receipt_sha256:
        raise ValueError("network closure omitted the approved receipt")
    for name, metadata in verified["private_files"].items():
        path = Path(name)
        identity = (metadata.get("source_package"), metadata.get("source_version"))
        if (not path.is_absolute() or not path.is_relative_to(prefix) or path.resolve() != path
                or metadata.get("component") not in NETWORK_COMPONENTS
                or identity != NETWORK_COMPONENTS[metadata["component"]]
                or files.get(name) != metadata.get("sha256")):
            raise ValueError("network private file lacks exact component/source/closure ownership")
    required = [prefix / "lib" / name for name in NETWORK_ARCHIVES]
    required += [prefix / "include" / name for name in NETWORK_HEADERS]
    if any(str(path) not in verified["private_files"] for path in required):
        raise ValueError("network closure lacks a required private archive or header")
    return {**verified, "helper": {"path": str(helper_path), "sha256": helper_hash}}


def verify_network(network: dict) -> None:
    check_record(network["helper"], "network helper")
    current = validate_network(Path(network["receipt"]["path"]), network["receipt"]["sha256"],
                               Path(network["helper"]["path"]),
                               openssl_prefix=Path(network["build"]["openssl"]["prefix"]))
    if current != network:
        raise ValueError("network descriptor or complete source/provider closure changed")


def private_network_cmake(prefix: Path, openssl: Path) -> bytes:
    """Bind global static imports before upstream find modules can omit links."""
    for path in (prefix, openssl):
        if not path.is_absolute() or any(character in str(path) for character in '\t\n\r;"$\\{}'):
            raise ValueError("network/OpenSSL path contains unsupported CMake characters")
    targets = (
        ("OpenSSL::Crypto", openssl / "lib/libcrypto.a", openssl / "include", "Threads::Threads;${CMAKE_DL_LIBS}"),
        ("OpenSSL::SSL", openssl / "lib/libssl.a", openssl / "include", "OpenSSL::Crypto"),
        ("ZLIB::ZLIB", prefix / "lib/libz.a", prefix / "include", ""),
        ("biominer_nghttp2", prefix / "lib/libnghttp2.a", prefix / "include", ""),
        ("biominer_charset", prefix / "lib/libcharset.a", prefix / "include", ""),
        ("Iconv::Iconv", prefix / "lib/libiconv.a", prefix / "include", ""),
        ("LibXml2::LibXml2", prefix / "lib/libxml2.a", prefix / "include/libxml2", "Iconv::Iconv;ZLIB::ZLIB;m"),
        ("CURL::libcurl", prefix / "lib/libcurl.a", prefix / "include", "ZLIB::ZLIB;OpenSSL::SSL;biominer_nghttp2;Threads::Threads;${CMAKE_DL_LIBS}"),
    )
    lines = ["# Recipe-owned exact private imports; no host pkg-config interfaces.",
             "include_guard(GLOBAL)", "find_package(Threads REQUIRED)"]
    for target, archive, include, dependencies in targets:
        lines += [f"if(TARGET {target})", f'  message(FATAL_ERROR "private provider target already exists: {target}")',
                  "endif()", f"add_library({target} STATIC IMPORTED GLOBAL)",
                  f"set_target_properties({target} PROPERTIES", f'  IMPORTED_LOCATION "{archive}"',
                  '  IMPORTED_LINK_INTERFACE_LANGUAGES "C"', f'  INTERFACE_INCLUDE_DIRECTORIES "{include}"',
                  f'  INTERFACE_LINK_LIBRARIES "{dependencies}")']
    lines += ['set_property(TARGET CURL::libcurl APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "CURL_STATICLIB")',
              'set_property(TARGET LibXml2::LibXml2 APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS "LIBXML_STATIC")']
    return ("\n".join(lines) + "\n").encode()


def build_environment(jobs: int, *, python: str, cc: str, cxx: str) -> dict:
    if not 1 <= jobs <= 4:
        raise ValueError("--jobs must be between 1 and 4")
    # An allowlist prevents inherited compiler/backend/dependency overrides.
    env = {name: os.environ[name] for name in ("PATH", "HOME", "SSL_CERT_FILE", "SSL_CERT_DIR") if name in os.environ}
    env.update(LANG="C.UTF-8", LC_ALL="C.UTF-8", CC=cc, CXX=cxx,
               CMAKE_BUILD_PARALLEL_LEVEL="1", MAKEFLAGS="-j1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONNOUSERSITE="1", PYARROW_BUNDLE_ARROW_CPP="ON",
               PYARROW_WITH_PARQUET_ENCRYPTION="ON", PYARROW_WITH_GANDIVA="OFF")
    for feature in FEATURES:
        env[f"PYARROW_WITH_{feature}"] = "ON"
    return env


def expected_cache(context: dict, *, stage: str) -> dict:
    expected = {"CMAKE_C_COMPILER": context["cc"], "CMAKE_CXX_COMPILER": context["cxx"]}
    expected.update({name: context["python"] for name in ("Python3_EXECUTABLE", "Python_EXECUTABLE", "PYTHON_EXECUTABLE")})
    prefix, network = Path(context["openssl_prefix"]), Path(context["network_prefix"])
    expected.update({
        "CMAKE_PROJECT_INCLUDE": str(Path(context["work"]) / "private-network.cmake"),
        "CMAKE_FIND_PACKAGE_PREFER_CONFIG": "OFF", "ZLIB_SOURCE": "SYSTEM",
        "ZLIB_LIBRARY": str(network / "lib/libz.a"), "ZLIB_LIBRARY_RELEASE": str(network / "lib/libz.a"),
        "ZLIB_LIBRARY_DEBUG": str(network / "lib/libz.a"), "ZLIB_INCLUDE_DIR": str(network / "include"),
        "CURL_NO_CURL_CMAKE": "TRUE", "CURL_USE_STATIC_LIBS": "TRUE",
        "CURL_LIBRARY": str(network / "lib/libcurl.a"), "CURL_LIBRARY_RELEASE": str(network / "lib/libcurl.a"),
        "CURL_LIBRARY_DEBUG": str(network / "lib/libcurl.a"), "CURL_INCLUDE_DIR": str(network / "include"),
        "LIBXML2_LIBRARY": str(network / "lib/libxml2.a"), "LIBXML2_INCLUDE_DIR": str(network / "include/libxml2"),
        "Iconv_LIBRARY": str(network / "lib/libiconv.a"), "Iconv_INCLUDE_DIR": str(network / "include"),
        "Iconv_IS_BUILT_IN": "FALSE", "NGHTTP2_LIBRARY": str(network / "lib/libnghttp2.a"),
        "NGHTTP2_INCLUDE_DIR": str(network / "include"), "OPENSSL_USE_STATIC_LIBS": "TRUE",
        "OPENSSL_ROOT_DIR": str(prefix), "OPENSSL_INCLUDE_DIR": str(prefix / "include"),
        "OPENSSL_CRYPTO_LIBRARY": str(prefix / "lib/libcrypto.a"), "OPENSSL_SSL_LIBRARY": str(prefix / "lib/libssl.a"),
    })
    if stage == "cpp":
        expected.update({f"ARROW_{name}": "ON" for name in FEATURES})
        expected.update({f"ARROW_WITH_{name}": "ON" for name in CODECS})
        expected.update({f"ARROW_{name}": "OFF" for name in ("TESTING", "BUILD_TESTS", "BUILD_BENCHMARKS", "BUILD_INTEGRATION", "GANDIVA")})
        expected.update({name + "_SOURCE": "BUNDLED" for name in (
            "simdjson", "uriparser", "Boost", "AWSSDK", "gRPC", "Protobuf",
            "absl", "c-ares", "Thrift", "jemalloc", "RapidJSON",
        )})
        expected.update(ARROW_BUILD_SHARED="ON", ARROW_BUILD_STATIC="OFF", ARROW_DEPENDENCY_SOURCE="AUTO",
                        ARROW_DEPENDENCY_USE_SHARED="OFF", ARROW_JEMALLOC="ON", ARROW_MIMALLOC="ON",
                        PARQUET_REQUIRE_ENCRYPTION="ON", URIPARSER_BUILD_TESTS="OFF")
    elif stage == "python":
        expected.update({"PYARROW_" + name: "ON" for name in PYTHON_FEATURES})
        expected.update(PYARROW_GANDIVA="OFF", PYARROW_BUNDLE_ARROW_CPP="ON")
        for package in ("Arrow", "ArrowCompute", "ArrowAcero", "ArrowDataset", "ArrowFlight", "ArrowS3", "ArrowSubstrait", "Parquet"):
            expected[package + "_DIR"] = str(Path(context["work"]) / "arrow-prefix/lib/cmake" / package)
    else:
        raise ValueError("unknown CMake cache stage")
    return expected


def verify_zlib_cache_alias(path: Path, context: dict) -> None:
    """Bind the ORC cache normalization to the exact private imported archive."""
    try:
        work = Path(context["work"])
        build = work / "cpp-build"
        if path != build / "CMakeCache.txt":
            raise ValueError("cache is outside the selected C++ build")
        manifest = work / "build-inputs.json"
        if manifest.is_symlink():
            raise ValueError("build input manifest is a symlink")
        inputs = json.loads(manifest.read_bytes())
        if (inputs.get("schema_version") != 1 or inputs.get("source_policy") != SOURCE_POLICY
                or inputs.get("arrow_commit") != ARROW_COMMIT or inputs.get("context") != context):
            raise ValueError("build input identity differs from the requested configuration")
        network = inputs["network"]
        if (network["prefix"] != context["network_prefix"]
                or network["build"]["openssl"]["prefix"] != context["openssl_prefix"]):
            raise ValueError("private network/OpenSSL prefix differs from the requested configuration")
        verify_network(network)
        metadata = work / "private-network.cmake"
        if (metadata.is_symlink() or inputs["provider_cmake"] != file_record(metadata)
                or metadata.read_bytes() != private_network_cmake(Path(network["prefix"]), Path(context["openssl_prefix"]))):
            raise ValueError("private imported-target metadata differs from its pinned definition")
        archive = Path(network["prefix"]) / "lib/libz.a"
        provider = native_input_provider(archive, None, network=network)
        if ({key: provider.get(key) for key in ("kind", "component", "source_package", "source_version")}
                != {"kind": "private_network", "component": "zlib", "source_package": "zlib", "source_version": "1.3.2"}):
            raise ValueError("zlib archive lacks exact private source ownership")
        source = work / f"arrow-{ARROW_COMMIT}/cpp"
        targets, _ = capture_cmake_targets(build, source)
        selected = [row for row in targets if row["target"] == "ZLIB::ZLIB"]
        if len(selected) != 1:
            raise ValueError("one scoped ZLIB::ZLIB target is required")
        target = selected[0]
        original = json.loads(check_record(target["record"], "ZLIB_LIBRARY target").read_bytes())
        if (target["kind"] != "STATIC_LIBRARY" or target["imported"] is not True
                or target["local"] is not False or original.get("abstract") is not True
                or target["source"] != str(source) or target["binary"] != str(build)
                or target["artifacts"] != [str(archive)]):
            raise ValueError("ZLIB::ZLIB is not the global imported abstract private static archive")
    except (KeyError, TypeError, OSError, ValueError) as error:
        raise ValueError(f"cpp CMake cache rejected ZLIB_LIBRARY alias binding: {error}") from error


def verify_cache(path: Path, context: dict, *, stage: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlink {stage} CMake cache")
    cache = {}
    for line in path.read_text().splitlines():
        match = re.match(r"([^/#][^:]*):[^=]+=(.*)$", line)
        if match:
            if match[1] in cache:
                raise ValueError(f"duplicate CMake cache key: {match[1]}")
            cache[match[1]] = match[2]
    for name, expected in expected_cache(context, stage=stage).items():
        actual = cache.get(name)
        if stage == "cpp" and name == "ZLIB_LIBRARY" and actual == "ZLIB::ZLIB":
            verify_zlib_cache_alias(path, context)
            continue
        if actual != expected:
            raise ValueError(f"{stage} CMake cache rejected {name}: {actual!r}, expected {expected!r}")
    if cache.get("CMAKE_GENERATOR") not in (None, "Ninja"):
        raise ValueError("CMake generator is not bounded Ninja")
    return cache


def build_commands(source: Path, work: Path, prefix: Path, *, network_prefix: Path, python: str, cc: str, cxx: str, jobs: int,
                   cmake: str = "cmake", ninja: str = "ninja") -> dict:
    build_environment(jobs, python=python, cc=cc, cxx=cxx)
    context = {"python": python, "cc": cc, "cxx": cxx, "openssl_prefix": str(prefix),
               "network_prefix": str(network_prefix), "work": str(work)}
    flags = expected_cache(context, stage="cpp") | {
        "CMAKE_PREFIX_PATH": str(network_prefix) + ";" + str(prefix), "CMAKE_BUILD_TYPE": "Release", "CMAKE_INSTALL_LIBDIR": "lib",
        "CMAKE_INSTALL_PREFIX": str(work / "arrow-prefix"), "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_UNITY_BUILD": "ON", "ARROW_RPATH_ORIGIN": "ON", "CMAKE_MAKE_PROGRAM": ninja,
        "CMAKE_FIND_USE_PACKAGE_REGISTRY": "OFF", "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY": "OFF",
        "CMAKE_SHARED_LINKER_FLAGS": "-Wl,-Map,%", "CMAKE_MODULE_LINKER_FLAGS": "-Wl,-Map,%",
    }
    python_flags = expected_cache(context, stage="python") | {
        "CMAKE_PREFIX_PATH": ";".join(map(str, (work / "arrow-prefix", network_prefix, prefix))),
        "CMAKE_MAKE_PROGRAM": ninja, "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_SHARED_LINKER_FLAGS": "-Wl,-Map,%", "CMAKE_MODULE_LINKER_FLAGS": "-Wl,-Map,%",
        "CMAKE_FIND_USE_PACKAGE_REGISTRY": "OFF", "CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY": "OFF",
    }
    return {
        "cpp_configure": [cmake, "-S", str(source / "cpp"), "-B", str(work / "cpp-build"), "-G", "Ninja",
                          *[f"-D{name}={value}" for name, value in flags.items()]],
        "cpp_build": [cmake, "--build", str(work / "cpp-build"), "--target", "install", "--parallel", str(jobs)],
        "python_build": [python, "-I", "-B", "-m", "build", "--wheel", "--no-isolation", "--outdir", str(work / "wheelhouse"),
                         f"--config-setting=build-dir={work / 'python-build'}", f"--config-setting=build.tool-args=-j{jobs}",
                         *[f"--config-setting=cmake.define.{name}={value}" for name, value in python_flags.items()], str(source / "python")],
    }


def tree_inventory(root: Path) -> list:
    rows = []
    for relative, path in sorted(tree_paths(root).items()):
        if path.is_symlink():
            rows.append({"relative": relative, "type": "symlink", "target": os.readlink(path)})
        elif path.is_file():
            rows.append({"relative": relative, "type": "file", "sha256": sha256(path)})
        elif path.is_dir():
            rows.append({"relative": relative, "type": "directory"})
        else:
            raise ValueError(f"unsupported source/build object: {path}")
    return rows


def system_provider(path: Path) -> dict:
    ownership = subprocess.run(["/usr/bin/dpkg-query", "-S", str(path)], capture_output=True, text=True)
    if ownership.returncode:
        raise ValueError(f"unowned system native input: {path}")
    packages = {line.split(": ", 1)[0] for line in ownership.stdout.splitlines() if ": " in line}
    if len(packages) != 1:
        raise ValueError(f"ambiguous system native input provider: {path}")
    package = next(iter(packages))
    fields = subprocess.check_output([
        "/usr/bin/dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}", package], text=True).split("\t")
    if len(fields) != 4 or not all(fields):
        raise ValueError(f"missing system source-package/version identity: {path}")
    return {"package": fields[0], "version": fields[1], "source_package": fields[2], "source_version": fields[3],
            "source_and_advisory_review": "required_before_runtime_admission"}


def validate_python_runtime(base_python: Path) -> dict:
    """Reuse the exact standalone runtime attestation, including native inputs."""
    prefix = base_python.resolve().parent.parent
    receipt = prefix.parent / "receipt.json"
    check_record({"path": str(receipt), "sha256": "43a7e96c319752ec519a3c7e213cbb063bb8d140bbb598e69ceb6b49b2c787f4"},
                 "Python runtime source receipt")
    runtime = json.loads(receipt.read_bytes())
    if (runtime.get("schema_version") != 1 or runtime.get("source_build") != "Astral20260929"
            or runtime.get("identity") != {"version": "3.14.7", "openssl_version": "3.5.9"}
            or runtime.get("python") != file_record(base_python)
            or runtime.get("archive", {}).get("sha256") != "da5357f47b1d9c8d4c439004d463d893ef688a44c27438a62b47073d26d1377e"):
        raise ValueError("Python runtime source identity rejected")
    check_record(runtime["archive"], "Python runtime source archive")
    files = []
    for row in runtime["verified_archive_files"]:
        path = check_record(row, "Python runtime archive member")
        if not path.is_relative_to(prefix):
            raise ValueError("Python runtime source member escapes approved prefix")
        if path.is_relative_to(prefix / "include") or re.search(r"\.(a|so(?:\.[0-9]+)*)$", path.name):
            files.append(row)
    return {"source_build": runtime["source_build"], "receipt": file_record(receipt),
            "archive": runtime["archive"], "python": runtime["python"], "files": files}


def protected_native_name(path: Path) -> bool:
    return bool(re.fullmatch(r"lib(?:curl(?:-[^.]+)?|xml2|z|iconv|charset|ssl|crypto|nghttp2)"
                             r"(?:\.a|\.so(?:\.[0-9]+)*)", path.name))


def native_input_provider(path: Path, python_runtime: dict | None, *, network: dict | None = None,
                          openssl: dict | None = None) -> dict:
    path = path.resolve()
    if network:
        metadata = network["private_files"].get(str(path))
        if metadata:
            reference = {"path": str(path), "sha256": metadata["sha256"]}
            check_record(reference, "network private input")
            check_record(network["receipt"], "network receipt")
            check_record(network["helper"], "network helper")
            if network["helper"]["sha256"] != NETWORK_HELPER_SHA256:
                raise ValueError("network helper checksum is not the approved source pin")
            return {"kind": "private_network", "helper": network["helper"], "receipt": network["receipt"],
                    "file": reference, **{name: metadata[name] for name in ("component", "source_package", "source_version")}}
        if path.is_relative_to(Path(network["prefix"])):
            raise ValueError(f"unapproved private network provider: {path}")
    if openssl:
        for row in openssl["static_archives"]:
            if Path(row["path"]).resolve() == path:
                check_record(row, "private OpenSSL archive")
                check_record(openssl["receipt"], "private OpenSSL receipt")
                return {"kind": "private_openssl", "version": "3.6.5", "receipt": openssl["receipt"],
                        "source_commit": openssl["source_commit"]}
    if python_runtime:
        for row in python_runtime["files"]:
            if Path(row["path"]).resolve() == path.resolve():
                check_record(row, "Python native input")
                check_record(python_runtime["receipt"], "Python runtime source receipt")
                return {"kind": "python_runtime", "source_build": python_runtime["source_build"],
                        "receipt": python_runtime["receipt"]}
    if protected_native_name(path):
        raise ValueError(f"unapproved network/crypto provider: {path}")
    return system_provider(path)


def capture_cmake_targets(build: Path, expected_source: Path) -> tuple[list, list]:
    """Use CMake's scoped target identities and already evaluated link inputs."""
    reply = build / ".cmake/api/v1/reply"
    if reply.resolve() != reply:
        raise ValueError("File API reply directory contains a symlink")
    indexes = list(reply.glob("index-*.json"))
    if len(indexes) != 1:
        raise ValueError("File API requires one current successful index")
    records = {}

    def read(path):
        if path.parent != reply or path.is_symlink() or not path.is_file():
            raise ValueError("File API reply reference is missing or escapes its directory")
        records[str(path)] = file_record(path)
        try:
            return json.loads(path.read_bytes())
        except (ValueError, UnicodeError) as error:
            raise ValueError("File API reply is not valid JSON") from error

    def referenced(item):
        name = item.get("jsonFile")
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("File API reply reference is invalid")
        return reply / name

    index = read(indexes[0])
    model = read(referenced(index.get("reply", {}).get("codemodel-v2", {})))
    version = model.get("version", {})
    if (model.get("kind") != "codemodel" or version.get("major") != 2
            or type(version.get("minor")) is not int or version["minor"] < 9
            or model.get("paths") != {"source": str(expected_source), "build": str(build)}):
        raise ValueError("File API codemodel version or source/build identity rejected")
    configurations = model.get("configurations", [])
    if len(configurations) != 1:
        raise ValueError("File API requires a single Ninja configuration")
    configuration = configurations[0]
    rows, identities = [], set()
    for entry in configuration.get("targets", []) + configuration.get("abstractTargets", []):
        path = referenced(entry)
        target = read(path)
        identity = target.get("id")
        if (not isinstance(identity, str) or not identity or identity in identities
                or identity != entry.get("id") or target.get("name") != entry.get("name")):
            raise ValueError("File API target identity is missing, duplicated or mismatched")
        identities.add(identity)
        paths = target.get("paths", {})
        if not all(isinstance(paths.get(key), str) for key in ("source", "build")):
            raise ValueError("File API target source/build paths are missing")
        artifacts = [str((build / item["path"]).absolute()) for item in target.get("artifacts", [])]
        rows.append({
            "target": target["name"], "id": identity, "kind": target["type"],
            "source": str((expected_source / paths["source"]).absolute()),
            "binary": str((build / paths["build"]).absolute()),
            "artifact": artifacts[0] if artifacts else "", "artifacts": artifacts,
            "imported": target.get("imported", False), "local": target.get("local", False),
            "includes": [item for group in target.get("compileGroups", []) for item in group.get("includes", [])],
            "links": target.get("link", {}).get("commandFragments", []),
            "interface_links": target.get("interfaceLinkLibraries", []),
            "record": file_record(path),
        })
    if not rows:
        raise ValueError("File API target inventory is empty")
    return sorted(rows, key=lambda row: row["id"]), [records[name] for name in sorted(records)]


def capture_dependencies(work: Path, *, compiled: bool) -> dict:
    build = work / "cpp-build"
    inputs = json.loads((work / "build-inputs.json").read_bytes())
    network = inputs.get("network")
    if network:
        verify_network(network)
    if (build / "zlib_ep").exists():
        raise ValueError("unapproved second bundled zlib provider selected")
    targets, file_api = capture_cmake_targets(build, work / f"arrow-{ARROW_COMMIT}/cpp")
    boost = build / "biominer-boost-selection.txt"
    if not boost.is_file():
        raise ValueError("actual Boost component selection missing")
    components = sorted(set(boost.read_text().strip().split(";")))
    required_boost = {"algorithm", "crc", "numeric/conversion", "scope_exit", "throw_exception", "tokenizer", "locale", "thread", "uuid"}
    if not required_boost.issubset(components):
        raise ValueError("actual Boost component selection lacks required components")
    static_archives, native_inputs, source_roots = {}, {}, set()
    for row in targets:
        target, source = row["target"], row["source"]
        if re.search(r"(^|::)(gtest|gmock)(_|$)", target, re.I) or "aws-lc" in source:
            raise ValueError(f"unapproved testing/crypto dependency selected: {target}")
        candidates = {Path(value) for value in row["artifacts"]}
        for fragment in row["links"]:
            if fragment.get("role") == "libraries":
                for value in shlex.split(fragment["fragment"]):
                    if not value.startswith("-") and re.search(r"\.(a|so(?:\.[0-9]+)*)$", value):
                        candidates.add((build / value).absolute())
        for candidate in sorted(candidates):
            if not re.search(r"\.(a|so(?:\.[0-9]+)*)$", candidate.name) or not candidate.is_file():
                continue
            item = file_record(candidate)
            if protected_native_name(candidate.resolve()) or not candidate.resolve().is_relative_to(work):
                item["provider"] = native_input_provider(candidate.resolve(), inputs.get("python_runtime"),
                                                         network=network, openssl=inputs.get("openssl"))
            native_inputs[str(candidate)] = item
            if candidate.name.endswith(".a"):
                static_archives[str(candidate)] = item
        source_path = Path(source)
        if source_path.is_dir() and source_path.is_relative_to(build):
            source_roots.add(source_path)
    # Capture each fetched source tree once, including headers used by interface
    # targets. Keep configure-generated changes explicit, rather than guessing
    # that upstream source versions describe a system-selected package.
    source_roots.update(path for path in (build / "_deps").glob("*-src") if path.is_dir())
    source_roots = {path for path in source_roots if not any(parent in source_roots for parent in path.parents)}
    forced = {}
    for name in ("simdjson", "uriparser", "boost"):
        source = build / "_deps" / (name + "-src")
        if not source.is_dir():
            raise ValueError(f"required bundled {name} source missing")
        with tarfile.open(work / "inputs" / DEPENDENCIES[name]["filename"]) as archive:
            forced[name] = verify_archive_tree(archive, source)
    if network:
        selected = {str(Path(row["path"]).resolve()) for row in static_archives.values()}
        if any(str(Path(network["prefix"]) / "lib" / name) not in selected for name in NETWORK_ARCHIVES):
            raise ValueError("configured target inventory omitted a required private network static archive")
    versions = (work / f"arrow-{ARROW_COMMIT}/cpp/thirdparty/versions.txt").read_text()
    reviewed_hashes = set(re.findall(r"_SHA256_CHECKSUM=([0-9a-f]{64})", versions))
    archives = []
    for path in build.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.name.endswith((".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip")):
            item = file_record(path)
            item["matches_arrow_versions_pin"] = item["sha256"] in reviewed_hashes
            archives.append(item)
    return {"schema_version": 1, "stage": "compiled" if compiled else "configured", "targets": targets,
            "private_network": {name: network[name] for name in ("receipt", "helper", "prefix")} if network else None,
            "file_api": file_api, "native_inputs": sorted(native_inputs.values(), key=lambda row: row["path"]),
            "boost_components": components, "boost_selection": file_record(boost),
            "static_archives": sorted(static_archives.values(), key=lambda row: row["path"]),
            "source_trees": [{"path": str(path), "files": tree_inventory(path)} for path in sorted(source_roots)],
            "forced_bundled_source_verification": forced, "downloaded_archives": archives,
            "coverage": "CMake File API 2.9 includes build and scoped imported targets plus resolved link fragments. Private network inputs are receipt/source-bound. Bare -l flags require final linker-map resolution. Targets are a superset; final maps establish actual linkage. Strict archive/tree proof currently covers simdjson, uriparser and Boost; expanded SDK and deferred ExternalProject source verification remains required before runtime admission. Other system source/advisory review remains required."}


def verify_configured_selection(configured: dict, current: dict) -> None:
    """Keep configured inputs fixed while permitting checkpointed build outputs.

    The caller verifies the complete, explicitly hash-approved resume state
    before this comparison. New archives/generated files are therefore retained
    evidence, not an exemption from resume integrity checking.
    """
    for name in ("targets", "boost_components", "boost_selection", "forced_bundled_source_verification"):
        if configured[name] != current[name]:
            raise ValueError(f"configured dependency selection changed: {name}")
    if configured.get("file_api") != current.get("file_api"):
        raise ValueError("configured File API evidence changed")
    if configured.get("private_network") != current.get("private_network"):
        raise ValueError("configured private network provider changed")
    for name in ("static_archives", "native_inputs", "downloaded_archives"):
        observed = {row["path"]: row for row in current.get(name, [])}
        for row in configured.get(name, []):
            if observed.get(row["path"]) != row:
                raise ValueError(f"configured dependency input changed: {row['path']}")
    observed_trees = {row["path"]: row for row in current["source_trees"]}
    for tree in configured["source_trees"]:
        current_tree = observed_trees.get(tree["path"])
        if current_tree is None:
            raise ValueError(f"configured source tree disappeared: {tree['path']}")
        files = {row["relative"]: row for row in current_tree["files"]}
        for row in tree["files"]:
            if files.get(row["relative"]) != row:
                raise ValueError(f"configured source changed: {tree['path']}/{row['relative']}")


def inspect_wheel(work: Path, build_record: dict) -> dict:
    if file_record(work / "build-inputs.json") != build_record.get("build_inputs"):
        raise ValueError("immutable build manifest changed during compilation")
    wheels = list((work / "wheelhouse").glob("*.whl"))
    if len(wheels) != 1 or wheels[0].is_symlink():
        raise ValueError("expected exactly one regular Arrow wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate wheel member")
        for info in bundle.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("unsafe wheel member")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA") and len(PurePosixPath(name).parts) == 2]
        if len(metadata_names) != 1:
            raise ValueError("wheel distribution metadata is ambiguous")
        metadata = email.message_from_bytes(bundle.read(metadata_names[0]))
        if metadata["Name"] != "pyarrow" or metadata["Version"] != build_record["wheel_version"]:
            raise ValueError("wheel version does not match immutable build inputs")
        elfs = [name for name in names if not name.endswith("/") and bundle.open(name).read(4) == b"\x7fELF"]
        if not elfs:
            raise ValueError("wheel contains no native ELF")
        for name in REQUIRED_ARROW_LIBRARIES:
            if "pyarrow/" + name not in elfs:
                raise ValueError(f"required wheel native ELF missing: {name}")
        wheel_metadata = bundle.read(metadata_names[0].replace("METADATA", "WHEEL")).decode()
        if "Tag: cp314-cp314-linux_x86_64" not in wheel_metadata:
            raise ValueError("wheel Python/ABI/platform tag is not the approved cp314 Linux build")
        unpack = work / "wheel-inventory"
        unpack.mkdir()
        bundle.extractall(unpack)
    native_files, linkage = {}, {}
    wheel_paths = {(unpack / name).resolve(): name for name in elfs}
    for path, member in wheel_paths.items():
        native_files[str(path)] = {"path": "{generation}/env/lib/python3.14/site-packages/" + member,
                                   "sha256": sha256(path), "owner": "wheel:pyarrow"}
        result = subprocess.run(["/usr/bin/ldd", str(path)], capture_output=True, text=True, check=True,
                                env={"PATH": "/usr/bin:/bin", "LANG": "C"})
        if "not found" in result.stdout:
            raise ValueError(f"unresolved ELF dependency: {member}")
        linkage[member] = result.stdout
        dependencies = re.findall(r"=> (/\S+)", result.stdout) + re.findall(r"^\s*(/\S+) \(", result.stdout, re.M)
        for dependency in dependencies:
            resolved = Path(dependency).resolve()
            if resolved in wheel_paths or str(resolved) in native_files:
                continue
            provider = native_input_provider(resolved, build_record.get("python_runtime"),
                                             network=build_record.get("network"), openssl=build_record.get("openssl"))
            owner = ("python-runtime:" + provider["source_build"] if provider.get("kind") == "python_runtime"
                     else "private-network:" + provider["component"] if provider.get("kind") == "private_network"
                     else "private-openssl:3.6.5" if provider.get("kind") == "private_openssl"
                     else "dpkg:" + provider["package"])
            native_files[str(resolved)] = file_record(resolved) | {"owner": owner, "provider": provider}
    return {"wheel": file_record(wheel), "version": metadata["Version"],
            "native_files": list(native_files.values()), "linkage": linkage}


def write_resume_state(work: Path) -> dict:
    path = work / "resume-state.json"
    number = 2
    while path.exists() or path.is_symlink():
        path = work / f"resume-state-{number}.json"
        number += 1
    files = [row for row in tree_inventory(work) if row["relative"] != "build-receipt.json"]
    return preserve(path, canonical_json({"schema_version": 1, "files": files}), resume=False, label="resume state")


def verify_resume_state(work: Path, state: dict | None = None) -> None:
    path = check_record(state, "resume state") if state else work / "resume-state.json"
    if path.is_symlink() or path.parent != work or not path.is_file():
        raise ValueError("resume state must be a regular file in the owned work directory")
    value = json.loads(path.read_bytes())
    current = [row for row in tree_inventory(work) if row["relative"] not in ("build-receipt.json", path.name)]
    if value.get("schema_version") != 1 or value.get("files") != current:
        raise ValueError("resume state changed: intermediate outputs or inputs differ")


def verify_link_maps(work: Path, prefix: Path, dependencies: dict, *, network: dict) -> dict:
    verify_network(network)
    inputs_path = work / "build-inputs.json"
    inputs = json.loads(inputs_path.read_bytes()) if inputs_path.is_file() else {}
    maps, final_maps, linked, contributed = [], [], set(), set()
    closure = {row["path"]: row for row in network["files"]}
    required = [prefix / "lib/libcrypto.a", prefix / "lib/libssl.a"]
    required += [Path(network["prefix"]) / "lib" / name for name in NETWORK_ARCHIVES if name != "libcharset.a"]
    for path in required:
        if str(path) not in closure:
            raise ValueError(f"private network closure omitted required static input: {path.name}")
        check_record(closure[str(path)], "network static link input")
    for path in sorted(work.rglob("*.map")):
        if path.is_symlink() or not path.is_file():
            continue
        text = path.read_text(errors="replace")
        if "Linker script and memory map" not in text:
            continue
        maps.append(file_record(path))
        if (not (path.is_relative_to(work / "cpp-build") or path.is_relative_to(work / "python-build"))
                or any(part in {"CMakeScratch", "CMakeTmp", "_deps"} for part in path.relative_to(work).parts)
                or not re.fullmatch(r".+\.so(?:\.[0-9]+)*\.map", path.name)):
            continue
        if path.resolve() != path:
            raise ValueError("final linker map path contains a symlink")
        final_maps.append(file_record(path))
        stage_build = work / ("python-build" if path.is_relative_to(work / "python-build") else "cpp-build")
        for archive in re.findall(r"(?:^|\s)([^\s()]+\.a)\([^\s()]+\)", text, re.M):
            contributed.add(str((stage_build / archive).resolve()))
        libraries = re.findall(r"(?:^|\s)([^\s()]+(?:\.a|\.so(?:\.[0-9]+)*))(?=[\s()]|$)", text, re.M)
        for name in libraries:
            library = (stage_build / name).resolve()
            if protected_native_name(library):
                if library in (prefix / "lib/libssl.a", prefix / "lib/libcrypto.a"):
                    check_record(closure[str(library)], "network private OpenSSL link input")
                else:
                    native_input_provider(library, inputs.get("python_runtime"), network=network,
                                          openssl=inputs.get("openssl"))
            if library.suffix == ".a":
                linked.add(str(library))
    if not maps:
        raise ValueError("build produced no per-output linker maps")
    resolved = linked
    for path in required:
        if str(path.resolve()) not in resolved:
            raise ValueError(f"final link maps do not prove actual static linkage: {path.name}")
        if str(path.resolve()) not in contributed:
            raise ValueError(f"final link maps do not prove a consumed archive member: {path.name}")
    if len([name for name in resolved if Path(name).name == "libz.a"]) != 1:
        raise ValueError("exactly one approved private libz.a required in final linkage")
    known = {str(Path(row["path"]).resolve()): row for row in dependencies["static_archives"]}
    records = []
    for name in sorted(resolved):
        path = Path(name)
        if name in known:
            check_record(known[name], "linked static input")
            item = known[name]
        else:
            item = file_record(path)
            if not path.is_relative_to(work) and not path.is_relative_to(prefix):
                item["provider"] = native_input_provider(path, inputs.get("python_runtime"),
                                                         network=network, openssl=inputs.get("openssl"))
        if name in network["private_files"]:
            item = {**item, "provider": native_input_provider(path, inputs.get("python_runtime"), network=network,
                                                             openssl=inputs.get("openssl"))}
        records.append(item)
    return {"maps": maps, "final_maps": final_maps, "linked_static_archives": records,
            "qualification": "Every required private archive contributes an object member in final Arrow/Python consumer maps, excluding configure probes; LOAD entries alone do not establish consumption. Final ELF closure and all other system source/provider reviews remain required."}


def validate_openssl(prefix: Path, source_archive: Path) -> dict:
    header = prefix / "include/openssl/opensslv.h"
    if not header.is_file() or '"OpenSSL 3.6.5 ' not in header.read_text():
        raise ValueError("OpenSSL 3.6.5 required")
    check_record({"path": str(header), "sha256": "80d9b9534fdbdab12c29cb8c62887ede14bbb7112b93e838a63d6815400f5f0b"}, "OpenSSL header")
    expected = {"libcrypto.a": "e2d8161275df2868b63e28d8d1f5b49efa3e40a3697d1a21a07890c58a2d3bc2",
                "libssl.a": "3ac9b8f49ce67cc03e9a1073754faaec92226195d610bd59800e4df5c41a82b6"}
    archives = [file_record(check_record({"path": str(prefix / "lib" / name), "sha256": value}, "approved static OpenSSL archive"))
                for name, value in expected.items()]
    receipt = prefix.parent / "openssl-build-receipt.json"
    check_record({"path": str(receipt), "sha256": "66d41e5fa8cee1cdb865a7df41620685776c96fd8eba9abcc6a70505ca3da13a"}, "OpenSSL source receipt")
    check_record({"path": str(source_archive), "sha256": "f3091b7ccb41a11d877b7406b7afc9dd9c5003d299c6bb33cff882127bc06c91"}, "OpenSSL source archive")
    return {"version": "3.6.5", "source_commit": "c8bd5a57108599ac650bbae77fcabe3109dab2e8",
            "receipt": file_record(receipt), "source_archive": file_record(source_archive),
            "header": file_record(header), "headers": tree_inventory(prefix / "include"), "static_archives": archives}


def expected_generated_version(python: str, version: str) -> bytes:
    # This private API is bound to the exact frozen vcs-versioning 2.6.0 wheel.
    # Formatting prints bytes and does not write into or import the Arrow tree.
    code = (
        "import sys,base64; from vcs_versioning._dump_version import DEFAULT_TEMPLATES,DummyScmVersion; "
        "from vcs_versioning._version_cls import _version_as_tuple; "
        "v=sys.argv[1]; content=DEFAULT_TEMPLATES['.py'].format(version=v,version_tuple=_version_as_tuple(v),"
        "scm_version=DummyScmVersion()); print(base64.b64encode(content.encode()).decode())")
    return base64.b64decode(subprocess.check_output([python, "-I", "-B", "-c", code, version], text=True).strip(), validate=True)


def compile_source_evidence(work: Path) -> dict:
    path = work / "cpp-build/compile_commands.json"
    entries = json.loads(path.read_bytes())
    expected = {work / f"arrow-{ARROW_COMMIT}/cpp/src/arrow/compute/kernels" / name
                for name in ("vector_replace.cc", "scalar_if_else.cc")}
    found, units = set(), []
    for row in entries:
        source = Path(row["file"])
        if not source.is_absolute():
            source = Path(row["directory"]) / source
        if source in expected:
            found.add(source)
            units.append(file_record(source))
        elif "Unity" in source.parts and source.is_file():
            includes = {Path(name) for name in re.findall(r'^#include "([^"]+)"', source.read_text(), re.M)}
            relevant = includes & expected
            if relevant:
                found.update(relevant)
                units.append(file_record(source))
    if found != expected:
        raise ValueError("compile commands do not bind both integrated header consumers")
    return {"compile_commands": file_record(path), "translation_units": units,
            "integrated_header_consumers": [file_record(path) for path in sorted(found)]}


def run_stage(work: Path, name: str, command: list, env: dict) -> dict:
    logs = work / "logs"
    logs.mkdir(exist_ok=True)
    number = 1
    while (logs / f"{name}-{number}.log").exists():
        number += 1
    path = logs / f"{name}-{number}.log"
    with path.open("x") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return {"command": command, "log": file_record(path), "exit_code": 0}


def run_network_stage(work: Path, name: str, command: list, env: dict, network: dict, *, frozen_inputs: tuple = ()) -> dict:
    def guard():
        verify_network(network)
        for reference in frozen_inputs:
            check_record(reference, "immutable build input")
    guard()
    try:
        return run_stage(work, name, command, env)
    finally:
        guard()


def main(argv: list | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    recipe_bytes = Path(__file__).read_bytes()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("openssl-prefix", "archive", "work", "builder-receipt", "network-receipt", "network-helper"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in DEPENDENCIES:
        parser.add_argument("--" + name.replace("_", "-") + "-archive", required=True, type=Path)
    for name in ("builder-receipt-sha256", "network-receipt-sha256", "python", "cc", "cxx"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--openssl-source-archive", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--check", action="store_true", help="verify approved inputs without creating a build generation")
    parser.add_argument("--configure-only", action="store_true", help="stop for review of actual native dependency selection")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-state-sha256")
    parser.add_argument("--configured-inputs-sha256")
    args = parser.parse_args(argv)
    work, archive, prefix = args.work.absolute(), args.archive.absolute(), args.openssl_prefix.absolute()
    builder, record, return_code = None, None, 1
    try:
        env = build_environment(args.jobs, python=args.python, cc=args.cc, cxx=args.cxx)
        if work.is_relative_to("/tmp") or work.is_relative_to("/var/tmp"):
            raise ValueError("build work must use persistent storage outside /tmp and /var/tmp")
        if any(character in str(work) for character in "\t\n\r{}"):
            raise ValueError("work path contains unsupported CMake/template characters")
        if not args.check and not args.configure_only and (not args.resume or not args.configured_inputs_sha256):
            raise ValueError("compilation requires --resume and reviewed --configured-inputs-sha256")
        builder = validate_builder(args.builder_receipt.absolute(), args.builder_receipt_sha256,
                                   python=args.python, cc=args.cc, cxx=args.cxx)
        observed_builder = probe_builder(builder)
        python_runtime = validate_python_runtime(Path(builder["base_python"]["path"]))
        openssl_archive = args.openssl_source_archive.absolute() if args.openssl_source_archive else prefix.parent / "downloads/openssl.tar.gz"
        openssl = validate_openssl(prefix, openssl_archive)
        network = validate_network(args.network_receipt, args.network_receipt_sha256, args.network_helper,
                                   openssl_prefix=prefix)
        if sha256(archive) != ARROW_ARCHIVE_SHA256:
            raise ValueError("Arrow source archive checksum mismatch")
        archives = {name: getattr(args, name + "_archive").absolute() for name in DEPENDENCIES}
        for name, path in archives.items():
            check_record({"path": str(path), "sha256": DEPENDENCIES[name]["sha256"]}, name + " archive")
        if args.check:
            return 0
        if args.resume:
            if not args.resume_state_sha256:
                raise ValueError("resume requires approved --resume-state-sha256")
            previous = json.loads((work / "build-receipt.json").read_bytes())
            state = previous["resume_state"]
            if state["sha256"] != args.resume_state_sha256:
                raise ValueError("resume state does not match approved hash")
            verify_resume_state(work, state)
        source, source_verification = prepare_source(archive, work, resume=args.resume)
        recipe_record = preserve_recipe(work, recipe_bytes, resume=args.resume)
        helper_record = preserve(work / "private-network-helper.py", Path(network["helper"]["path"]).read_bytes(),
                                 resume=args.resume, label="network helper source")
        captured_network = validate_network(args.network_receipt, args.network_receipt_sha256, Path(helper_record["path"]),
                                            openssl_prefix=prefix)
        if {key: value for key, value in captured_network.items() if key != "helper"} != {
                key: value for key, value in network.items() if key != "helper"}:
            raise ValueError("network source/provider closure changed during preparation")
        network = captured_network
        network_prefix = Path(network["prefix"])
        provider_cmake = preserve(work / "private-network.cmake", private_network_cmake(network_prefix, prefix),
                                  resume=args.resume, label="private network CMake metadata")
        (work / "inputs").mkdir(exist_ok=True)
        query = work / "cpp-build/.cmake/api/v1/query/codemodel-v2"
        query.parent.mkdir(parents=True, exist_ok=True)
        query_record = preserve(query, b"", resume=args.resume, label="CMake File API query")
        dependencies = {}
        for name, path in archives.items():
            copied = preserve(work / "inputs" / DEPENDENCIES[name]["filename"], path.read_bytes(),
                              resume=args.resume, label=name + " source archive")
            dependencies[name] = {**DEPENDENCIES[name], "archive": copied,
                                  "role": "retained_source_pin_only" if name == "zlib" else "bundled_provider"}
        context = {"python": args.python, "cc": args.cc, "cxx": args.cxx,
                   "openssl_prefix": str(prefix), "network_prefix": str(network_prefix), "work": str(work)}
        commands = build_commands(source, work, prefix, network_prefix=network_prefix,
                                   python=args.python, cc=args.cc, cxx=args.cxx, jobs=args.jobs,
                                   cmake=builder["toolchain"]["cmake"]["path"], ninja=builder["toolchain"]["ninja"]["path"])
        tool_versions = {name: subprocess.check_output([row["path"], "--version"], text=True).splitlines()[0]
                         for name, row in builder["toolchain"].items()}
        inputs = {
            "schema_version": 1, "source_policy": SOURCE_POLICY, "arrow_commit": ARROW_COMMIT,
            "source_archive": file_record(archive), "source_verification": file_record(work / "source-verification.json"),
            "integrated_sources": source_verification["integrated_sources"], "recipe": recipe_record,
            "builder": builder, "observed_builder": observed_builder, "toolchain_versions": tool_versions,
            "python_runtime": python_runtime,
            "openssl": openssl, "network": network, "provider_cmake": provider_cmake, "dependencies": dependencies,
            "commands": commands, "context": context,
            "cmake_file_api_query": query_record,
            "compiler_budget": {"maximum_jobs": args.jobs, "outer_ninja_jobs": args.jobs, "nested_cmake_jobs": 1, "nested_make_jobs": 1},
            "features": FEATURES, "codecs": CODECS,
            "version_environment": {"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW": "{wheel_version}"},
            "python_feature_environment": {key: value for key, value in env.items() if key.startswith("PYARROW_")},
            "generated_version_policy": "frozen vcs-versioning 2.6.0 default Python template; None SCM node; exact derived version",
            "resume_limit": "Generated Python source from failed wheel builds is preserved and requires a fresh generation.",
            "qualification": "unqualified host-specific candidate; expanded SDK/deferred ExternalProject archive/tree proof, all selected system sources and advisory gaps require independent review before runtime admission",
        }
        manifest, version = preserve_build_inputs(work, inputs, resume=args.resume)
        frozen_inputs = (recipe_record, provider_cmake, manifest, inputs["source_verification"], query_record,
                         *(dependency["archive"] for dependency in dependencies.values()))
        expected_version = expected_generated_version(args.python, version)
        preserve(work / "expected-generated-version.py", expected_version, resume=args.resume, label="generated version template")
        # Validate the reviewed configuration before replacing the mutable
        # incomplete-build receipt or adding a new checkpoint/log. A rejected
        # approval must leave the prior evidence exactly as it was.
        if not args.configure_only:
            configured_path = check_record({"path": str(work / "configured-inputs.json"), "sha256": args.configured_inputs_sha256}, "reviewed configured inputs")
            configured = json.loads(configured_path.read_bytes())
            check_record(configured["cache"], "reviewed configured cache")
            if configured["build_inputs"] != manifest:
                raise ValueError("configured dependency selection changed before compilation")
            verify_configured_selection(configured["dependencies"], capture_dependencies(work, compiled=False))
            verify_cache(work / "cpp-build/CMakeCache.txt", context, stage="cpp")
        record = {**inputs, "status": "building", "build_inputs": manifest, "wheel_version": version, "stages": []}
        receipt_path = work / "build-receipt.json"
        receipt_path.write_bytes(canonical_json(record))
        env.update(GIT_CEILING_DIRECTORIES=str(work), OPENSSL_ROOT_DIR=str(prefix),
                   PKG_CONFIG_PATH=os.pathsep.join(map(str, (network_prefix / "lib/pkgconfig", prefix / "lib/pkgconfig"))),
                   ARROW_HOME=str(work / "arrow-prefix"),
                   CMAKE_PREFIX_PATH=";".join(map(str, (work / "arrow-prefix", network_prefix, prefix))),
                   SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW=version, LDFLAGS="-Wl,-Map,%", TMPDIR=str(work / "tmp"))
        env["PATH"] = os.pathsep.join(dict.fromkeys([str(Path(row["path"]).parent) for row in builder["toolchain"].values()] +
                                                   [str(Path(args.python).parent), "/usr/bin", "/bin"]))
        for name, dependency in dependencies.items():
            if dependency["role"] == "bundled_provider":
                env[dependency["env"]] = dependency["archive"]["path"]
        (work / "tmp").mkdir(exist_ok=True)
        if args.configure_only:
            record["stages"].append(run_network_stage(work, "configure", commands["cpp_configure"], env, network,
                                                     frozen_inputs=frozen_inputs))
            verify_cache(work / "cpp-build/CMakeCache.txt", context, stage="cpp")
            configured = {"schema_version": 1, "build_inputs": manifest,
                          "cache": file_record(work / "cpp-build/CMakeCache.txt"),
                          "dependencies": capture_dependencies(work, compiled=False)}
            configured_record = preserve(work / "configured-inputs.json", canonical_json(configured), resume=False, label="configured inputs")
            record.update(status="configured_unverified", configured_inputs=configured_record)
            return_code = 0
        else:
            record["stages"].append(run_network_stage(work, "cpp-build", commands["cpp_build"], env, network,
                                                     frozen_inputs=frozen_inputs))
            verify_cache(work / "cpp-build/CMakeCache.txt", context, stage="cpp")
            dependencies_compiled = capture_dependencies(work, compiled=True)
            source_compilation = compile_source_evidence(work)
            record["stages"].append(run_network_stage(work, "python-build", commands["python_build"], env, network,
                                                     frozen_inputs=frozen_inputs))
            verify_cache(work / "python-build/CMakeCache.txt", context, stage="python")
            validate_builder(args.builder_receipt.absolute(), args.builder_receipt_sha256,
                             python=args.python, cc=args.cc, cxx=args.cxx)
            if validate_python_runtime(Path(builder["base_python"]["path"])) != python_runtime:
                raise ValueError("Python runtime source inputs changed during build")
            if validate_openssl(prefix, openssl_archive) != openssl:
                raise ValueError("approved OpenSSL inputs changed during build")
            with tarfile.open(archive) as bundle:
                root, _ = archive_members(bundle)
                final_sources = verify_archive_tree(bundle, source, source_transformations(bundle, root),
                    {"python/pyarrow/_generated_version.py": expected_version})
            maps = verify_link_maps(work, prefix, dependencies_compiled, network=network)
            wheel = inspect_wheel(work, record)
            verify_network(network)
            for reference in frozen_inputs:
                check_record(reference, "immutable build input")
            dependency_record = preserve(work / "compiled-dependencies.json", canonical_json(dependencies_compiled), resume=False, label="compiled dependencies")
            final_source_record = preserve(work / "compiled-sources.json", canonical_json({"files": final_sources, **source_compilation}), resume=False, label="compiled sources")
            record = {
                "schema_version": 1, "source_policy": SOURCE_POLICY, "status": "compiled_unverified", "name": "pyarrow", **wheel,
                "wheel_sha256": wheel["wheel"]["sha256"], "source_revision": ARROW_COMMIT,
                "recipe": recipe_record, "build_inputs": manifest, "source_verification": inputs["source_verification"],
                "network": network, "openssl": openssl, "provider_cmake": provider_cmake,
                "compiled_dependencies": dependency_record, "compiled_sources": final_source_record,
                "native_link_artifacts": maps["maps"], "final_native_link_artifacts": maps["final_maps"],
                "linked_static_archives": maps["linked_static_archives"],
                "toolchain": builder["toolchain"], "toolchain_versions": tool_versions,
                "python_build_distributions": builder["distributions"],
                "python_runtime": python_runtime,
                "effective_versions": {"pyarrow": version, "openssl": "3.6.5", **{name: dep["version"] for name, dep in DEPENDENCIES.items()}},
                "build": record, "qualification": "compiled host-specific candidate; native/source/security/API qualification and admission still required",
            }
            return_code = 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as error:
        if record is None:
            parser.error(str(error))
        record.update(status="failed", error=str(error))
        print(str(error), file=sys.stderr)
    finally:
        if record is not None:
            if record["status"] != "compiled_unverified":
                record["resume_state"] = write_resume_state(work)
            (work / "build-receipt.json").write_bytes(canonical_json(record))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
