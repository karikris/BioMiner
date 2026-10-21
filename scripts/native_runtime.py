#!/usr/bin/env python3
"""Fail-closed, profile-bound native runtime generations.

Profiles are operator-approved input, not self-authenticating certificates. Build
receipts are candidates; only an installed, immutable generation passing host and
runtime checks is admitted. No command updates an existing environment.
"""
from __future__ import annotations

import argparse
from collections import deque
import email
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any


class ProfileError(ValueError):
    """Missing, incompatible, or unverifiable native runtime evidence."""


VERSIONS = {'psycopg-c': '3.3.6', 'pyarrow': '25.0.1+biominer.1',
            'duckdb': '1.5.6+biominer.1', 'lxml': '6.1.3'}
LXML_NATIVE_VERSIONS = {'lxml': '6.1.3', 'libxml2': '2.15.4', 'libxslt': '1.1.45',
                        'libiconv': '1.19', 'zlib': '1.3.2'}
LXML_FEATURES = {'catalog', 'html', 'iconv', 'regexp', 'schematron',
                 'xmlschema', 'xpath', 'zlib'}
LXML_ZLIB_PATCH_COMMIT = 'df84af25dc1942490e1d1c899a07619152a46148'
LXML_ZLIB_PATCH_SHA256 = '110ff14375733173d8aa54574473424fbd7dfe4b81f1ca34a759c6fe14b15b14'
ARROW_SOURCE_REVISION = 'beccec0d0c451b7aa3e4530416ac431b3c035c69'
ARROW_ARCHIVE_SHA256 = 'a2cb8711cb74251ae8c7e9051f4a85b5034fd689867e202d774e417d532bdcf7'
ARROW_RECIPE_SHA256 = '65077fc43ba4542eb051b0a07a53ba147614551d192e774e13773aef0567da83'
ARROW_SECURITY_PATCH_COMMIT = 'a769c291e01093b73d03a075179cf7a09bf92ad8'
ARROW_SECURITY_PATCH_SHA256 = 'd51722f4ddb3934a0c846b691a60a730cb77d5a814d7077cab1c5d145c54aaa1'
ARROW_SECURITY_FILES = {
    'cpp/src/arrow/compute/kernels/vector_replace.cc': {
        'original_sha256': '201ee04949830bdd07c1f2f87efda6201b39e6aad2d6a76d8dd868e77d6f156e',
        'patched_sha256': 'ddd28a89eb41b79e3dc5ecec4cc0e26eb7cec535516a05831b625c8b598afd13',
    },
    'cpp/src/arrow/compute/kernels/vector_replace_test.cc': {
        'original_sha256': '24fdcfad027471f6833dc6efd1f07ba04a3adab7f1edb7aa485c9d2175a28e73',
        'patched_sha256': '2f926df7f6fbce5f7c9c0a32575e4eda39f6c446fa25ff0fa083046b63bb4360',
    },
}
ARROW_OFFSET_PATCH_COMMIT = '0efbdec7a10da1a4043fc1210c61e73c88bc9b4f'
ARROW_OFFSET_PATCH_SHA256 = '6c64e115aa53621ae7e1992417b26ab7c6c51fb265d3e64216580f7dcae87d50'
ARROW_OFFSET_FILES = {
    'cpp/src/arrow/compute/kernels/copy_data_internal.h': {
        'original_sha256': 'd1f2979d5a4d4f73f4ab0a40043d377962871f0d85ff3b4a494bcbf94c948c99',
        'patched_sha256': '21b1f8acce63e0ca301a968339d731eddb5efd8e8eeb1a55638ffe665a046621',
    },
    'cpp/src/arrow/compute/kernels/vector_replace_test.cc': {
        'original_sha256': '2f926df7f6fbce5f7c9c0a32575e4eda39f6c446fa25ff0fa083046b63bb4360',
        'patched_sha256': 'c087392bbd8114e903608b36c1e4e88a087677cbcfda484cb270d227b385788e',
    },
}
# Arrow 26 is a separate integrated-source policy. The completed raw build
# receipt gets its own exact pin only after the actual native closure review.
ARROW26_BUILD_RECEIPT_SHA256 = None
ARROW26_SOURCE_REVISION = '446167169a0dc547e00e1bc29f417a40c0ca267a'
ARROW26_ARCHIVE_SHA256 = '764adbb5601a4d0d9d67a10cf04892d33cfac9fd43012f08a2d7c9956a7a90d2'
ARROW26_RECIPE_SHA256 = '3bd6d35656f89fa71d1eaccd63477f5a94fbe074a56368af5b372660bbeac567'
ARROW26_BUILDER_RECEIPT_SHA256 = '303c3f5561d8e93e9e60d5bef476c3da8ebfedaa33c379433bda26e5352da561'
ARROW26_PYTHON_RUNTIME_PINS = {
    'receipt': '43a7e96c319752ec519a3c7e213cbb063bb8d140bbb598e69ceb6b49b2c787f4',
    'archive': 'da5357f47b1d9c8d4c439004d463d893ef688a44c27438a62b47073d26d1377e',
    'python': '1747426f232ecbe022fc83eef7e56ccedcb9841c3acea82c43f1f6a013c05027',
}
ARROW26_INTEGRATED_SOURCES = {
    'cpp/src/arrow/compute/kernels/copy_data_internal.h': '21b1f8acce63e0ca301a968339d731eddb5efd8e8eeb1a55638ffe665a046621',
    'cpp/src/arrow/compute/kernels/vector_replace.cc': '12cd5bd3ab8e487aba96743c92cea9f234e3efba279f1200126029f84717d9c1',
    'cpp/src/arrow/compute/kernels/vector_replace_test.cc': 'c087392bbd8114e903608b36c1e4e88a087677cbcfda484cb270d227b385788e',
    'cpp/src/arrow/compute/kernels/scalar_if_else.cc': '87f5ce8b675c9dcbaf1f45b25e2515a161fe3bf4b1b567acba0ecea287510fa8',
}
ARROW26_TRANSFORMATIONS = [
    {
        "relative": "cpp/cmake_modules/ThirdpartyToolchain.cmake",
        "original_sha256": "3dff91dcec6ab6f12e38021c383db580de571d722c5fa39b173aeb044317c6e3",
        "sha256": "1789d7d856a58ddec7aa8207ff15e2edaefc43a5afa8defd8aa382cc09894345"
    },
    {
        "relative": "cpp/thirdparty/versions.txt",
        "original_sha256": "4c071a79172875a47f389f65e47a43bcb89531822b32d8b0c5c32909f0043628",
        "sha256": "da8d6287e7f60f54d142bfcd3a794eb564a0cf8d147aa2a63f8908315d1819c0"
    }
]
ARROW26_DEPENDENCIES = {
    'boost': {'version': '1.92.0', 'sha256': 'f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984'},
    'simdjson': {'version': '4.6.11', 'sha256': '9157ef4acb368d727873b4dd592f7b1d67d19fcb9201e8bc5a6b093e54e5171a'},
    'uriparser': {'version': '1.0.2', 'sha256': 'dd2e4843f43de6f5aedf430e530f1e2d159eba8ca4d64c2787af1c20f707a9e4'},
    'zlib': {'version': '1.3.2', 'sha256': 'bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16'},
}
ARROW26_OPENSSL_PINS = {
    'receipt': '66d41e5fa8cee1cdb865a7df41620685776c96fd8eba9abcc6a70505ca3da13a',
    'source_archive': 'f3091b7ccb41a11d877b7406b7afc9dd9c5003d299c6bb33cff882127bc06c91',
    'header': '80d9b9534fdbdab12c29cb8c62887ede14bbb7112b93e838a63d6815400f5f0b',
}
ARROW26_OPENSSL_STATIC = {
    'libcrypto.a': 'e2d8161275df2868b63e28d8d1f5b49efa3e40a3697d1a21a07890c58a2d3bc2',
    'libssl.a': '3ac9b8f49ce67cc03e9a1073754faaec92226195d610bd59800e4df5c41a82b6',
}
ARROW26_BUILDER_PINS = {
    'build': '1.6.1', 'cython': '3.3.0', 'libcst': '1.9.0', 'numpy': '2.5.4',
    'packaging': '26.3', 'pathspec': '1.1.1', 'pyproject-hooks': '1.3.3',
    'pyyaml': '6.0.3', 'scikit-build-core': '1.1.1', 'setuptools': '84.2.1',
    'setuptools-scm': '10.3.4', 'vcs-versioning': '2.6.0',
}
ARROW26_FEATURES = {'ACERO', 'AZURE', 'COMPUTE', 'CSV', 'DATASET', 'FILESYSTEM',
                    'FLIGHT', 'GCS', 'HDFS', 'JSON', 'ORC', 'PARQUET', 'S3', 'SUBSTRAIT'}
ARROW26_CODECS = {'BROTLI', 'BZ2', 'LZ4', 'SNAPPY', 'ZLIB', 'ZSTD'}

# These two private-provider families have never been admitted. Raw receipt and
# private-prefix pins stay closed until independent actual-build review.
ARROW26_PRIVATE_FAMILY = 'arrow26-integrated-private-network-v2'
ARROW25_NETWORK_FAMILY = 'arrow25-two-compute-patches-private-network-v1'
ARROW25_NETWORK_BUILD_RECEIPT_SHA256 = None
ARROW25_NETWORK_SOURCE_REVISION = ARROW_SOURCE_REVISION
ARROW25_NETWORK_ARCHIVE_SHA256 = ARROW_ARCHIVE_SHA256
ARROW25_NETWORK_RECIPE_SHA256 = 'c73a4e5b11c3abea2391a65587362c073c6b97ee6f36c83bc8e0c909ff870a4d'
ARROW25_NETWORK_TRANSFORMATIONS = [
    {
        "relative": "cpp/cmake_modules/ThirdpartyToolchain.cmake",
        "original_sha256": "36873bdf3a91dbe13830eccefb70bf6369a948e1c1231aa45201d9c8ee85763b",
        "sha256": "f49fdca1cb0bab086ae85b377dd38170317f0d3060f7146e4da4b5405e153c17"
    },
    {
        "relative": "cpp/thirdparty/versions.txt",
        "original_sha256": "c90e546c2a0706b6dda862a19eff8c13d77e3c0972291de75f3bac9d2e21fea7",
        "sha256": "aa6b048322591b6fb5e68f06d523ea74fcbe63ed0feea266cfd96b3f04193e6e"
    }
]
ARROW25_NETWORK_SOURCES = {
    'cpp/src/arrow/compute/kernels/vector_replace.cc': ARROW_SECURITY_FILES['cpp/src/arrow/compute/kernels/vector_replace.cc']['patched_sha256'],
    'cpp/src/arrow/compute/kernels/vector_replace_test.cc': ARROW_OFFSET_FILES['cpp/src/arrow/compute/kernels/vector_replace_test.cc']['patched_sha256'],
    'cpp/src/arrow/compute/kernels/copy_data_internal.h': ARROW_OFFSET_FILES['cpp/src/arrow/compute/kernels/copy_data_internal.h']['patched_sha256'],
    'cpp/src/arrow/compute/kernels/scalar_if_else.cc': 'd2f6c4f91e7e1bd9a0281d6c60989dd4cda493aa4eda51e503da61c2ab4d5770',
}
ARROW25_NETWORK_DEPENDENCIES = {'boost': ARROW26_DEPENDENCIES['boost']}
ARROW_NETWORK_HELPER_SHA256 = "a8cb0355e3abfa4415093268c17678d63ea1e745a48f4a5dd750a6a01d037414"
ARROW_NETWORK_BUILD_RECEIPT_SHA256 = "9371f44d18d198179e2a4dc808dc534feef7faca9e3d27328f4bcbfcb84c6b99"
ARROW_NETWORK_ARCHIVES = ('libcurl.a', 'libxml2.a', 'libiconv.a', 'libcharset.a', 'libz.a', 'libnghttp2.a')
ARROW_NETWORK_COMPONENTS = {
    'libcurl': ('curl', '8.23.0'), 'libxml2': ('libxml2', '2.15.4'),
    'libiconv': ('libiconv', '1.19'), 'libcharset': ('libiconv', '1.19'),
    'zlib': ('zlib', '1.3.2'), 'libnghttp2': ('nghttp2', '1.68.0-2ubuntu0.2'),
}
ARROW_NETWORK_ZLIB_PATCH_SHA256 = '110ff14375733173d8aa54574473424fbd7dfe4b81f1ca34a759c6fe14b15b14'
ARROW_NETWORK_ZLIB_ORIGINAL_SHA256 = '2ce783294c688330c716bfe106f43167d3dae5be84994176bc304d205d4f5b04'
ARROW_NETWORK_ZLIB_PATCHED_SHA256 = '31d0da140edf382b82b13e2d87a22f59e158d9014ee026459e01cf437d0ab07d'
# New cohort schemas retain the preceding never-admitted family pins unchanged.
ARROW26_COHORT_FAMILY = 'arrow26-integrated-private-network-v3'
ARROW25_COHORT_FAMILY = 'arrow25-two-compute-patches-private-network-v2'
ARROW_COHORT_FAMILIES = {ARROW26_COHORT_FAMILY, ARROW25_COHORT_FAMILY}
ARROW_SOURCE_ATTESTER_SHA256 = None
ARROW_COHORT_METADATA_SHA256 = '584b47d20f2ec9ce3d3f98ed4e616c9b866de96d5033cd8f8406060b7d88a69f'
ARROW_SDK_COHORT = {'aws_sdk_cpp': {'variable': 'AWSSDK',
                 'version': '1.11.914',
                 'build_version': 'd207a37ecb8dc33b5e131d0d59843d264b7f7e86',
                 'commit': 'd207a37ecb8dc33b5e131d0d59843d264b7f7e86',
                 'sha256': '108382fed71e70ada73d4aaa45e0ba54777f7eca6bbd44ac24b9af9ef54f069e',
                 'filename': 'aws-sdk-cpp-d207a37ecb8dc33b5e131d0d59843d264b7f7e86.tar.gz',
                 'url': 'https://github.com/aws/aws-sdk-cpp/archive/d207a37ecb8dc33b5e131d0d59843d264b7f7e86.tar.gz',
                 'env': 'ARROW_AWSSDK_URL',
                 'old_version': '1.11.778',
                 'old_sha256': '06371849792f1a8633c3e0592a17b4700f39c748ad374f4e9a6592739ecc4d9c'},
 'aws_crt_cpp': {'variable': 'AWS_CRT_CPP',
                 'version': '0.43.8',
                 'build_version': '9c319786d1d83d5ace6a9de817dcfe1861003609',
                 'commit': '9c319786d1d83d5ace6a9de817dcfe1861003609',
                 'sha256': 'f107357efa7d6b9be783d96d49cdaf000fe881417c431ae9fe27faf1bb55f431',
                 'filename': 'aws-crt-cpp-9c319786d1d83d5ace6a9de817dcfe1861003609.tar.gz',
                 'url': 'https://github.com/awslabs/aws-crt-cpp/archive/9c319786d1d83d5ace6a9de817dcfe1861003609.tar.gz',
                 'env': 'ARROW_AWS_CRT_CPP_URL',
                 'old_version': 'v0.38.0',
                 'old_sha256': 'f57b82f0a9e6890e4d47aa1c48b9324b5beb5b58000715c4c6ef9f147be13891'},
 'aws_c_auth': {'variable': 'AWS_C_AUTH',
                'version': '4b5d524bf1a511b05e0fffe5bdc51800770b9427',
                'build_version': '4b5d524bf1a511b05e0fffe5bdc51800770b9427',
                'commit': '4b5d524bf1a511b05e0fffe5bdc51800770b9427',
                'sha256': 'b4f6690b382abe049d543804f40230e667a95a18f64506ef7647925e8c6b9b67',
                'filename': 'aws-c-auth-4b5d524bf1a511b05e0fffe5bdc51800770b9427.tar.gz',
                'url': 'https://github.com/awslabs/aws-c-auth/archive/4b5d524bf1a511b05e0fffe5bdc51800770b9427.tar.gz',
                'env': 'ARROW_AWS_C_AUTH_URL',
                'old_version': 'v0.10.1',
                'old_sha256': '85d737f0f735256f1931e85e4cadbe228d88698f7b59a9b390b49ef5d0778a43'},
 'aws_c_cal': {'variable': 'AWS_C_CAL',
               'version': '8aa2a48a09f93c65d4cf06388e143a6584de6321',
               'build_version': '8aa2a48a09f93c65d4cf06388e143a6584de6321',
               'commit': '8aa2a48a09f93c65d4cf06388e143a6584de6321',
               'sha256': '29706ab10a4d95a74db278925b82c85077c08224c0efe837841d2354555c8817',
               'filename': 'aws-c-cal-8aa2a48a09f93c65d4cf06388e143a6584de6321.tar.gz',
               'url': 'https://github.com/awslabs/aws-c-cal/archive/8aa2a48a09f93c65d4cf06388e143a6584de6321.tar.gz',
               'env': 'ARROW_AWS_C_CAL_URL',
               'old_version': 'v0.9.13',
               'old_sha256': '80b7c6087b0af461b4483e4c9483aea2e0dac5d9fb2289b057159ea6032409e1'},
 'aws_c_common': {'variable': 'AWS_C_COMMON',
                  'version': 'f7d471b1feaea9a310851672580f9ac7e031cfc0',
                  'build_version': 'f7d471b1feaea9a310851672580f9ac7e031cfc0',
                  'commit': 'f7d471b1feaea9a310851672580f9ac7e031cfc0',
                  'sha256': 'a87a89fa6678fda59dd66ea9246e70d837bf71ecec39829ec712e6863d1453e4',
                  'filename': 'aws-c-common-f7d471b1feaea9a310851672580f9ac7e031cfc0.tar.gz',
                  'url': 'https://github.com/awslabs/aws-c-common/archive/f7d471b1feaea9a310851672580f9ac7e031cfc0.tar.gz',
                  'env': 'ARROW_AWS_C_COMMON_URL',
                  'old_version': 'v0.12.6',
                  'old_sha256': '138822ecdcaff1d702f37d4751f245847d088592724921cc6bf61c232b198d6b'},
 'aws_c_compression': {'variable': 'AWS_C_COMPRESSION',
                       'version': 'd8264e64f698341eb03039b96b4f44702a9b3f83',
                       'build_version': 'd8264e64f698341eb03039b96b4f44702a9b3f83',
                       'commit': 'd8264e64f698341eb03039b96b4f44702a9b3f83',
                       'sha256': '537971fe42e989b638e0f436683357ed3d07f07b06b910b1c37ab3a8094bb8fd',
                       'filename': 'aws-c-compression-d8264e64f698341eb03039b96b4f44702a9b3f83.tar.gz',
                       'url': 'https://github.com/awslabs/aws-c-compression/archive/d8264e64f698341eb03039b96b4f44702a9b3f83.tar.gz',
                       'env': 'ARROW_AWS_C_COMPRESSION_URL',
                       'old_version': 'v0.3.2',
                       'old_sha256': 'f93f5a5d8b3fee3a6d97b14ba279efacd4d4016ef9cc7dc4be7d43519ecfbe93'},
 'aws_c_event_stream': {'variable': 'AWS_C_EVENT_STREAM',
                        'version': '51bef3c44e1058b1689751539170b2e0f589ccdb',
                        'build_version': '51bef3c44e1058b1689751539170b2e0f589ccdb',
                        'commit': '51bef3c44e1058b1689751539170b2e0f589ccdb',
                        'sha256': 'b890d4940957d1ddade1a9d7de4adc37b7cb98dfdfdfec7436527fec8cb558df',
                        'filename': 'aws-c-event-stream-51bef3c44e1058b1689751539170b2e0f589ccdb.tar.gz',
                        'url': 'https://github.com/awslabs/aws-c-event-stream/archive/51bef3c44e1058b1689751539170b2e0f589ccdb.tar.gz',
                        'env': 'ARROW_AWS_C_EVENT_STREAM_URL',
                        'old_version': 'v0.7.0',
                        'old_sha256': '88835b4c78462547917f622fd9dda45c991b7e356d9c07e2f0537d4d97fbd4fb'},
 'aws_c_http': {'variable': 'AWS_C_HTTP',
                'version': '8aefd899fc3210bfd0e3fd414011a3cb708bf6e4',
                'build_version': '8aefd899fc3210bfd0e3fd414011a3cb708bf6e4',
                'commit': '8aefd899fc3210bfd0e3fd414011a3cb708bf6e4',
                'sha256': '61b1770e2e3a2bf046005cbfa8e26d621d8f266f05d03589ebee36bb4a0ac39f',
                'filename': 'aws-c-http-8aefd899fc3210bfd0e3fd414011a3cb708bf6e4.tar.gz',
                'url': 'https://github.com/awslabs/aws-c-http/archive/8aefd899fc3210bfd0e3fd414011a3cb708bf6e4.tar.gz',
                'env': 'ARROW_AWS_C_HTTP_URL',
                'old_version': 'v0.10.13',
                'old_sha256': 'd8352e7a1fb1996694a4dc31219ce03452882abf8d0858c104727f975e11b9c7'},
 'aws_c_io': {'variable': 'AWS_C_IO',
              'version': '28e4eb351a536daf1627bf69279e50671f7729b5',
              'build_version': '28e4eb351a536daf1627bf69279e50671f7729b5',
              'commit': '28e4eb351a536daf1627bf69279e50671f7729b5',
              'sha256': 'ea51cbe20981ed34b7a8c01a60c848e4ccb716a29cd740ead07ffd61be5169c3',
              'filename': 'aws-c-io-28e4eb351a536daf1627bf69279e50671f7729b5.tar.gz',
              'url': 'https://github.com/awslabs/aws-c-io/archive/28e4eb351a536daf1627bf69279e50671f7729b5.tar.gz',
              'env': 'ARROW_AWS_C_IO_URL',
              'old_version': 'v0.26.3',
              'old_sha256': '521fd0848fca661130bbb7278a414d7a38bdcb9bc8ffa89f6660d84e5838a303'},
 'aws_c_mqtt': {'variable': 'AWS_C_MQTT',
                'version': 'e35b9ca3f9fcbf1a972c831c5e79046ce56959d1',
                'build_version': 'e35b9ca3f9fcbf1a972c831c5e79046ce56959d1',
                'commit': 'e35b9ca3f9fcbf1a972c831c5e79046ce56959d1',
                'sha256': '54d1987ce06eb7b39faace854818e9322d3df79cb310ada350beca0b12cb89cc',
                'filename': 'aws-c-mqtt-e35b9ca3f9fcbf1a972c831c5e79046ce56959d1.tar.gz',
                'url': 'https://github.com/awslabs/aws-c-mqtt/archive/e35b9ca3f9fcbf1a972c831c5e79046ce56959d1.tar.gz',
                'env': 'ARROW_AWS_C_MQTT_URL',
                'old_version': 'v0.15.2',
                'old_sha256': '66f3f5edff4ad1f765a86d3342b6017d0f29f950c1c24f8c1edacdc895202edc'},
 'aws_c_s3': {'variable': 'AWS_C_S3',
              'version': '226c3e6937e3d5f7db9e10651f7bd20575e98187',
              'build_version': '226c3e6937e3d5f7db9e10651f7bd20575e98187',
              'commit': '226c3e6937e3d5f7db9e10651f7bd20575e98187',
              'sha256': 'a8f33816bf7aff38467fb15ca233a33a9113be78dad47c8a7ee30d323b42ba74',
              'filename': 'aws-c-s3-226c3e6937e3d5f7db9e10651f7bd20575e98187.tar.gz',
              'url': 'https://github.com/awslabs/aws-c-s3/archive/226c3e6937e3d5f7db9e10651f7bd20575e98187.tar.gz',
              'env': 'ARROW_AWS_C_S3_URL',
              'old_version': 'v0.12.0',
              'old_sha256': '1a8a8ceda0585d52028a1f3daa5861f924e7d8d2f6a17bec05813dc0b74d6eed'},
 'aws_c_sdkutils': {'variable': 'AWS_C_SDKUTILS',
                    'version': 'a1cc19f53b63658f1b1400b36f199eafeeb895a6',
                    'build_version': 'a1cc19f53b63658f1b1400b36f199eafeeb895a6',
                    'commit': 'a1cc19f53b63658f1b1400b36f199eafeeb895a6',
                    'sha256': 'f8ef9e93b30f9833d9c9ea474110c8fa2d29fd78b885546b07e68558ece3bb1c',
                    'filename': 'aws-c-sdkutils-a1cc19f53b63658f1b1400b36f199eafeeb895a6.tar.gz',
                    'url': 'https://github.com/awslabs/aws-c-sdkutils/archive/a1cc19f53b63658f1b1400b36f199eafeeb895a6.tar.gz',
                    'env': 'ARROW_AWS_C_SDKUTILS_URL',
                    'old_version': 'v0.2.4',
                    'old_sha256': '493cbed4fa57e0d4622fcff044e11305eb4fc12445f32c8861025597939175fc'},
 'aws_checksums': {'variable': 'AWS_CHECKSUMS',
                   'version': '1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e',
                   'build_version': '1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e',
                   'commit': '1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e',
                   'sha256': 'c58d8c5319319dcade0926da44ad12effecb21372983e22835a657a32f03c9c9',
                   'filename': 'aws-checksums-1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e.tar.gz',
                   'url': 'https://github.com/awslabs/aws-checksums/archive/1d5f2f1f3e5d013aae8810878ceb5b3f6f258c4e.tar.gz',
                   'env': 'ARROW_AWS_CHECKSUMS_URL',
                   'old_version': 'v0.2.10',
                   'old_sha256': 'cb6509f75e42ee25c372a6d379e8582ce5179e5335183842e808f7d8abb0c314'},
 's2n_tls': {'variable': 'S2N_TLS',
             'version': '1.7.12',
             'build_version': '6459002cd61e4bf3bbc53db23f068b21da9ff6bd',
             'commit': '6459002cd61e4bf3bbc53db23f068b21da9ff6bd',
             'sha256': '94a57c4f4713cc20088b723bcdf0bb254e32a559ffcbed583e67fe3d8ddec442',
             'filename': 's2n-tls-6459002cd61e4bf3bbc53db23f068b21da9ff6bd.tar.gz',
             'url': 'https://github.com/aws/s2n-tls/archive/6459002cd61e4bf3bbc53db23f068b21da9ff6bd.tar.gz',
             'env': 'ARROW_S2N_TLS_URL',
             'old_version': 'v1.7.0',
             'old_sha256': 'a6e8228e238239bb3c17b1eda3ed702bcbb2eaebc792eac4d754cc5619b0ea06'},
 'thrift': {'variable': 'THRIFT',
            'version': '0.25.0',
            'build_version': '0.25.0',
            'commit': '27e8a425ffb498e190df3a12e239326bf5ba9ed6',
            'sha256': '66da4707214c54c94bac082103dc67adaf9e08925662700f269170a7b534b214',
            'filename': 'thrift-27e8a425ffb498e190df3a12e239326bf5ba9ed6.tar.gz',
            'url': 'https://www.apache.org/dyn/closer.lua/thrift/0.25.0/thrift-0.25.0.tar.gz?action=download',
            'env': 'ARROW_THRIFT_URL',
            'old_version': '0.22.0',
            'old_sha256': '794a0e455787960d9f27ab92c38e34da27e8deeda7a5db0e59dc64a00df8a1e5'},
 'grpc': {'variable': 'GRPC',
          'version': '1.83.1',
          'build_version': 'v1.83.1',
          'commit': 'aae267021b1ac256f8b9038d0ef528c3798cc137',
          'sha256': '60caa8397426d8a500e3e57ab6a49d1cb5aa62a36f2591eb9da3d77fa38ad8c9',
          'filename': 'grpc-v1.83.1.tar.gz',
          'url': 'https://github.com/grpc/grpc/archive/v1.83.1.tar.gz',
          'env': 'ARROW_GRPC_URL',
          'old_version': 'v1.76.0',
          'old_sha256': '0af37b800953130b47c075b56683ee60bdc3eda3c37fc6004193f5b569758204'},
 'protobuf': {'variable': 'PROTOBUF',
              'version': '35.1',
              'build_version': 'v35.1',
              'commit': '35cd01f9fe9afbeea38cc7b979a3b6bfcde82c03',
              'sha256': 'f0b6838e7522a8da96126d487068c959bc624926368f3024ac8fd03abd0a1ac4',
              'filename': 'protobuf-35.1.tar.gz',
              'url': 'https://github.com/protocolbuffers/protobuf/releases/download/v35.1/protobuf-35.1.tar.gz',
              'env': 'ARROW_PROTOBUF_URL',
              'old_version': 'v31.1',
              'old_sha256': '12bfd76d27b9ac3d65c00966901609e020481b9474ef75c7ff4601ac06fa0b82'},
 'absl': {'variable': 'ABSL',
          'version': '20250512.1',
          'build_version': '20250512.1',
          'commit': '76bb24329e8bf5f39704eb10d21b9a80befa7c81',
          'sha256': '9b7a064305e9fd94d124ffa6cc358592eb42b5da588fb4e07d09254aa40086db',
          'filename': 'absl-20250512.1.tar.gz',
          'url': 'https://github.com/abseil/abseil-cpp/archive/20250512.1.tar.gz',
          'env': 'ARROW_ABSL_URL',
          'old_version': '20250127.0',
          'old_sha256': '16242f394245627e508ec6bb296b433c90f8d914f73b9c026fddb905e27276e8'},
 'cares': {'variable': 'CARES',
           'version': '1.34.9',
           'build_version': '1.34.9',
           'commit': 'bc91829cce983a35ec01f5dbc05465858a3df469',
           'sha256': '9364b70a12eb59a06830fb488d9357f90100dff0d98b23ea3593a183d43e6f2f',
           'filename': 'c-ares-1.34.9.tar.gz',
           'url': 'https://github.com/c-ares/c-ares/releases/download/v1.34.9/c-ares-1.34.9.tar.gz',
           'env': 'ARROW_CARES_URL',
           'old_version': '1.34.6',
           'old_sha256': '912dd7cc3b3e8a79c52fd7fb9c0f4ecf0aaa73e45efda880266a2d6e26b84ef5'},
 'rapidjson': {'variable': 'RAPIDJSON',
               'version': '24b5e7a8b27f42fa16b96fc70aade9106cf7102f',
               'build_version': '24b5e7a8b27f42fa16b96fc70aade9106cf7102f',
               'commit': '24b5e7a8b27f42fa16b96fc70aade9106cf7102f',
               'sha256': '2d2601a82d2d3b7e143a3c8d43ef616671391034bc46891a9816b79cf2d3e7a8',
               'filename': 'rapidjson-24b5e7a8b27f42fa16b96fc70aade9106cf7102f.tar.gz',
               'url': 'https://github.com/Tencent/rapidjson/archive/24b5e7a8b27f42fa16b96fc70aade9106cf7102f.tar.gz',
               'env': 'ARROW_RAPIDJSON_URL',
               'old_version': '232389d4f1012dddec4ef84861face2d2ba85709',
               'old_sha256': 'b9290a9a6d444c8e049bd589ab804e0ccf2b05dc5984a19ed5ae75d090064806'},
 'jemalloc': {'variable': 'JEMALLOC',
              'version': '5.3.1',
              'build_version': '5.3.1',
              'commit': '81034ce1f1373e37dc865038e1bc8eeecf559ce8',
              'sha256': '3826bc80232f22ed5c4662f3034f799ca316e819103bdc7bb99018a421706f92',
              'filename': 'jemalloc-5.3.1.tar.bz2',
              'url': 'https://github.com/jemalloc/jemalloc/releases/download/5.3.1/jemalloc-5.3.1.tar.bz2',
              'env': 'ARROW_JEMALLOC_URL',
              'old_version': '5.3.0',
              'old_sha256': '2db82d1e7119df3e71b7640219b6dfe84789bc0537983c3b7ac4f7189aecfeaa'}}
ARROW26_COHORT_BUILD_RECEIPT_SHA256 = None
ARROW26_COHORT_ATTESTATION_SHA256 = None
ARROW26_COHORT_SOURCE_REVISION = ARROW26_SOURCE_REVISION
ARROW26_COHORT_ARCHIVE_SHA256 = ARROW26_ARCHIVE_SHA256
ARROW26_COHORT_RECIPE_SHA256 = '1bc69557acd83f150040f5001fd8243b6bcf98250023990e6d22e179cf6264b1'
ARROW26_COHORT_TRANSFORMATIONS = None
ARROW26_COHORT_DEPENDENCIES = {**ARROW26_DEPENDENCIES, **ARROW_SDK_COHORT}
ARROW25_COHORT_BUILD_RECEIPT_SHA256 = None
ARROW25_COHORT_ATTESTATION_SHA256 = None
ARROW25_COHORT_SOURCE_REVISION = ARROW25_NETWORK_SOURCE_REVISION
ARROW25_COHORT_ARCHIVE_SHA256 = ARROW25_NETWORK_ARCHIVE_SHA256
ARROW25_COHORT_RECIPE_SHA256 = '4f7df429cc94697f155b7e252a8d0241225b1ea5800319f44bb6fb7b08da9d9b'
ARROW25_COHORT_TRANSFORMATIONS = None
ARROW25_COHORT_DEPENDENCIES = {**ARROW25_NETWORK_DEPENDENCIES, **ARROW_SDK_COHORT}
ARROW_PRIVATE_FAMILIES = {
    ARROW26_PRIVATE_FAMILY: ('Arrow 26', 'ARROW26', '26.0.0', 'arrow26_build_receipt', '26'),
    ARROW25_NETWORK_FAMILY: ('Arrow 25', 'ARROW25_NETWORK', '25.0.1', 'arrow25_network_build_receipt', '25'),
    ARROW26_COHORT_FAMILY: ('Arrow 26 cohort', 'ARROW26_COHORT', '26.0.0', 'arrow26_cohort_build_receipt', '26'),
    ARROW25_COHORT_FAMILY: ('Arrow 25 cohort', 'ARROW25_COHORT', '25.0.1', 'arrow25_cohort_build_receipt', '25'),
}


def _arrow_private_pins(family):
    if family not in ARROW_PRIVATE_FAMILIES:
        raise ProfileError('unreviewed private Arrow family')
    label, prefix, version, raw_field, schema_kind = ARROW_PRIVATE_FAMILIES[family]
    return {'label': label, 'version': version, 'raw_field': raw_field, 'schema_kind': schema_kind,
            'requires_source_attestation': family in ARROW_COHORT_FAMILIES,
            'attestation_sha256': globals()[prefix + '_ATTESTATION_SHA256'] if family in ARROW_COHORT_FAMILIES else None,
            **{name: globals()[prefix + '_' + suffix] for name, suffix in (
                ('source_revision', 'SOURCE_REVISION'), ('archive_sha256', 'ARCHIVE_SHA256'),
                ('recipe_sha256', 'RECIPE_SHA256'), ('transformations', 'TRANSFORMATIONS'),
                ('dependencies', 'DEPENDENCIES'), ('raw_sha256', 'BUILD_RECEIPT_SHA256'))},
            'sources': ARROW26_INTEGRATED_SOURCES if schema_kind == '26' else ARROW25_NETWORK_SOURCES}


def _arrow_private_toolchain(raw, family):
    return raw.get('toolchain') if _arrow_private_pins(family)['schema_kind'] == '26' else raw.get('builder', {}).get('toolchain')


def _arrow_private_effective_versions(raw, family):
    if _arrow_private_pins(family)['schema_kind'] == '26':
        return raw.get('effective_versions')
    return {'pyarrow': raw.get('version'), 'openssl': raw.get('openssl', {}).get('version'),
            **{key: item.get('version') for key, item in raw.get('dependencies', {}).items()},
            'curl': '8.23.0', 'libxml2': '2.15.4', 'libiconv': '1.19', 'zlib': '1.3.2',
            'nghttp2': '1.68.0-2ubuntu0.2'}


def _arrow_protected_provider(path):
    return isinstance(path, str) and bool(re.fullmatch(
        r'lib(?:curl(?:-[^.]+)?|xml2|z|iconv|charset|ssl|crypto|nghttp2)(?:\.a|\.so(?:\.[0-9]+)*)', Path(path).name))


def _arrow_private_flags(context):
    prefix, network = Path(context['openssl_prefix']), Path(context['network_prefix'])
    return {
        'CMAKE_PROJECT_INCLUDE': str(Path(context['work']) / 'private-network.cmake'),
        'CMAKE_FIND_PACKAGE_PREFER_CONFIG': 'OFF', 'ZLIB_SOURCE': 'SYSTEM',
        'ZLIB_LIBRARY': str(network / 'lib/libz.a'), 'ZLIB_LIBRARY_RELEASE': str(network / 'lib/libz.a'),
        'ZLIB_LIBRARY_DEBUG': str(network / 'lib/libz.a'), 'ZLIB_INCLUDE_DIR': str(network / 'include'),
        'CURL_NO_CURL_CMAKE': 'TRUE', 'CURL_USE_STATIC_LIBS': 'TRUE',
        'CURL_LIBRARY': str(network / 'lib/libcurl.a'), 'CURL_LIBRARY_RELEASE': str(network / 'lib/libcurl.a'),
        'CURL_LIBRARY_DEBUG': str(network / 'lib/libcurl.a'), 'CURL_INCLUDE_DIR': str(network / 'include'),
        'LIBXML2_LIBRARY': str(network / 'lib/libxml2.a'), 'LIBXML2_INCLUDE_DIR': str(network / 'include/libxml2'),
        'Iconv_LIBRARY': str(network / 'lib/libiconv.a'), 'Iconv_INCLUDE_DIR': str(network / 'include'),
        'Iconv_IS_BUILT_IN': 'FALSE', 'NGHTTP2_LIBRARY': str(network / 'lib/libnghttp2.a'),
        'NGHTTP2_INCLUDE_DIR': str(network / 'include'), 'OPENSSL_USE_STATIC_LIBS': 'TRUE',
        'OPENSSL_ROOT_DIR': str(prefix), 'OPENSSL_INCLUDE_DIR': str(prefix / 'include'),
        'OPENSSL_CRYPTO_LIBRARY': str(prefix / 'lib/libcrypto.a'), 'OPENSSL_SSL_LIBRARY': str(prefix / 'lib/libssl.a'),
    }


RUNTIME_CHECKS = {'effective_versions', 'native_inventory', 'extension_policy'}
INSTALL_CHECKS = RUNTIME_CHECKS | {'tls', 'geometry'}
BLOCKED_ENV = {'LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'GDAL_DRIVER_PATH', 'OPENSSL_MODULES', 'OPENSSL_CONF'}
SHA = re.compile(r'[0-9a-f]{64}')


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ProfileError(f'cannot read profile/receipt {path}: {exc}') from exc
    if not isinstance(result, dict):
        raise ProfileError(f'profile/receipt must be an object: {path}')
    return result


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ProfileError(f'{label}: expected lowercase SHA256')
    return value


def _file(item: Any, base: Path, label: str) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get('path'), str):
        raise ProfileError(f'{label}: expected path and sha256')
    path = Path(item['path'])
    path = (base / path).resolve() if not path.is_absolute() else path.resolve()
    expected = _hash(item.get('sha256'), label)
    try:
        if not path.is_file() or sha256(path) != expected:
            raise ProfileError(f'{label}: missing file or SHA256 mismatch: {path}')
    except OSError as exc:
        raise ProfileError(f'{label}: {exc}') from exc
    return path


def load_profile(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    doc = _read_json(path)
    if doc.get('schema_version') != 1:
        raise ProfileError('profile schema_version must be 1')
    expected = {'system': 'Linux', 'machine': 'x86_64'}
    if doc.get('platform') != expected or platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ProfileError('profile platform must match Linux x86_64 host')
    for name in ('inputs', 'artifacts', 'native_files', 'host_packages', 'checks', 'recipes'):
        if not isinstance(doc.get(name), list):
            raise ProfileError(f'profile {name} must be a list')
        if any(not isinstance(item, dict) for item in doc[name]):
            raise ProfileError(f'profile {name} entries must be objects')
    for name in ('base_lock', 'project', 'python', 'uv', 'extension_policy'):
        if not isinstance(doc.get(name), dict):
            raise ProfileError(f'profile {name} must be an object')
    policy = doc['extension_policy']
    _hash(policy.get('policy_id'), 'extension_policy.policy_id')
    if not isinstance(policy.get('allowlist'), list):
        raise ProfileError('extension_policy.allowlist must be a list')
    for extension in policy['allowlist']:
        if not isinstance(extension, dict) or not isinstance(extension.get('name'), str):
            raise ProfileError('extension allowlist requires name and sha256')
        _hash(extension.get('sha256'), 'extension allowlist')
    return doc, path


def _inputs(doc: dict[str, Any], base: Path) -> None:
    _file(doc['base_lock'], base, 'base_lock')
    _file(doc['project'], base, 'project')
    if not doc['inputs']:
        raise ProfileError('profile requires checksummed source/build inputs')
    for item in doc['inputs']:
        _file(item, base, 'input')
        if not isinstance(item.get('kind'), str):
            raise ProfileError('input requires kind')
        if item['kind'] == 'source' and not item.get('source_revision'):
            raise ProfileError('source input requires exact source_revision')


def _lxml_materials(receipt: dict[str, Any], base: Path, wheel: Path) -> None:
    """Require retained, qualified relink material for the static native closure."""
    def retained(record, label):
        if not isinstance(record, dict):
            raise ProfileError(f'lxml receipt requires {label}')
        digest = _hash(record.get('sha256'), f'lxml {label} sha256')
        matches = [item for item in receipt['inputs'] if item.get('sha256') == digest]
        if not matches:
            raise ProfileError(f'lxml {label} must be a checksummed retained input')
        return _file(matches[0], base, f'lxml {label}')

    licenses = _read_json_list(retained(receipt.get('license_manifest'), 'license_manifest'))
    relink = receipt.get('relink')
    if not isinstance(relink, dict):
        raise ProfileError('lxml receipt requires relink materials')
    for name in ('bundle', 'manifest'):
        retained(relink.get(name), f'relink {name}')
    result = _read_json(retained(relink.get('qualification'), 'relink qualification'))
    if result.get('status') != 'passed' or result.get('relocated_bundle') is not True:
        raise ProfileError('lxml relocated relink qualification did not pass')
    _hash(result.get('replacement_library_sha256'), 'lxml replacement library sha256')
    outputs = result.get('outputs')
    if (not isinstance(outputs, list) or not outputs
            or not all(isinstance(item, dict) and isinstance(item.get('name'), str)
                       for item in outputs)):
        raise ProfileError('lxml relink qualification requires extension outputs')
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.endswith('.so')]
        native = {Path(name).name: hashlib.sha256(archive.read(name)).hexdigest() for name in members}
        if (len(native) != len(members) or len(outputs) != len(native)
                or {item['name'] for item in outputs} != set(native)):
            raise ProfileError('lxml relink outputs differ from the wheel extensions')
        for item in outputs:
            if (item.get('original_sha256') != native[item['name']]
                    or item.get('relinked_sha256') != native[item['name']]):
                raise ProfileError('lxml relink output differs from the admitted wheel')
        required = {'libiconv-COPYING.LIB', 'libcharset-COPYING.LIB', 'NOTICE.txt',
                    'zlib-LICENSE', 'libxml2-Copyright', 'libxslt-Copyright'}
        if (len(licenses) != len(required)
                or not all(isinstance(item.get('path'), str) for item in licenses)
                or {Path(item['path']).name for item in licenses} != required):
            raise ProfileError('lxml native license manifest is incomplete')
        for item in licenses:
            retained(item, 'native license')
            members = [name for name in archive.namelist()
                       if name.endswith('/' + Path(item['path']).name)]
            if len(members) != 1 or hashlib.sha256(archive.read(members[0])).hexdigest() != item['sha256']:
                raise ProfileError('lxml native license differs from the wheel')


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ProfileError(f'lxml invalid native license manifest: {path}') from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ProfileError('lxml native license manifest must contain file records')
    return value


def _arrow_security(receipt: dict[str, Any], base: Path, version: str) -> None:
    """Bind this Arrow25 candidate to both ordered fixes and the exact recipe.

    Retained inputs may have moved during profile assembly. Nested manifests
    remain byte-identical, so resolve their descriptors by retained SHA256.
    These operator-reviewed receipts still require actual runtime qualification.
    """
    def retained(record, label, expected=None):
        if not isinstance(record, dict) or not isinstance(record.get('path'), str):
            raise ProfileError(f'Arrow receipt requires {label}')
        digest = _hash(record.get('sha256'), f'Arrow {label}')
        if expected is not None and digest != expected:
            raise ProfileError(f'Arrow {label} differs from the reviewed input')
        matches = [item for item in receipt['inputs'] if item.get('sha256') == digest]
        if not matches:
            raise ProfileError(f'Arrow {label} must be a retained checksummed input')
        return _file(matches[0], base, f'Arrow {label}')

    if receipt['source_revision'] != ARROW_SOURCE_REVISION:
        raise ProfileError('Arrow receipt source revision differs from the reviewed source')
    retained(receipt.get('recipe'), 'recipe', ARROW_RECIPE_SHA256)
    application = _read_json(retained(receipt.get('patch_application'), 'patch application'))
    patches = application.get('patches')
    if (type(application.get('schema_version')) is not int
            or application['schema_version'] != 1
            or application.get('source_revision') != ARROW_SOURCE_REVISION
            or not isinstance(patches, list) or len(patches) != 2
            or any(not isinstance(item, dict) for item in patches)):
        raise ProfileError('Arrow patch application requires both ordered reviewed patches')
    expected_stages = (
        (ARROW_SECURITY_PATCH_COMMIT, ARROW_SECURITY_PATCH_SHA256, ARROW_SECURITY_FILES),
        (ARROW_OFFSET_PATCH_COMMIT, ARROW_OFFSET_PATCH_SHA256, ARROW_OFFSET_FILES),
    )
    final_hashes = {}
    for patch, (commit, patch_sha256, source_pins) in zip(patches, expected_stages):
        if patch.get('commit') != commit:
            raise ProfileError('Arrow security patch order or commit differs from the reviewed fixes')
        retained(patch.get('patch'), 'security patch', patch_sha256)
        files = patch.get('files')
        expected_files = [{'relative': relative, **pins}
                          for relative, pins in source_pins.items()]
        if (not isinstance(files, list) or len(files) != len(expected_files)
                or any(not isinstance(item, dict) for item in files)
                or any(files.count(item) != 1 for item in expected_files)):
            raise ProfileError('Arrow patch application lacks the exact reviewed source hashes')
        for relative, pins in source_pins.items():
            if relative in final_hashes and final_hashes[relative] != pins['original_sha256']:
                raise ProfileError('Arrow reviewed patch source transitions do not join')
            final_hashes[relative] = pins['patched_sha256']
    final_sources = application.get('final_sources')
    expected_final = [{'relative': relative, 'sha256': digest}
                      for relative, digest in final_hashes.items()]
    if (not isinstance(final_sources, list) or len(final_sources) != len(expected_final)
            or any(not isinstance(item, dict) for item in final_sources)
            or any(final_sources.count(item) != 1 for item in expected_final)):
        raise ProfileError('Arrow patch application lacks the exact final source inventory')
    # The overlapping regression file has an intermediate hash after stage1.
    # It is historical evidence; only the final state must remain on disk.
    for item in expected_final:
        retained({'path': item['relative'], 'sha256': item['sha256']}, 'patched source')

    build = _read_json(retained(receipt.get('build_inputs'), 'build inputs'))
    if (type(build.get('schema_version')) is not int or build['schema_version'] != 1
            or build.get('arrow_commit') != ARROW_SOURCE_REVISION
            or build.get('arrow_archive_sha256') != ARROW_ARCHIVE_SHA256):
        raise ProfileError('Arrow build inputs differ from the reviewed source archive')
    retained(build.get('source_archive'), 'source archive', ARROW_ARCHIVE_SHA256)
    retained(build.get('recipe'), 'build recipe', ARROW_RECIPE_SHA256)
    retained(build.get('patch_application'), 'build patch application',
             receipt['patch_application']['sha256'])
    if build.get('security_patches') != application:
        raise ProfileError('Arrow build inputs do not bind the security patch application')
    sources = build.get('patched_sources')
    expected_hashes = set(final_hashes.values())
    if (not isinstance(sources, list) or len(sources) != len(expected_hashes)
            or any(not isinstance(item, dict) for item in sources)
            or {_hash(item.get('sha256'), 'Arrow build source') for item in sources}
            != expected_hashes):
        raise ProfileError('Arrow build inputs do not bind every patched source')
    for item in sources:
        retained(item, 'build patched source')
    expected_version = '25.0.1+biominer.' + receipt['build_inputs']['sha256'][:12]
    if (version != expected_version or receipt.get('version') != expected_version
            or receipt['effective_versions'].get('pyarrow') != expected_version):
        raise ProfileError('Arrow wheel version does not match the retained build inputs')


def _arrow_private_cmake(prefix: Path, openssl: Path) -> bytes:
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




def _arrow_network_evidence(raw, manifest, pins, take, read, tree, rows, require):
    """Retain the reviewed prefix's complete byte closure without running it."""
    network = manifest.get('network')
    require(isinstance(network, dict) and raw.get('network') == network,
            'private network descriptor is missing or inconsistent')
    require(isinstance(ARROW_NETWORK_HELPER_SHA256, str) and isinstance(ARROW_NETWORK_BUILD_RECEIPT_SHA256, str),
            'private network helper/build review and exact pins are pending')
    helper = take(network.get('helper'), 'private network helper', ARROW_NETWORK_HELPER_SHA256)
    build = read(network.get('receipt'), 'private network receipt', ARROW_NETWORK_BUILD_RECEIPT_SHA256)
    require(network.get('build') == build
            and (build.get('schema_version'), build.get('kind'), build.get('status'), build.get('ready'))
            == (1, 'private_network_static_build', 'compiled_unverified', False),
            'private network build identity differs')
    prefix = network.get('prefix')
    require(isinstance(prefix, str) and Path(prefix).is_absolute()
            and build.get('prefix') == prefix and manifest.get('context', {}).get('network_prefix') == prefix
            and Path(network['receipt']['path']).parent / 'prefix' == Path(prefix),
            'private network prefix identity differs')
    listed = rows(network.get('files'), 'complete private network closure')
    closure = {}
    for reference in listed:
        require(isinstance(reference.get('path'), str) and reference['path'] not in closure,
                'private network closure has duplicate/malformed paths')
        take(reference, 'private network closure')
        closure[reference['path']] = reference['sha256']
    require(closure.get(network['receipt']['path']) == network['receipt']['sha256'],
            'private network closure omits its original receipt')
    private = network.get('private_files')
    require(isinstance(private, dict), 'private network ownership is missing')
    for path, metadata in private.items():
        require(isinstance(path, str) and Path(path).is_absolute() and Path(path).is_relative_to(prefix)
                and isinstance(metadata, dict) and metadata.get('component') in ARROW_NETWORK_COMPONENTS,
                'private network ownership path/component differs')
        require((metadata.get('source_package'), metadata.get('source_version'))
                == ARROW_NETWORK_COMPONENTS[metadata['component']]
                and closure.get(path) == metadata.get('sha256'), 'private network source ownership differs')
    required = [str(Path(prefix) / 'lib' / name) for name in ARROW_NETWORK_ARCHIVES]
    required += [str(Path(prefix) / 'include' / name) for name in
                 ('curl/curl.h', 'libxml2/libxml/parser.h', 'iconv.h', 'zlib.h', 'nghttp2/nghttp2.h')]
    require(all(path in private for path in required), 'private network closure omits required archives/headers')
    work = Path(network['receipt']['path']).parent
    parsed = set()

    def consume(reference):
        if isinstance(reference.get('path'), str) and Path(reference['path']).is_relative_to(work):
            require(closure.get(reference.get('path')) == reference.get('sha256'),
                    'nested private network evidence is absent from its closure')
        path = take(reference, 'nested private network evidence')
        # External source/advisory receipts remain exact leaf evidence. Their
        # observations are historical, while the helper's descriptor enumerates
        # every physically consumed source/build/provider byte separately.
        if (path.suffix == '.json' and Path(reference['path']).is_relative_to(work)
                and reference['path'] not in parsed):
            parsed.add(reference['path'])
            document = json.loads(path.read_bytes())
            require(isinstance(document, (dict, list)), 'nested private network JSON must be an object or array')
            walk(document)
        return path

    def walk(value, key=''):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                consume(value)
            # original_source is historical: its file bytes are in the original
            # archive, not at the final patched source root.
            if key != 'original_source' and isinstance(value.get('root'), str) and isinstance(value.get('files'), list):
                converted = []
                for row in value['files']:
                    require(isinstance(row, dict), 'private network inventory row is malformed')
                    if row.get('relative') == '.' and row.get('kind') == 'directory':
                        continue
                    converted.append({**row, 'type': row.get('kind', row.get('type', 'file'))})
                tree(value['root'], converted, 'private network inventory')
                for row in converted:
                    if row['type'] in ('file', 'generated'):
                        require(closure.get(str(Path(value['root']) / row['relative'])) == row['sha256'],
                                'private network inventory leaf is absent from its closure')
            for name, child in value.items():
                if key == 'original_source' and name == 'files':
                    continue
                walk(child, name)
        elif isinstance(value, list):
            for child in value:
                walk(child, key)

    walk(build)
    take(build.get('recipe'), 'private prefix captured recipe', ARROW_NETWORK_HELPER_SHA256)
    require(build.get('openssl', {}).get('prefix') == manifest['context']['openssl_prefix'],
            'private network selected different OpenSSL')
    for row in manifest['openssl']['static_archives']:
        require(closure.get(row['path']) == row['sha256'], 'private network closure omits selected OpenSSL')
    patch = build.get('zlib_patch')
    require(isinstance(patch, dict) and patch.get('patch_commit') == LXML_ZLIB_PATCH_COMMIT
            and patch.get('original_gzwrite_sha256') == ARROW_NETWORK_ZLIB_ORIGINAL_SHA256
            and patch.get('patched_gzwrite_sha256') == ARROW_NETWORK_ZLIB_PATCHED_SHA256,
            'private zlib patch/source chain differs')
    take(patch.get('patch'), 'private zlib patch', ARROW_NETWORK_ZLIB_PATCH_SHA256)
    ngpatch = build.get('nghttp2_patch')
    require(isinstance(ngpatch, dict) and ngpatch.get('source_package') == 'nghttp2'
            and ngpatch.get('source_version') == '1.68.0-2ubuntu0.2',
            'private nghttp2 distribution source identity differs')
    original = ngpatch.get('original_source')
    require(isinstance(original, dict) and isinstance(original.get('root'), str),
            'nghttp2 original source inventory is absent')
    old_files = {row.get('relative'): row for row in rows(original.get('files'), 'original nghttp2 inventory')}
    original_archive = take(ngpatch.get('archive'), 'original nghttp2 archive')
    # Check original inventory hashes directly against retained archive bytes.
    # This remains valid after the original patched build tree is removed.
    with tarfile.open(original_archive) as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            require(parts and not Path(member.name).is_absolute() and '..' not in parts,
                    'nghttp2 source archive has an unsafe path')
            relative = Path(*parts[1:]).as_posix()
            row = old_files.get(relative)
            if member.isfile():
                stream = archive.extractfile(member)
                require(isinstance(row, dict) and row.get('sha256') == hashlib.sha256(stream.read()).hexdigest(),
                        'historical nghttp2 source differs from its original archive')
    stages = rows(ngpatch.get('stages'), 'ordered nghttp2 source patches')
    final_inventory = ngpatch.get('final_source')
    require(isinstance(final_inventory, dict), 'final nghttp2 source inventory is absent')
    final = {row.get('relative'): row.get('sha256') for row in rows(
        final_inventory.get('files'), 'final nghttp2 source inventory') if row.get('kind') == 'file'}
    current = {name: row.get('sha256') for name, row in old_files.items() if row.get('kind') == 'file'}
    for stage in stages:
        take(stage.get('patch'), 'nghttp2 ordered patch')
        command = read(stage.get('command'), 'nghttp2 patch command')
        require(command.get('exit_code') == 0 and command.get('cleanup') == 'terminal',
                'nghttp2 patch command lacks successful terminal evidence')
        before, after = stage.get('before_files'), stage.get('after_files')
        require(isinstance(before, dict) and isinstance(after, dict) and set(before) == set(after)
                and all(current.get(name) == value for name, value in before.items()),
                'nghttp2 ordered patch chain is discontinuous')
        current.update(after)
    require(all(final.get(name) == value for name, value in current.items()),
            'nghttp2 final source differs from ordered patches')
    compiled = build.get('nghttp2_build')
    require(isinstance(compiled, dict), 'nghttp2 source-built PIC output evidence is absent')
    for name in ('archive', 'public_header', 'version_template', 'generated_version_header',
                 'compile_commands', 'target_source_manifest'):
        take(compiled.get(name), 'nghttp2 compiled ' + name)
    for name in ('objects', 'compiled_sources', 'installed'):
        for item in rows(compiled.get(name), 'nghttp2 compiled ' + name):
            take(item, 'nghttp2 compiled ' + name)
    command_docs = {}
    for reference in rows(build.get('commands'), 'private build commands'):
        command = read(reference, 'private build command')
        require(command.get('exit_code') == 0 and command.get('cleanup') == 'terminal',
                'private build command lacks successful terminal evidence')
        if isinstance(command.get('log'), dict):
            take(command['log'], 'private build command log')
        command_docs[Path(reference['path']).name] = command
    for name in ('probe-whole-archive-pic-link', 'probe-charset-whole-archive-pic-link'):
        command = command_docs.get(name + '-command.json', {})
        args = command.get('argv', [])
        require(isinstance(args, list) and '-shared' in args and '-Wl,-z,defs' in args
                and '-Wl,-z,text' in args and '-Wl,--whole-archive' in args,
                'private static whole-archive PIC proof is absent')
    linkage = build.get('linkage')
    require(isinstance(linkage, dict) and linkage.get('dynamic_private_dependencies') == [],
            'private prefix dynamic linkage proof differs')
    observation = {'curl': '8.23.0', 'openssl': '3.6.5', 'zlib': '1.3.2', 'libxml2': '2.15.4',
                   'libiconv': '1.19', 'nghttp2': '1.68.0',
                   **{key: True for key in ('http', 'https', 'http2', 'xml', 'xpath', 'encoding', 'gzip_xml')}}
    require(build.get('probe', {}).get('observation') == observation and build.get('effective_versions') == observation,
            'private prefix HTTP/TLS/HTTP2/compression/XML capability evidence differs')
    metadata_key = 'provider_cmake' if pins['version'].startswith('26') else 'private_network_cmake'
    metadata = take(manifest.get(metadata_key), 'private network CMake metadata')
    require(metadata.read_bytes() == _arrow_private_cmake(Path(prefix), Path(manifest['context']['openssl_prefix'])),
            'private network static target metadata differs from the reviewed interfaces')
    return network


def _arrow_retained_symlinks(inventory, label, require):
    """Resolve inventory links component by component without historical I/O."""
    directories = set()
    for name in inventory:
        parts = name.split('/')
        require(all(part not in ('', '.', '..') for part in parts), label + ' has unsafe symlink paths')
        directories.update('/'.join(parts[:number]) for number in range(1, len(parts)))
    for name in directories:
        row = inventory.get(name)
        require(row is None or row.get('type', 'file') == 'directory',
                label + ' source tree has a non-directory inventory parent')
    for name, row in inventory.items():
        if row.get('type', 'file') != 'symlink':
            continue
        pending, resolved, active = deque(name.split('/')), [], set()
        while pending:
            component = pending.popleft()
            if isinstance(component, tuple):
                active.remove(component[1])
                continue
            if component in ('', '.'):
                continue
            if component == '..':
                require(bool(resolved), label + ' symlink chain escapes source tree')
                resolved.pop()
                continue
            candidate = '/'.join([*resolved, component])
            target_row = inventory.get(candidate)
            kind = target_row.get('type', 'file') if target_row is not None else 'directory'
            if kind == 'symlink':
                target = target_row.get('target')
                require(isinstance(target, str) and bool(target) and not target.startswith('/')
                        and '\\' not in target and '\x00' not in target,
                        label + ' symlink target is unsafe')
                require(candidate not in active, label + ' symlink chain is cyclic')
                active.add(candidate)
                # Expand links before later '..' components; normalizing a
                # whole target first would discard directory-link semantics.
                pending.extendleft(reversed([*target.split('/'), ('complete', candidate)]))
            else:
                # Preserve direct-mode handling of dangling in-root targets;
                # source/archive equality is checked independently.
                resolved.append(component)


def _arrow_sdk_source_evidence(raw, family, pins, manifest, selected, attestation,
                               retained_inputs, take, read, tree, rows, require):
    """Verify a separately reviewed proof without rewriting frozen build history."""
    require(isinstance(pins['attestation_sha256'], str)
            and isinstance(ARROW_SOURCE_ATTESTER_SHA256, str),
            'source attestation and verifier review pins are pending')
    document = read(attestation, 'source attestation', pins['attestation_sha256'])
    require(document.get('schema_version') == 1
            and document.get('kind') == 'bundled_native_source_attestation'
            and document.get('status') == 'source_attested' and document.get('ready') is False
            and document.get('schema_kind') == pins['schema_kind']
            and document.get('source_policy') == family,
            'source attestation has the wrong family or status')
    subject = document.get('subject')
    require(isinstance(subject, dict), 'source attestation subject is missing')
    require(isinstance(pins['raw_sha256'], str), 'completed raw receipt review pin is pending')
    require(read(subject.get('raw_receipt'), 'attested raw receipt', pins['raw_sha256']) == raw,
            'source attestation refers to another raw receipt')
    for name in ('recipe', 'build_inputs', 'compiled_dependencies', 'compiled_sources', 'wheel'):
        require(subject.get(name) == raw[name], 'source attestation subject differs: ' + name)
        take(subject[name], 'attested ' + name)
    cohort = read(document.get('cohort'), 'source cohort', ARROW_COHORT_METADATA_SHA256)
    expected = ARROW_SDK_COHORT
    metadata = rows(cohort.get('sources'), 'source cohort components')
    require(len(metadata) == len(expected) and {row.get('name') for row in metadata} == set(expected),
            'source cohort component set differs')
    for row in metadata:
        require({key: value for key, value in row.items() if key not in ('name', 'path')}
                == expected[row['name']], 'source cohort metadata differs')
    legacy = selected['forced_bundled_source_verification']
    supplemental = document.get('supplemental')
    require(document.get('legacy') == legacy and isinstance(supplemental, dict)
            and set(supplemental) == set(expected) and not set(legacy) & set(supplemental)
            and set(legacy) | set(supplemental)
            == set(pins['dependencies']) - ({'zlib'} if pins['schema_kind'] == '26' else set()),
            'source attestation must complete the exact disjoint bundled source set')
    for name, component in supplemental.items():
        require(isinstance(component, dict)
                and component.get('archive') == manifest['dependencies'][name]['archive'],
                name + ' attested archive differs from its owned build input')
        take(component['archive'], name + ' attested archive', expected[name]['sha256'])
        tree(component.get('source_root'), component.get('files'), name + ' attested source tree')
    # Only a reviewed standalone verifier may interpret the source derivations.
    # Load its exact bytes directly, avoiding cached bytecode or adjacent modules.
    script = take(document.get('producer'), 'source verifier', ARROW_SOURCE_ATTESTER_SHA256)
    source_bytes = script.read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest() == ARROW_SOURCE_ATTESTER_SHA256,
            'source verifier changed before loading')
    spec = importlib.util.spec_from_file_location('biominer_arrow_source_attestor', script)
    require(spec is not None and spec.loader is not None, 'source verifier cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(script), 'exec'), module.__dict__)
    verified = module.verify_completed_attestation(
        take(attestation, 'source attestation', pins['attestation_sha256']),
        pins['attestation_sha256'], retained_inputs=retained_inputs)
    require(script.read_bytes() == source_bytes, 'source verifier changed during verification')
    require(isinstance(verified, dict) and verified.get('attestation') == document
            and verified.get('supplemental') == supplemental and verified.get('legacy') == legacy,
            'source verifier returned different proof identities')
    receipt = verified.get('receipt')
    require(isinstance(receipt, dict) and receipt.get('sha256') == pins['attestation_sha256'],
            'source verifier returned another attestation receipt')
    take(receipt, 'verified source attestation', pins['attestation_sha256'])
    for item in rows(verified.get('files'), 'source attestation closure'):
        take(item, 'source attestation retained input')


def _arrow_private_evidence(raw: dict[str, Any], base: Path, family: str, retained_inputs=None,
                            *, source_attestation=None) -> list[dict]:
    """Check the known 26 schemas and enumerate their exact retained byte closure.

    Nested JSON is never rewritten. During admission, historical descriptors
    resolve through the wrapper's retained hashes, including root-relative
    source inventories. Host observations remain history, not retained inputs.
    """
    pins = _arrow_private_pins(family)
    if not pins['requires_source_attestation'] and source_attestation is not None:
        raise ProfileError(pins['label'] + ' does not accept a supplemental source attestation')
    family_label = pins['label']
    collected, verified = {}, {}
    retained_by_hash = None
    if retained_inputs is not None:
        retained_by_hash = {}
        for row in retained_inputs:
            retained_by_hash.setdefault(row.get('sha256'), row)

    def require(condition, label):
        if not condition:
            raise ProfileError(family_label + ' ' + label)

    def rows(value, label, *, nonempty=True):
        require(isinstance(value, list) and (bool(value) or not nonempty)
                and all(isinstance(item, dict) for item in value), label + ' requires records')
        return value

    def take(item, label, expected=None):
        require(isinstance(item, dict) and isinstance(item.get('path'), str), label + ' requires a file')
        digest = _hash(item.get('sha256'), family_label + ' ' + label)
        require(expected is None or digest == expected, label + ' differs from reviewed bytes')
        selected = item
        if retained_by_hash is not None:
            selected = retained_by_hash.get(digest)
            require(selected is not None, label + ' must be a retained input')
        key = (selected['path'], digest)
        if key not in verified:
            verified[key] = _file(selected, base, family_label + ' ' + label)
        path = verified[key]
        record = {'path': str(path), 'sha256': digest}
        collected[(str(path), digest)] = record
        return path

    def read(item, label, expected=None):
        return _read_json(take(item, label, expected))

    def text(path, label):
        try:
            return path.read_text(errors='strict')
        except UnicodeDecodeError as exc:
            raise ProfileError(family_label + ' ' + label + ' must be UTF-8 text') from exc

    def tree(root, records, label):
        require(isinstance(root, (str, Path)) and Path(root).is_absolute(), label + ' root must be absolute')
        result = {}
        for row in rows(records, label):
            relative = row.get('relative')
            require(isinstance(relative, str) and relative not in result
                    and not Path(relative).is_absolute() and '..' not in Path(relative).parts
                    and '\\' not in relative and relative not in ('', '.'), label + ' has unsafe/duplicate paths')
            result[relative] = row
            kind = row.get('type', 'file')
            require(kind in ('file', 'generated', 'directory', 'symlink'), label + ' has unknown object type')
            if kind in ('file', 'generated'):
                take({'path': str(Path(root) / relative), 'sha256': row.get('sha256')}, label + ' source')
            elif kind == 'symlink':
                target = row.get('target')
                require(isinstance(target, str) and not Path(target).is_absolute(), label + ' has an unsafe symlink')
                if retained_by_hash is None:
                    resolved = (Path(root) / relative).parent / target
                    require(resolved.resolve().is_relative_to(Path(root).resolve()), label + ' symlink escapes source tree')
        if retained_by_hash is not None:
            _arrow_retained_symlinks(result, label, require)
        return result

    required = {'recipe', 'build_inputs', 'source_verification', 'compiled_dependencies',
                'compiled_sources', 'native_link_artifacts', 'final_native_link_artifacts',
                'linked_static_archives', 'wheel', 'wheel_sha256'}
    require(type(raw.get('schema_version')) is int and raw['schema_version'] == 1
            and raw.get('status') == 'compiled_unverified' and raw.get('name') == 'pyarrow'
            and required <= raw.keys(), 'raw receipt is incomplete or not compiled_unverified')
    require(all(isinstance(raw[name], dict) for name in (
        'recipe', 'build_inputs', 'source_verification', 'compiled_dependencies', 'compiled_sources', 'wheel')),
        'raw receipt object fields are malformed')
    require(raw.get('source_revision') == pins['source_revision'], 'source revision is unreviewed')
    manifest = read(raw['build_inputs'], 'build inputs')
    require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1
            and manifest.get('source_policy') == family
            and manifest.get('arrow_commit') == pins['source_revision'],
            'build manifest has the wrong source family/policy')
    version = pins['version'] + '+biominer.' + raw['build_inputs']['sha256'][:12]
    effective_versions = _arrow_private_effective_versions(raw, family)
    require(isinstance(effective_versions, dict), 'effective versions must be an object')
    require(raw.get('version') == version and effective_versions.get('pyarrow') == version,
            'wheel version differs from the immutable build-input digest')
    toolchain = _arrow_private_toolchain(raw, family)
    require(isinstance(toolchain, dict) and isinstance(raw.get('toolchain_versions'), dict),
            'raw receipt toolchain fields are malformed')
    take(raw['recipe'], 'recipe', pins['recipe_sha256'])
    take(manifest.get('recipe'), 'manifest recipe', pins['recipe_sha256'])
    take(manifest.get('source_archive'), 'source archive', pins['archive_sha256'])
    source = read(raw['source_verification'], 'source verification')
    require(manifest.get('source_verification') == raw['source_verification'],
            'source verification differs between receipt and manifest')
    require(isinstance(pins['transformations'], list), 'reviewed source transformations are pending')
    require(all(isinstance(value, str) and SHA.fullmatch(value) for value in pins['sources'].values()),
            'reviewed source hashes are pending')
    expected_sources = [{'relative': name, 'sha256': value} for name, value in sorted(pins['sources'].items())]
    require(source.get('schema_version') == 1 and source.get('source_revision') == pins['source_revision']
            and source.get('archive') == manifest['source_archive']
            and source.get('transformations') == pins['transformations'],
            'source verification lacks exact reviewed transformations')
    if pins['schema_kind'] == '26':
        require(source.get('integrated_sources') == expected_sources
                and manifest.get('integrated_sources') == expected_sources,
                'source verification lacks exact integrated fixes')
    else:
        require(raw.get('kind') == 'corrective_arrow25_private_network_build' and raw.get('ready') is False
                and raw.get('source_policy') == family and source.get('source_policy') == family,
                'corrective source family identity differs')
        security = source.get('security_patches')
        require(isinstance(security, dict) and manifest.get('security_patches') == security,
                'ordered compute patch evidence differs')
        application = read(manifest.get('patch_application'), 'compute patch application')
        require(application == security and raw.get('patch_application') == manifest['patch_application']
                and security.get('source_revision') == ARROW_SOURCE_REVISION,
                'compute patch application differs')
        stages = rows(security.get('patches'), 'ordered compute patches')
        require(len(stages) == 2, 'both ordered compute patches are required')
        final_hashes = {}
        for stage, (commit, digest, files) in zip(stages, (
                (ARROW_SECURITY_PATCH_COMMIT, ARROW_SECURITY_PATCH_SHA256, ARROW_SECURITY_FILES),
                (ARROW_OFFSET_PATCH_COMMIT, ARROW_OFFSET_PATCH_SHA256, ARROW_OFFSET_FILES))):
            require(stage.get('commit') == commit
                    and stage.get('files') == [{'relative': relative, **values} for relative, values in files.items()],
                    'reviewed ordered compute patch stage differs')
            take(stage.get('patch'), 'ordered compute patch', digest)
            for relative, values in files.items():
                require(relative not in final_hashes or final_hashes[relative] == values['original_sha256'],
                        'compute patch chain is discontinuous')
                final_hashes[relative] = values['patched_sha256']
        require(security.get('final_sources') == [{'relative': relative, 'sha256': value}
                for relative, value in sorted(final_hashes.items())], 'final compute patch source set differs')
        take(manifest.get('patch_tool'), 'compute patch tool')
    context = manifest.get('context')
    require(isinstance(context, dict) and isinstance(context.get('work'), str)
            and Path(context['work']).is_absolute()
            and isinstance(context.get('openssl_prefix'), str)
            and Path(context['openssl_prefix']).is_absolute(), 'build context is missing')
    source_root = Path(context['work']) / ('arrow-' + pins['source_revision'])
    initial_sources = tree(source_root, source.get('files'), 'verified Arrow tree')
    if pins['schema_kind'] == '25':
        backend = source.get('backend')
        require(isinstance(backend, dict) and backend.get('path') == 'python/pyproject.toml'
                and backend.get('backend') == '_build_backend' and backend.get('implementation') == 'scikit-build-core'
                and initial_sources.get('python/pyproject.toml', {}).get('sha256') == backend.get('sha256'),
                'Python backend differs from the exact Arrow 25 source')
    for item in [*expected_sources, *pins['transformations']]:
        require(initial_sources.get(item['relative'], {}).get('sha256') == item['sha256'],
                'verified source inventory omits a reviewed integrated/transformed file')

    builder = manifest.get('builder')
    require(isinstance(builder, dict), 'builder is missing')
    builder_doc = read(builder.get('receipt'), 'builder receipt', ARROW26_BUILDER_RECEIPT_SHA256)
    require({k: v for k, v in builder.items() if k != 'receipt'} == builder_doc
            and (builder.get('schema_version'), builder.get('kind'), builder.get('status'))
            == (1, 'arrow26_python_builder', 'frozen_verified'), 'builder differs from its approved receipt')
    for name in ('python', 'base_python', 'requirements', 'verification', 'inventory'):
        take(builder.get(name), 'builder ' + name)
    for name in ('inputs', 'installer', 'operator', 'phase3_handshake'):
        if name in builder:
            take(builder[name], 'builder ' + name)
    inventory = read(builder['inventory'], 'builder inventory')
    tree(inventory.get('root'), inventory.get('files'), 'builder inventory')
    distributions = rows(builder.get('distributions'), 'builder distributions')
    require(len(distributions) == len(ARROW26_BUILDER_PINS)
            and {row.get('name'): row.get('version') for row in distributions} == ARROW26_BUILDER_PINS
            and (pins['schema_kind'] != '26' or raw.get('python_build_distributions') == distributions),
            'builder distributions differ from approved pins')
    for row in distributions:
        for name in ('wheel', 'metadata', 'record'):
            take(row.get(name), 'builder distribution ' + name)
    require(toolchain == builder.get('toolchain')
            and set(toolchain) == {'cc', 'cxx', 'cmake', 'ninja', 'ld'}
            and raw['toolchain_versions'] == manifest.get('toolchain_versions'),
            'toolchain identity differs from the approved builder')
    for name, item in toolchain.items():
        take(item, 'toolchain ' + name)
    for name in ('python', 'cc', 'cxx'):
        selected = builder['python'] if name == 'python' else toolchain[name]
        require(context.get(name) == selected['path'], 'build context tool differs from the approved builder')
    python_runtime = manifest.get('python_runtime')
    require(isinstance(python_runtime, dict) and raw.get('python_runtime') == python_runtime
            and python_runtime.get('source_build') == 'Astral20260929',
            'private Python runtime source identity is missing or inconsistent')
    for name, expected in ARROW26_PYTHON_RUNTIME_PINS.items():
        take(python_runtime.get(name), 'Python runtime ' + name, expected)
    python_source = read(python_runtime['receipt'], 'Python runtime source receipt')
    require(python_source.get('source_build') == 'Astral20260929'
            and python_source.get('python') == python_runtime['python'] == builder['base_python']
            and python_source.get('archive') == python_runtime['archive'],
            'private Python runtime differs from its exact source receipt')
    approved_python = {row['path']: row['sha256'] for row in rows(
        python_source.get('verified_archive_files'), 'attested Python archive files')}
    python_files = rows(python_runtime.get('files'), 'selected Python runtime files')
    require(len({row.get('path') for row in python_files}) == len(python_files),
            'selected Python runtime file inventory is duplicated')
    for item in python_files:
        require(approved_python.get(item.get('path')) == item.get('sha256'),
                'private Python native/header file lacks source-archive attestation')
        take(item, 'selected Python runtime file')

    dependencies = manifest.get('dependencies')
    require(isinstance(dependencies, dict) and set(dependencies) == set(pins['dependencies']),
            'pinned dependency archive set is incomplete')
    for name, dependency_pins in pins['dependencies'].items():
        item = dependencies[name]
        require(isinstance(item, dict) and item.get('version') == dependency_pins['version']
                and item.get('sha256') == dependency_pins['sha256'], name + ' dependency pin differs')
        take(item.get('archive'), name + ' source archive', dependency_pins['sha256'])
        require(effective_versions.get(name) == dependency_pins['version'], name + ' effective version differs')
        if pins['requires_source_attestation'] and name in ARROW_SDK_COHORT:
            require(all(item.get(key) == value for key, value in ARROW_SDK_COHORT[name].items())
                    and item.get('role') == 'bundled_provider', name + ' source metadata differs')
    openssl = manifest.get('openssl')
    require(isinstance(openssl, dict) and openssl.get('version') == '3.6.5'
            and openssl.get('source_commit') == 'c8bd5a57108599ac650bbae77fcabe3109dab2e8'
            and effective_versions.get('openssl') == '3.6.5', 'private OpenSSL source identity differs')
    for name, value in ARROW26_OPENSSL_PINS.items():
        take(openssl.get(name), 'OpenSSL ' + name, value)
    private_static = rows(openssl.get('static_archives'), 'OpenSSL static archives')
    for item in private_static:
        take(item, 'OpenSSL static archive')
    require(len(private_static) == len(ARROW26_OPENSSL_STATIC)
            and {Path(row.get('path', '')).name: row.get('sha256') for row in private_static}
            == ARROW26_OPENSSL_STATIC, 'OpenSSL static archive set differs')
    tree(Path(openssl['header']['path']).parent.parent, openssl.get('headers'), 'OpenSSL headers')
    require('zlib_patch' not in manifest, 'obsolete bundled zlib patch execution is forbidden')
    take(manifest.get('cmake_file_api_query'), 'File API query')
    network = _arrow_network_evidence(raw, manifest, pins, take, read, tree, rows, require)
    network_files = network['private_files']

    budget = manifest.get('compiler_budget', {})
    require(isinstance(budget, dict) and type(budget.get('maximum_jobs')) is int
            and 1 <= budget['maximum_jobs'] <= 4
            and budget.get('outer_ninja_jobs') == budget['maximum_jobs']
            and budget.get('nested_cmake_jobs') == budget.get('nested_make_jobs') == 1,
            'compiler budget exceeds the reviewed bounds')
    for name, expected in (('features', ARROW26_FEATURES), ('codecs', ARROW26_CODECS)):
        values = manifest.get(name)
        require(isinstance(values, list) and len(values) == len(expected)
                and all(isinstance(value, str) for value in values) and set(values) == expected,
                'required ' + name + ' are missing or changed')
    python_env = {**{'PYARROW_WITH_' + name: 'ON' for name in ARROW26_FEATURES},
                  'PYARROW_WITH_GANDIVA': 'OFF', 'PYARROW_WITH_PARQUET_ENCRYPTION': 'ON',
                  'PYARROW_BUNDLE_ARROW_CPP': 'ON'}
    require(manifest.get('python_feature_environment') == python_env
            and manifest.get('version_environment') == {
                'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYARROW': '{wheel_version}'},
            'Python feature or version environment differs')
    commands = manifest.get('commands')
    require(isinstance(commands, dict) and set(commands) == {'cpp_configure', 'cpp_build', 'python_build'}
            and all(isinstance(argv, list) and all(isinstance(arg, str) for arg in argv)
                    for argv in commands.values()), 'immutable build commands are missing')
    require(commands['cpp_build'] == [builder['toolchain']['cmake']['path'], '--build',
            str(Path(context['work']) / 'cpp-build'), '--target', 'install', '--parallel',
            str(budget['maximum_jobs'])], 'C++ build command exceeds the reviewed budget or targets another build')

    def flags(argv, prefix):
        values = {}
        for arg in argv:
            if arg.startswith(prefix):
                key, separator, value = arg[len(prefix):].partition('=')
                require(bool(separator) and key not in values, 'build flags are malformed or duplicated')
                values[key] = value
        return values

    private_flags = _arrow_private_flags(context)
    common_flags = {'CMAKE_C_COMPILER': context['cc'], 'CMAKE_CXX_COMPILER': context['cxx'],
                    **{name: context['python'] for name in ('Python3_EXECUTABLE', 'Python_EXECUTABLE', 'PYTHON_EXECUTABLE')},
                    **private_flags}
    bundled = ('simdjson', 'uriparser', 'Boost') if pins['schema_kind'] == '26' else ('Boost',)
    if pins['requires_source_attestation']:
        bundled += ('AWSSDK', 'gRPC', 'Protobuf', 'absl', 'c-ares', 'Thrift', 'jemalloc', 'RapidJSON')
    cpp_expected = {
        **common_flags, **{'ARROW_' + name: 'ON' for name in ARROW26_FEATURES},
        **{'ARROW_WITH_' + name: 'ON' for name in ARROW26_CODECS},
        **{'ARROW_' + name: 'OFF' for name in ('TESTING', 'BUILD_TESTS', 'BUILD_BENCHMARKS', 'BUILD_INTEGRATION', 'GANDIVA')},
        **{name + '_SOURCE': 'BUNDLED' for name in bundled},
        'ARROW_BUILD_SHARED': 'ON', 'ARROW_BUILD_STATIC': 'OFF',
        'ARROW_DEPENDENCY_SOURCE': 'AUTO', 'ARROW_DEPENDENCY_USE_SHARED': 'OFF',
        'ARROW_JEMALLOC': 'ON', 'ARROW_MIMALLOC': 'ON', 'PARQUET_REQUIRE_ENCRYPTION': 'ON',
        'CMAKE_MAKE_PROGRAM': builder['toolchain']['ninja']['path'],
    }
    if pins['schema_kind'] == '26':
        cpp_expected['URIPARSER_BUILD_TESTS'] = 'OFF'
    elif pins['requires_source_attestation']:
        cpp_expected['CUSTOM_MEMORY_MANAGEMENT'] = 'OFF'
    cpp_flags = flags(commands['cpp_configure'], '-D')
    require(commands['cpp_configure'][:7] == [builder['toolchain']['cmake']['path'], '-S',
            str(source_root / 'cpp'), '-B', str(Path(context['work']) / 'cpp-build'), '-G', 'Ninja']
            and all(cpp_flags.get(key) == value for key, value in cpp_expected.items()),
            'C++ configure command omits or overrides reviewed features/static dependencies')
    python_expected = {**common_flags, **{'PYARROW_' + name: 'ON' for name in (
        'ACERO', 'AZURE', 'DATASET', 'FLIGHT', 'GCS', 'HDFS', 'ORC', 'PARQUET', 'S3',
        'SUBSTRAIT', 'PARQUET_ENCRYPTION')}, 'PYARROW_GANDIVA': 'OFF', 'PYARROW_BUNDLE_ARROW_CPP': 'ON'}
    packages = ['Arrow', 'ArrowCompute', 'ArrowAcero', 'ArrowDataset', 'ArrowFlight', 'ArrowSubstrait', 'Parquet']
    if pins['schema_kind'] == '26':
        packages.append('ArrowS3')
    python_expected.update({name + '_DIR': str(Path(context['work']) / 'arrow-prefix/lib/cmake' / name)
                           for name in packages})
    python_flags = flags(commands['python_build'], '--config-setting=cmake.define.')
    require(commands['python_build'][:7] == [context['python'], '-I', '-B', '-m', 'build', '--wheel', '--no-isolation']
            and commands['python_build'][-1] == str(source_root / 'python')
            and '--config-setting=build-dir=' + str(Path(context['work']) / 'python-build') in commands['python_build']
            and [arg for arg in commands['python_build'] if arg.startswith('--config-setting=build.tool-args=')]
            == ['--config-setting=build.tool-args=-j' + str(budget['maximum_jobs'])]
            and all(python_flags.get(key) == value for key, value in python_expected.items()),
            'Python build command omits reviewed features or exceeds the compiler budget')
    build = raw['build'] if pins['schema_kind'] == '26' else raw
    require(isinstance(build, dict) and all(build.get(k) == value for k, value in manifest.items())
            and build.get('build_inputs') == raw['build_inputs'] and build.get('wheel_version') == version,
            'compiled build history differs from immutable inputs')
    stages = rows(build.get('stages'), 'build stages')
    require(len(stages) == 2, 'build requires C++ and Python compilation evidence')
    for stage, name in zip(stages, ('cpp_build', 'python_build')):
        require(stage.get('exit_code') == 0 and stage.get('command') == manifest.get('commands', {}).get(name),
                'compiled stage command or result differs from immutable inputs')
        take(stage.get('log'), 'compiled stage log')

    compiled = read(raw['compiled_sources'], 'compiled sources')
    final_sources = tree(source_root, compiled.get('files'), 'compiled Arrow tree')
    require(all(final_sources.get(name) == value for name, value in initial_sources.items())
            and set(final_sources) - set(initial_sources) <= {'python/pyarrow/_generated_version.py'},
            'compiled source tree differs from the verified source tree')
    consumers = rows(compiled.get('integrated_header_consumers' if pins['schema_kind'] == '26' else 'patched_header_consumers'), 'integrated header consumers')
    expected_consumers = {str(source_root / name): value for name, value in pins['sources'].items()
                          if Path(name).name in ('scalar_if_else.cc', 'vector_replace.cc')}
    require(len(consumers) == len(expected_consumers)
            and {item.get('path'): item.get('sha256') for item in consumers} == expected_consumers,
            'compiled integrated header consumers are incomplete')
    for item in consumers:
        take(item, 'integrated header consumer')
    commands_path = take(compiled.get('compile_commands'), 'compile commands')
    try:
        commands = json.loads(commands_path.read_bytes())
    except ValueError as exc:
        raise ProfileError('Arrow 26 compile commands are malformed') from exc
    rows(commands, 'compile commands')
    units = rows(compiled.get('translation_units'), 'compiled translation units')
    unit_paths = {}
    for row in units:
        resolved = take(row, 'compiled translation unit')
        unit_paths[row['path']] = resolved
    found = set()
    for row in commands:
        filename = row.get('file')
        require(isinstance(filename, str) and isinstance(row.get('directory'), str), 'compile command path is missing')
        path = Path(filename) if Path(filename).is_absolute() else Path(row['directory']) / filename
        if str(path) in expected_consumers:
            require(str(path) in unit_paths, 'direct consumer lacks its compiled translation unit')
            found.add(str(path))
        elif str(path) in unit_paths:
            includes = set(re.findall(r'^#include "([^"]+)"', text(unit_paths[str(path)], 'translation unit'), re.M))
            found.update(includes & expected_consumers.keys())
    require(found == set(expected_consumers), 'compile commands do not bind both header consumers')

    selected = read(raw['compiled_dependencies'], 'compiled dependencies')
    require(selected.get('schema_version') == 1 and selected.get('stage') == 'compiled',
            'final compiled dependency inventory is missing')
    api = rows(selected.get('file_api'), 'File API records')
    api_records = {}
    for row in api:
        take(row, 'File API record')
        api_records[Path(row['path']).name] = row
    require(len(api_records) == len(api), 'File API filenames are ambiguous')
    documents = {name: _read_json(take(row, 'File API record')) for name, row in api_records.items()}
    indexes = [doc for name, doc in documents.items() if name.startswith('index-')]
    require(len(indexes) == 1, 'File API requires exactly one index')
    reply = indexes[0].get('reply')
    require(isinstance(reply, dict) and isinstance(reply.get('codemodel-v2'), dict),
            'File API codemodel reference is malformed')
    model_name = reply['codemodel-v2'].get('jsonFile')
    require(isinstance(model_name, str), 'File API codemodel filename is missing')
    model = documents.get(model_name, {})
    model_version = model.get('version')
    require(model.get('kind') == 'codemodel' and isinstance(model_version, dict)
            and model_version.get('major') == 2
            and type(model_version.get('minor')) is int and model_version['minor'] >= 9
            and model.get('paths') == {'source': str(source_root / 'cpp'), 'build': str(Path(context['work']) / 'cpp-build')},
            'File API codemodel/source identity differs')
    configurations = rows(model.get('configurations'), 'File API configurations')
    require(len(configurations) == 1, 'File API has multiple configurations')
    entries = (rows(configurations[0].get('targets', []), 'File API built targets', nonempty=False)
               + rows(configurations[0].get('abstractTargets', []), 'File API abstract targets', nonempty=False))
    rows(entries, 'File API target entries')
    targets = rows(selected.get('targets'), 'scoped targets')
    require(all(isinstance(row.get('id'), str) and row['id']
                and isinstance(row.get('jsonFile'), str) and row['jsonFile'] for row in entries)
            and all(isinstance(row.get('id'), str) and row['id'] for row in targets),
            'File API target IDs or filenames are malformed')
    require(len(targets) == len(entries) and len({row.get('id') for row in targets}) == len(targets),
            'scoped target inventory is incomplete or duplicated')
    expected_targets = {row.get('id'): row for row in entries}
    for target in targets:
        entry = expected_targets.get(target.get('id'), {})
        record = target.get('record', {})
        doc = documents.get(entry.get('jsonFile'), {})
        require(api_records.get(entry.get('jsonFile')) == record
                and doc.get('id') == target.get('id') and doc.get('name') == target.get('target')
                and doc.get('name') == entry.get('name') and doc.get('type') == target.get('kind'),
                'File API scoped target identity differs')
    take(selected.get('boost_selection'), 'Boost selection')
    components = selected.get('boost_components')
    require(isinstance(components, list) and all(isinstance(value, str) for value in components)
            and {'algorithm', 'crc', 'numeric/conversion', 'scope_exit',
            'throw_exception', 'tokenizer', 'locale', 'thread', 'uuid'} <= set(components),
            'actual Boost selection is incomplete')
    source_trees = rows(selected.get('source_trees'), 'dependency source trees')
    source_records = []
    for row in source_trees:
        tree(row.get('path'), row.get('files'), 'dependency source tree')
        source_records.append({json.dumps(item, sort_keys=True, separators=(',', ':'))
                               for item in row['files']})
    forced = selected.get('forced_bundled_source_verification')
    required_forced = ({'simdjson', 'uriparser', 'boost'} if pins['schema_kind'] == '26' else {'boost'})
    if not pins['requires_source_attestation']:
        required_forced = set(pins['dependencies']) - ({'zlib'} if pins['schema_kind'] == '26' else set())
    require(isinstance(forced, dict) and set(forced) == required_forced,
            'bundled dependency source verification is incomplete')
    for name, files in forced.items():
        required_records = {json.dumps(item, sort_keys=True, separators=(',', ':'))
                            for item in rows(files, name + ' source verification')}
        require(sum(required_records <= observed for observed in source_records) == 1,
                name + ' verified source lacks its exact source tree')
    if pins['requires_source_attestation']:
        _arrow_sdk_source_evidence(raw, family, pins, manifest, selected, source_attestation,
                                  retained_inputs, take, read, tree, rows, require)
    python_selected = {row['path']: row['sha256'] for row in python_files}
    private_selected = {row['path']: row['sha256'] for row in private_static}
    for name in ('native_inputs', 'static_archives', 'downloaded_archives'):
        for item in rows(selected.get(name), name, nonempty=name != 'downloaded_archives'):
            take(item, 'selected ' + name)
            provider = item.get('provider')
            if _arrow_protected_provider(item.get('path', '')):
                require(item['path'] in network_files or item['path'] in private_selected,
                        'unapproved host network/crypto provider selected')
            if provider is not None:
                require(isinstance(provider, dict), 'selected native provider must be an object')
                kind = provider.get('kind')
                if kind == 'private_openssl':
                    require(private_selected.get(item['path']) == item['sha256']
                            and provider.get('version') == '3.6.5'
                            and provider.get('receipt') == openssl['receipt']
                            and provider.get('source_commit') == openssl['source_commit'],
                            'private OpenSSL provider is not an approved source-attested archive')
                elif kind == 'private_network':
                    expected = network_files.get(item['path'])
                    require(expected is not None and item['sha256'] == expected['sha256']
                            and provider.get('file') == {key: item[key] for key in ('path', 'sha256')}
                            and provider.get('receipt') == network['receipt'] and provider.get('helper') == network['helper']
                            and all(provider.get(key) == expected[key] for key in ('component', 'source_package', 'source_version')),
                            'private network provider lacks exact source ownership')
                elif kind == 'python_runtime':
                    require(python_selected.get(item['path']) == item['sha256']
                            and provider.get('source_build') == 'Astral20260929'
                            and provider.get('receipt') == python_runtime['receipt'],
                            'private Python provider is not an approved source-attested file')
                else:
                    require(kind is None and all(isinstance(provider.get(field), str) and provider[field]
                                for field in ('package', 'version', 'source_package', 'source_version')),
                            'system native input lacks exact provider/source identity')
    linked = rows(raw['linked_static_archives'], 'linked static archives')
    linked_paths = set()
    for item in linked:
        take(item, 'linked static archive')
        linked_paths.add(item['path'])
        if _arrow_protected_provider(item['path']):
            require(item['path'] in network_files or item['path'] in private_selected,
                    'unapproved host network/crypto static provider linked')
    required_static = {str(Path(network['prefix']) / 'lib' / name) for name in ARROW_NETWORK_ARCHIVES
                       if name != 'libcharset.a'} | set(private_selected)
    require(required_static <= linked_paths, 'actual private network/OpenSSL linkage is incomplete')
    require(len([name for name in linked_paths if Path(name).name == 'libz.a']) == 1,
            'exactly one private zlib provider must be linked')
    all_maps = rows(raw['native_link_artifacts'], 'link maps')
    final_maps = rows(raw['final_native_link_artifacts'], 'final link maps')
    require(all(item in all_maps for item in final_maps), 'final maps are not retained link artifacts')
    mapped, contributed = set(), set()
    for item in all_maps:
        take(item, 'link map')
    for item in final_maps:
        name = Path(item['path'])
        work = Path(context['work'])
        require((name.is_relative_to(work / 'cpp-build') or name.is_relative_to(work / 'python-build'))
                and not any(part in {'CMakeScratch', 'CMakeTmp', '_deps'} for part in name.relative_to(work).parts)
                and re.fullmatch(r'.+\.so(?:\.[0-9]+)*\.map', name.name),
                'final map belongs to a probe or unsupported consumer')
        content = text(take(item, 'final link map'), 'link map')
        require('Linker script and memory map' in content, 'link map is not a linker output')
        stage = work / ('python-build' if name.is_relative_to(work / 'python-build') else 'cpp-build')
        for value in re.findall(r'(?:^|\s)([^\s()]+(?:\.a|\.so(?:\.[0-9]+)*))(?=[\s()]|$)', content, re.M):
            path = os.path.normpath(str(stage / value))
            if _arrow_protected_provider(path):
                require(path in network_files or path in private_selected,
                        'final map contains an unapproved host network/crypto provider')
            if path.endswith('.a'):
                mapped.add(path)
        contributed.update(os.path.normpath(str(stage / value)) for value in
                           re.findall(r'(?:^|\s)([^\s()]+\.a)\([^\s()]+\)', content, re.M))
    require(linked_paths <= mapped and required_static <= contributed,
            'final link maps omit a consumed private archive member')
    return list(collected.values())


def _arrow_private_checked(raw, base, family, retained_inputs=None, *, source_attestation=None):
    try:
        return _arrow_private_evidence(raw, base, family, retained_inputs,
                                       source_attestation=source_attestation)
    except ProfileError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, tarfile.TarError) as error:
        raise ProfileError(_arrow_private_pins(family)['label'] + ' malformed private build evidence: ' + str(error)) from error


def _arrow26_evidence(raw, base, retained_inputs=None):
    return _arrow_private_checked(raw, base, ARROW26_PRIVATE_FAMILY, retained_inputs)


def _arrow25_network_evidence(raw, base, retained_inputs=None):
    return _arrow_private_checked(raw, base, ARROW25_NETWORK_FAMILY, retained_inputs)


def _arrow_private_security(receipt, base, version, family):
    pins = _arrow_private_pins(family)
    label, field = pins['label'], pins['raw_field']
    record = receipt.get(field)
    if not isinstance(record, dict):
        raise ProfileError(label + ' requires its normalized raw receipt reference')
    if pins['raw_sha256'] is None:
        raise ProfileError(label + ' completed raw receipt review and exact pin are pending')
    if record.get('sha256') != pins['raw_sha256']:
        raise ProfileError(label + ' raw receipt differs from the independently reviewed build')
    matches = [row for row in receipt['inputs'] if row.get('sha256') == record['sha256']]
    if not matches:
        raise ProfileError(label + ' raw receipt must be a retained input')
    raw = _read_json(_file(matches[0], base, label + ' raw receipt'))
    _arrow_private_checked(raw, base, family, receipt['inputs'],
                           source_attestation=receipt.get('bundled_source_attestation'))
    if (receipt.get('source_policy') != family or receipt.get('version') != version
            or raw.get('version') != version or receipt['wheel_sha256'] != raw['wheel_sha256']
            or receipt['source_revision'] != raw['source_revision']
            or receipt['effective_versions'] != _arrow_private_effective_versions(raw, family)
            or {key: row['sha256'] for key, row in receipt['toolchain'].items()}
            != {key: row['sha256'] for key, row in _arrow_private_toolchain(raw, family).items()}
            or [row['sha256'] for row in receipt['native_link_artifacts']]
            != [row['sha256'] for row in raw['native_link_artifacts']]):
        raise ProfileError(label + ' wrapper identity differs from its retained raw receipt')


def _arrow26_security(receipt, base, version):
    family = receipt.get('source_policy', ARROW26_PRIVATE_FAMILY)
    if _arrow_private_pins(family)['schema_kind'] != '26':
        raise ProfileError('Arrow 26 receipt has the wrong source family')
    return _arrow_private_security(receipt, base, version, family)


def _arrow25_network_security(receipt, base, version):
    family = receipt.get('source_policy', ARROW25_NETWORK_FAMILY)
    if _arrow_private_pins(family)['schema_kind'] != '25':
        raise ProfileError('Arrow 25 receipt has the wrong source family')
    return _arrow_private_security(receipt, base, version, family)


def _wheel(artifact: dict[str, Any], base: Path, project: dict[str, Any]) -> Path:
    path = _file(artifact, base, f'wheel {artifact.get("name")}')
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            for name in names:
                pure = Path(name)
                if pure.is_absolute() or '..' in pure.parts or '\\' in name:
                    raise ProfileError(f'unsafe wheel member: {name}')
                if name.lower().endswith('.duckdb_extension'):
                    raise ProfileError(f'wheel bundles an external extension: {name}')
                if any(part.endswith('.data') for part in pure.parts):
                    raise ProfileError('wheel .data relocation is unsupported by this profile')
            metadata_names = [name for name in names if name.endswith('.dist-info/METADATA')]
            wheel_names = [name for name in names if name.endswith('.dist-info/WHEEL')]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                raise ProfileError(f'invalid wheel metadata: {path}')
            metadata = email.message_from_bytes(archive.read(metadata_names[0]))
            name = metadata['Name'].lower().replace('_', '-')
            if name != artifact['name'] or metadata['Version'] != artifact['version']:
                raise ProfileError(f'wheel identity mismatch: {path}')
            tags = email.message_from_bytes(archive.read(wheel_names[0])).get_all('Tag', [])
            supported = {'cp314-cp314-linux_x86_64', 'cp314-cp314-manylinux_2_39_x86_64', 'cp314-cp314-manylinux_2_28_x86_64', 'py3-none-any'}
            if not tags or not any(tag in supported for tag in tags):
                raise ProfileError(f'unsupported wheel platform/ABI: {tags}')
            if name in VERSIONS and not any(tag.startswith('cp314-cp314-') for tag in tags):
                raise ProfileError(f'native wheel must target CPython 3.14: {name}')
    except (OSError, zipfile.BadZipFile, KeyError, AttributeError) as exc:
        raise ProfileError(f'invalid wheel {path}: {exc}') from exc
    receipt_path = _file(artifact.get('source_receipt'), base, 'source_receipt')
    receipt = _read_json(receipt_path)
    for key in ('wheel_sha256', 'source_revision', 'toolchain', 'inputs', 'inputs_sha256', 'native_link_artifacts', 'effective_versions'):
        if key not in receipt:
            raise ProfileError(f'source receipt requires {key}: {receipt_path}')
    if receipt['wheel_sha256'] != artifact['sha256']:
        raise ProfileError('source receipt wheel_sha256 mismatch')
    if not isinstance(receipt['source_revision'], str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', receipt['source_revision']):
        raise ProfileError('source receipt requires exact source revision/hash')
    if not isinstance(receipt['toolchain'], dict) or not receipt['toolchain']:
        raise ProfileError('source receipt requires checksummed toolchain')
    for tool, item in receipt['toolchain'].items():
        _file(item, receipt_path.parent, f'toolchain {tool}')
    if not isinstance(receipt['inputs'], list) or not receipt['inputs']:
        raise ProfileError('source receipt requires checksummed inputs')
    for item in receipt['inputs']:
        _file(item, receipt_path.parent, 'receipt input')
    _hash(receipt['inputs_sha256'], 'receipt inputs_sha256')
    actual_inputs = hashlib.sha256(json.dumps(receipt['inputs'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if receipt['inputs_sha256'] != actual_inputs:
        raise ProfileError('receipt inputs_sha256 mismatch')
    if name == 'biominer':
        overlays = [item for item in receipt['inputs'] if Path(item['path']).name == 'pyproject.toml']
        if len(overlays) != 1:
            raise ProfileError('BioMiner source receipt requires exactly one pyproject.toml overlay')
        if overlays[0]['sha256'] != project.get('sha256'):
            raise ProfileError('BioMiner source receipt pyproject.toml does not match profile project')
    if not isinstance(receipt['native_link_artifacts'], list) or (name in VERSIONS and not receipt['native_link_artifacts']):
        raise ProfileError('source receipt requires checksummed native_link_artifacts')
    for item in receipt['native_link_artifacts']:
        _file(item, receipt_path.parent, 'native_link_artifacts')
    if not isinstance(receipt['effective_versions'], dict):
        raise ProfileError('source receipt effective_versions must be an object')
    if name in ('pyarrow', 'duckdb'):
        version = receipt['effective_versions'].get('openssl')
        if not isinstance(version, str) or not re.fullmatch(r'3\.6\.(?:[5-9]|[1-9][0-9]+)', version):
            raise ProfileError(f'{name} receipt requires patched OpenSSL 3.6.5+ on the approved line')
    if name == 'pyarrow':
        if artifact['version'].startswith('26.0.0+biominer.'):
            _arrow26_security(receipt, receipt_path.parent, artifact['version'])
        else:
            if ('arrow25_network_build_receipt' in receipt or 'arrow25_cohort_build_receipt' in receipt
                    or receipt.get('source_policy') in ARROW_PRIVATE_FAMILIES):
                _arrow25_network_security(receipt, receipt_path.parent, artifact['version'])
            else:
                _arrow_security(receipt, receipt_path.parent, artifact['version'])
    if name == 'lxml':
        if any(receipt['effective_versions'].get(key) != value
               for key, value in LXML_NATIVE_VERSIONS.items()):
            raise ProfileError('lxml receipt requires the reviewed patched native versions')
        policy = receipt.get('native_policy')
        if (not isinstance(policy, dict)
                or policy.get('zlib_patch_commit') != LXML_ZLIB_PATCH_COMMIT
                or policy.get('zlib_patch_sha256') != LXML_ZLIB_PATCH_SHA256
                or not any(item.get('sha256') == LXML_ZLIB_PATCH_SHA256
                           for item in receipt['inputs'])):
            raise ProfileError('lxml receipt requires the checksummed zlib security patch')
        features = policy.get('libxml_features')
        if (not isinstance(features, list)
                or not all(isinstance(item, str) for item in features)
                or not LXML_FEATURES <= set(features)):
            raise ProfileError('lxml receipt requires the reviewed libxml features')
        _lxml_materials(receipt, receipt_path.parent, path)
    return path


def verify_artifacts(profile: str | Path) -> dict[str, Any]:
    doc, path = load_profile(profile)
    _inputs(doc, path.parent)
    seen: set[str] = set()
    for artifact in doc['artifacts']:
        if not isinstance(artifact, dict) or not isinstance(artifact.get('name'), str):
            raise ProfileError('profile artifact requires name')
        name = artifact['name']
        if name in seen or name == 'psycopg-binary':
            raise ProfileError(f'duplicate or prohibited artifact: {name}')
        seen.add(name)
        if name in VERSIONS:
            version = artifact.get('version')
            if name == 'pyarrow':
                accepted = isinstance(version, str) and (
                    re.fullmatch(r'25\.0\.1\+biominer\.(?:1|[0-9a-f]{12})', version)
                    or re.fullmatch(r'26\.0\.0\+biominer\.[0-9a-f]{12}', version))
            else:
                accepted = version == VERSIONS[name] if name in ('psycopg-c', 'lxml') else isinstance(version, str) and re.fullmatch(re.escape(VERSIONS[name].split('+')[0]) + r'\+biominer\.(?:1|[0-9a-f]{12})', version)
            if not accepted:
                raise ProfileError(f'unapproved native artifact version: {name}')
        _wheel(artifact, path.parent, doc['project'])
    if seen != set(VERSIONS) | {'biominer'}:
        raise ProfileError('profile requires exactly psycopg-c, pyarrow, duckdb, lxml, biominer wheels')
    return {'status': 'candidate', 'ready': False, 'profile_sha256': sha256(path), 'artifacts': sorted(seen)}


def launch_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    source = dict(os.environ if source is None else source)
    for key in BLOCKED_ENV:
        if source.get(key):
            raise ProfileError(f'unapproved loader environment: {key}')
    # Discard inherited uv/pip settings, shell startup paths, and user-site state.
    environment = {key: value for key, value in source.items() if not key.startswith(('UV_', 'PIP_', 'PYTHON', 'LD_')) and key not in ('VIRTUAL_ENV', 'HOME')}
    environment['PYTHONNOUSERSITE'] = '1'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['HOME'] = '/nonexistent'
    environment['GDAL_DRIVER_PATH'] = 'disable'
    return environment


def _process_group_live(group: int, session: int) -> bool:
    """Observe only the original group/session; deliberately escaped sessions are outside this contract."""
    live = False
    try:
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            pid = int(entry.name)
            try:
                # Avoid reading stat for unrelated processes, including private
                # service managers. A disappearing process is already terminal.
                observed_group, observed_session = os.getpgid(pid), os.getsid(pid)
                if observed_group != group and observed_session != session:
                    continue
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                state, current_group, current_session = fields[0], int(fields[2]), int(fields[3])
                if len(state) != 1:
                    raise ValueError('invalid process state')
                # Membership can change between the syscalls and stat read.
                if current_group != group and current_session != session:
                    continue
                live |= state not in {'Z', 'X'}
            except (ProcessLookupError, FileNotFoundError):
                continue
    except (OSError, ValueError, IndexError) as exc:
        raise ProfileError('command cleanup cannot inspect original process group/session') from exc
    return live


def _bounded_process(argv: list[str], *, cwd: Path | None, env: dict[str, str], timeout: int, output: Any = subprocess.PIPE) -> str:
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    # start_new_session makes the original PID both process-group and session
    # leader. Keep those identities even after communicate reaps that leader.
    group = session = process.pid
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except BaseException:
        # The recipe may have spawned pip/CMake/compiler workers. Killing just
        # its interpreter would leave those workers modifying failed outputs.
        deadline = time.monotonic() + 5.0
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise ProfileError('command cleanup cannot signal original process group') from exc
        try:
            process.communicate(timeout=max(0.0, deadline - time.monotonic()))
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise ProfileError('command cleanup could not finish leader communication') from exc
        # Reaping the leader does not wait for file-log writers in its group.
        # Share one deadline with communicate rather than renewing the budget.
        while _process_group_live(group, session):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProfileError('command cleanup did not establish terminal process group/session')
            time.sleep(min(0.01, remaining))
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, argv)
    return stdout or ''


def _run(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 300, log: Path | None = None) -> str:
    try:
        if log is not None:
            with log.open('x') as target:
                _bounded_process(argv, cwd=cwd, env=env or launch_environment(), timeout=timeout, output=target)
            return ''
        result = _bounded_process(argv, cwd=cwd, env=env or launch_environment(), timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProfileError(f'command timed out after {timeout}s: {argv[0]}') from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        # Do not echo subprocess output, which may contain connection secrets.
        raise ProfileError(f'command failed: {argv[0]} ({getattr(exc, "returncode", "unavailable")})') from exc
    return result


def _host(doc: dict[str, Any]) -> list[dict[str, str]]:
    try:
        release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    except OSError as exc:
        raise ProfileError('cannot attest host release') from exc
    if release.get('ID', '').strip('"') != 'ubuntu' or release.get('VERSION_ID', '').strip('"') != '26.04':
        raise ProfileError('host requires Ubuntu 26.04 security package provenance')
    required = {'libssl3t64', 'libssl-dev', 'openssl', 'libpq5'}
    names = {item.get('name') for item in doc['host_packages'] if isinstance(item, dict)}
    if not required <= names:
        raise ProfileError('host_packages requires libssl3t64, libssl-dev, openssl, libpq5')
    global_policy = _run(['/usr/bin/apt-cache', 'policy'])
    if 'o=Ubuntu' not in global_policy:
        raise ProfileError('host apt metadata lacks Ubuntu Origin')
    observed = []
    for item in doc['host_packages']:
        if item.get('origin') != 'Ubuntu' or item.get('architecture') != 'amd64' or not item.get('advisory'):
            raise ProfileError('host package requires Ubuntu amd64 origin and advisory')
        name = item['name']
        if not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', name):
            raise ProfileError('invalid host package name')
        minimum = item.get('minimum_version')
        if not isinstance(minimum, str) or not minimum:
            raise ProfileError('host package requires minimum_version')
        if name in {'libssl3t64', 'libssl-dev', 'openssl'}:
            try:
                _run(['/usr/bin/dpkg', '--compare-versions', minimum, 'ge', '3.5.5-1ubuntu3.7'])
            except ProfileError as exc:
                raise ProfileError(f'host {name} profile minimum {minimum} is below reviewed 3.5.5-1ubuntu3.7 floor') from exc
        output = _run(['/usr/bin/dpkg-query', '-W', '-f=${Version}\t${Architecture}\t${Status}', name]).strip().split('\t')
        if len(output) != 3 or output[1] != 'amd64' or output[2] != 'install ok installed':
            raise ProfileError(f'host package unavailable: {name}')
        try:
            _run(['/usr/bin/dpkg', '--compare-versions', output[0], 'ge', item['minimum_version']])
        except ProfileError as exc:
            raise ProfileError(f'host {name} installed {output[0]} below required {minimum}; operator update required') from exc
        policy = _run(['/usr/bin/apt-cache', 'policy', name])
        stanza = []
        inside = False
        for line in policy.splitlines():
            header = re.fullmatch(r'\s*(\*\*\*\s+)?(\S+)\s+\d+\s*', line)
            if header:
                if inside:
                    break
                inside = bool(header.group(1)) and header.group(2) == output[0]
            elif inside:
                stanza.append(line)
        if not inside:
            raise ProfileError(f'host installed-version apt evidence absent: {name}')
        repositories = re.findall(r'\b(https?://\S+)\s+resolute(?:-[a-z]+)?/', '\n'.join(stanza))
        official = re.compile(r'https?://(?:(?:[a-z]{2}\.)?archive\.ubuntu\.com|security\.ubuntu\.com|ports\.ubuntu\.com)/ubuntu/?$')
        if not repositories or any(not official.fullmatch(url) for url in repositories):
            raise ProfileError(f'host installed-version lacks exclusive official Ubuntu origin: {name}')
        # Require receipt pinning of the policy that established package origin.
        if not isinstance(item.get('apt_policy_sha256'), str) or hashlib.sha256(policy.encode()).hexdigest() != item['apt_policy_sha256']:
            raise ProfileError(f'host apt origin evidence changed or absent: {name}')
        observed.append({'name': name, 'version': output[0], 'architecture': output[1]})
    return observed


def _load_python_attestor(record: dict[str, Any], base: Path):
    script = _file(record, base, 'Python runtime attestor')
    spec = importlib.util.spec_from_file_location('biominer_python_attestor', script)
    if spec is None or spec.loader is None:
        raise ProfileError('cannot load Python runtime attestor')
    module = importlib.util.module_from_spec(spec)
    exec(compile(script.read_bytes(), str(script), "exec"), module.__dict__)
    return module


def verify_python_prefix(doc: dict[str, Any], base: Path, python: Path) -> None:
    receipts = [item for item in doc['inputs'] if item.get('kind') == 'python-source-receipt']
    attestors = [item for item in doc['inputs'] if Path(item.get('path', '')).name == 'attest_native_python.py']
    if len(receipts) != 1 or len(attestors) != 1:
        raise ProfileError('Python runtime requires one source receipt and hashed archive attestor')
    receipt = _read_json(_file(receipts[0], base, 'Python runtime source receipt'))
    module = _load_python_attestor(attestors[0], base)
    if (receipt.get('schema_version') != 1 or receipt.get('source_build') != doc['python'].get('source_build')
            or receipt.get('url') != module.URL
            or receipt.get('identity') != {'version': '3.14.7', 'openssl_version': '3.5.9'}):
        raise ProfileError('Python runtime source identity differs')
    if _file(receipt.get('python'), base, 'Python runtime receipt executable') != python:
        raise ProfileError('Python runtime executable differs from source receipt')
    archive = _file(receipt.get('archive'), base, 'Python runtime archive')
    try:
        module.verify_archive(archive)
        matched = module.verify_prefix(archive, python.parent.parent)
    except (ValueError, OSError) as exc:
        raise ProfileError(f'Python runtime archive/prefix verification failed: {exc}') from exc
    if matched != receipt.get('verified_archive_files'):
        raise ProfileError('Python runtime archive file inventory differs from source receipt')


def _python(doc: dict[str, Any], base: Path) -> Path:
    python = _file(doc['python'], base, 'python')
    if doc['python'].get('version') != '3.14.7' or doc['python'].get('openssl_version') != '3.5.9':
        raise ProfileError('approved Python identity must be 3.14.7 / OpenSSL 3.5.9')
    verify_python_prefix(doc, base, python)
    output = _run([str(python), '-I', '-S', '-B', '-c', 'import json,ssl,platform; print(json.dumps({"version":platform.python_version(),"openssl_version":ssl.OPENSSL_VERSION.split()[1]}))'])
    try:
        observed = json.loads(output)
    except ValueError as exc:
        raise ProfileError('Python identity probe did not return JSON') from exc
    if not isinstance(observed, dict) or any(observed.get(key) != doc['python'][key] for key in ('version', 'openssl_version')):
        raise ProfileError(f'actual Python/OpenSSL identity differs: {observed}')
    return python


def _write(path: Path, value: Any) -> None:
    with path.open('x') as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write('\n')


def _new_output(path: Path, generation_root: str | None = None) -> None:
    if path.exists() or path.is_symlink():
        raise ProfileError(f'generation/output already exists: {path}')
    resolved = path.resolve()
    if generation_root is not None:
        if not isinstance(generation_root, str):
            raise ProfileError('profile generation_root must be an absolute directory')
        root = Path(generation_root)
        if (not root.is_absolute() or root.name != 'generations'
                or root.parent.name != '.native-runtime' or root.resolve() != root):
            raise ProfileError('profile generation_root must be an unaliased .native-runtime/generations directory')
        if resolved == root or not resolved.is_relative_to(root) or path.absolute() != resolved:
            raise ProfileError('generation must be below the profile generation_root without symlinks')
    else:
        repository_root = Path(__file__).resolve().parents[1]
        if not (resolved.is_relative_to(Path('/tmp')) or resolved.is_relative_to(repository_root / '.native-runtime')):
            raise ProfileError('outputs must be under /tmp or repository .native-runtime')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(mode=0o700)


def build(profile: str | Path, output: str | Path) -> dict[str, Any]:
    doc, path = load_profile(profile)
    if any(item.get('name') == 'pyarrow' and str(item.get('version', '')).startswith('26.')
           for item in doc['artifacts']):
        raise ProfileError('Arrow 26 profile rebuild is unsupported; use the reviewed isolated recipe')
    for item in doc['artifacts']:
        if item.get('name') == 'pyarrow' and str(item.get('version', '')).startswith('25.'):
            source = _read_json(_file(item.get('source_receipt'), path.parent, 'Arrow 25 source receipt'))
            if ('arrow25_network_build_receipt' in source or 'arrow25_cohort_build_receipt' in source
                    or source.get('source_policy') in ARROW_PRIVATE_FAMILIES):
                raise ProfileError('Arrow 25 private-network profile rebuild is unsupported; use the reviewed isolated recipe')
    _inputs(doc, path.parent)
    if not doc['recipes']:
        raise ProfileError('profile has no explicit build recipes')
    python = _python(doc, path.parent)
    scripts = []
    for recipe in doc['recipes']:
        if not isinstance(recipe, dict) or not isinstance(recipe.get('args', []), list):
            raise ProfileError('build recipe requires script and args array')
        timeout = recipe.get('timeout_seconds', 14400)
        if not isinstance(timeout, int) or not 1 <= timeout <= 43200:
            raise ProfileError('recipe timeout_seconds must be 1..43200')
        scripts.append((_file(recipe.get('script'), path.parent, 'recipe script'), recipe.get('args', []), timeout))
    output = Path(output).absolute()
    _new_output(output)
    try:
        for index, (script, args, timeout) in enumerate(scripts):
            if any(not isinstance(arg, str) for arg in args):
                raise ProfileError('recipe arguments must be strings')
            _run([str(python), '-I', str(script), *[arg.replace('{output}', str(output)).replace('{profile}', str(path)) for arg in args]], cwd=output, timeout=timeout, log=output / f'recipe-{index}.log')
        files = [{'path': str(file.relative_to(output)), 'sha256': sha256(file)} for file in sorted(output.rglob('*')) if file.is_file() and not file.is_symlink()]
        receipt = {'schema_version': 1, 'status': 'candidate', 'ready': False, 'profile_sha256': sha256(path), 'outputs': files}
        _write(output / 'build-receipt.json', receipt)
        return receipt
    except Exception:
        _write(output / 'FAILED.json', {'status': 'failed', 'ready': False, 'phase': 'build'})
        raise


def _inventory(root: Path) -> dict[str, dict[str, str]]:
    result = {}
    for file in sorted(root.rglob('*')):
        if file.is_symlink():
            result[str(file.relative_to(root))] = {'symlink': os.readlink(file), 'target_sha256': sha256(file.resolve()) if file.resolve().is_file() else ''}
        elif file.is_file():
            result[str(file.relative_to(root))] = {'sha256': sha256(file)}
    return result


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob('*'), reverse=True):
        if not path.is_symlink():
            mode = path.stat().st_mode
            path.chmod(0o555 if path.is_dir() or mode & 0o111 else 0o444)
    root.chmod(0o555)


def _checks(doc: dict[str, Any], profile: Path, generation: Path, *, installation: bool) -> list[str]:
    required = INSTALL_CHECKS if installation else RUNTIME_CHECKS
    applicable = [check for check in doc['checks'] if installation or check.get('name') in RUNTIME_CHECKS]
    names = [check.get('name') for check in applicable]
    if not required <= set(names) or len(names) != len(set(names)):
        raise ProfileError(f'missing/duplicate required checks: {sorted(required - set(names))}')
    for check in applicable:
        script = _file(check.get('script'), profile.parent, f'check {check["name"]}')
        args = check.get('args', [])
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            raise ProfileError('check args must be a string array')
        output = _run([str(generation / 'env/bin/python'), '-I', '-B', str(script), *[arg.replace('{generation}', str(generation)).replace('{profile}', str(profile)) for arg in args]])
        try:
            receipt = json.loads(output)
        except ValueError as exc:
            raise ProfileError(f'check {check["name"]} returned invalid JSON') from exc
        if not isinstance(receipt, dict) or receipt.get('status') != 'passed' or receipt.get('profile_sha256') != sha256(profile):
            raise ProfileError(f'check {check["name"]} did not attest this profile')
        if check['name'] == 'extension_policy' and receipt.get('policy_id') != doc['extension_policy']['policy_id']:
            raise ProfileError('compiled extension policy_id mismatch')
    return names


def _check_inputs(doc: dict[str, Any], base: Path) -> None:
    if any(not isinstance(check, dict) or not isinstance(check.get('name'), str) for check in doc['checks']):
        raise ProfileError('profile checks require named objects')
    names = [check['name'] for check in doc['checks']]
    if not INSTALL_CHECKS <= set(names) or len(names) != len(set(names)):
        raise ProfileError('missing/duplicate mandatory checks')
    for check in doc['checks']:
        _file(check.get('script'), base, f'check {check["name"]}')
    if not doc['native_files']:
        raise ProfileError('profile native_files must contain approved libpq/SSL/crypto mappings')
    owners = set()
    for native in doc['native_files']:
        if not isinstance(native, dict) or not isinstance(native.get('owner'), str):
            raise ProfileError('native file requires locked wheel or dpkg owner')
        owners.add(native['owner'])
        _hash(native.get('sha256'), 'native file')
        if not isinstance(native.get('path'), str):
            raise ProfileError('native file requires path')
    if not {'dpkg:libpq5', 'dpkg:libssl3t64'} <= owners:
        raise ProfileError('native_files requires dpkg:libpq5 and dpkg:libssl3t64')


def _installed_wheels(doc: dict[str, Any], profile: Path, generation: Path) -> None:
    site = generation / 'env/lib/python3.14/site-packages'
    if not site.is_dir():
        raise ProfileError('generation lacks CPython 3.14 site-packages')
    if list(site.glob('psycopg_binary*')) or list(site.glob('psycopg-binary*')):
        raise ProfileError('psycopg-binary is prohibited')
    for artifact in doc['artifacts']:
        wheel = _wheel(artifact, profile.parent, doc['project'])
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                if member.is_dir() or member.filename.endswith('/RECORD'):
                    continue
                installed = site / member.filename
                if not installed.is_file() or sha256(installed) != hashlib.sha256(archive.read(member)).hexdigest():
                    raise ProfileError(f'installed wheel bytes differ: {member.filename}')


def verify_native_file(native: dict[str, Any], profile: str | Path, generation: str | Path) -> None:
    """Verify a declared native file's bytes and real package ownership."""
    profile = Path(profile).resolve()
    generation = Path(generation).resolve()
    item = dict(native)
    if not isinstance(item.get('path'), str) or not isinstance(item.get('owner'), str):
        raise ProfileError('native file requires path and owner')
    item['path'] = item['path'].replace('{generation}', str(generation))
    file = _file(item, profile.parent, 'native file')
    owner = item['owner']
    if owner.startswith('dpkg:'):
        package = owner.removeprefix('dpkg:')
        if not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', package):
            raise ProfileError('invalid native dpkg owner')
        output = _run(['/usr/bin/dpkg-query', '-S', str(file)])
        owners = {line.split(': ', 1)[0].split(':', 1)[0] for line in output.splitlines() if ': ' in line}
        if owners != {package}:
            raise ProfileError(f'native file dpkg owner mismatch: {file}')
    elif owner.startswith('python-runtime:'):
        doc = _read_json(profile)
        python = _file(doc.get('python'), profile.parent, 'python runtime')
        source_build = doc['python'].get('source_build')
        if not source_build or owner != f'python-runtime:{source_build}' or not file.is_relative_to(python.parent.parent):
            raise ProfileError(f'native file Python runtime owner/prefix mismatch: {file}')
    elif owner.startswith('wheel:'):
        package = owner.removeprefix('wheel:')
        if not re.fullmatch(r'[a-z0-9][a-z0-9_.-]*', package):
            raise ProfileError('invalid native wheel owner')
        code = 'import importlib.metadata as m,pathlib,sys; d=m.distribution(sys.argv[1]); p=pathlib.Path(sys.argv[2]).resolve(); sys.exit(0 if any(pathlib.Path(d.locate_file(f)).resolve()==p for f in (d.files or [])) else 1)'
        try:
            _run([str(generation / 'env/bin/python'), '-I', '-B', '-c', code, package, str(file)])
        except ProfileError as exc:
            raise ProfileError(f'native file wheel owner mismatch: {file}') from exc
    else:
        raise ProfileError('native file owner must use dpkg:, wheel:, or python-runtime: namespace')


def install(profile: str | Path, generation: str | Path) -> dict[str, Any]:
    generation = Path(generation).absolute()
    if generation.exists() or generation.is_symlink():
        raise ProfileError(f'generation already exists: {generation}')
    verify_artifacts(profile)
    doc, path = load_profile(profile)
    _check_inputs(doc, path.parent)
    host = _host(doc)
    python = _python(doc, path.parent)
    uv = _file(doc['uv'], path.parent, 'uv')
    _new_output(generation, doc.get('generation_root'))
    try:
        project = generation / 'project'
        project.mkdir()
        shutil.copyfile(_file(doc['project'], path.parent, 'project'), project / 'pyproject.toml')
        shutil.copyfile(_file(doc['base_lock'], path.parent, 'base_lock'), project / 'uv.lock')
        environment = launch_environment()
        environment['UV_PROJECT_ENVIRONMENT'] = str(generation / 'env')
        environment['UV_CACHE_DIR'] = str(generation / 'cache')
        _run([str(uv), 'venv', '--python', str(python), str(generation / 'env')], env=environment)
        _run([str(uv), 'sync', '--locked', '--no-editable', '--no-install-project', '--extra', 'postgres', '--extra', 'geo', '--extra', 'ml', '--no-install-package', 'psycopg-c', '--no-install-package', 'pyarrow', '--no-install-package', 'duckdb', '--no-install-package', 'lxml', '--project', str(project)], env=environment, timeout=3600)
        requirements = generation / 'native-wheels.txt'
        requirements.write_text('\n'.join(f'{_file(artifact, path.parent, "wheel").as_uri()} --hash=sha256:{artifact["sha256"]}' for artifact in doc['artifacts']) + '\n')
        _run([str(uv), 'pip', 'install', '--python', str(generation / 'env/bin/python'), '--no-index', '--no-deps', '--require-hashes', '-r', str(requirements)], env=environment)
        _installed_wheels(doc, path, generation)
        checks = _checks(doc, path, generation, installation=True)
        # This cache belongs to this new generation; installed files remain.
        if (generation / 'cache').is_dir() and not (generation / 'cache').is_symlink():
            shutil.rmtree(generation / 'cache')
        receipt = {'schema_version': 1, 'status': 'installed', 'ready': False, 'profile_sha256': sha256(path), 'host_packages': host, 'checks': checks, 'files': _inventory(generation)}
        _write(generation / 'generation.json', receipt)
        _freeze(generation)
        return verify(path, generation)
    except Exception:
        if os.access(generation, os.W_OK) and not (generation / 'FAILED.json').exists():
            _write(generation / 'FAILED.json', {'status': 'failed', 'ready': False, 'phase': 'install'})
        raise


def verify(profile: str | Path, generation: str | Path | None = None) -> dict[str, Any]:
    verify_artifacts(profile)
    if generation is None:
        raise ProfileError('generation is required for readiness; artifacts alone are candidate evidence')
    doc, path = load_profile(profile)
    _check_inputs(doc, path.parent)
    generation = Path(generation).resolve()
    receipt = _read_json(generation / 'generation.json')
    if receipt.get('status') != 'installed' or receipt.get('profile_sha256') != sha256(path):
        raise ProfileError('generation receipt/profile mismatch')
    if not INSTALL_CHECKS <= set(receipt.get('checks', [])):
        raise ProfileError('generation lacks mandatory installation checks')
    inventory = _inventory(generation)
    inventory.pop('generation.json', None)
    if inventory != receipt.get('files'):
        raise ProfileError('generation inventory changed')
    for file in [generation, *generation.rglob('*')]:
        if not file.is_symlink() and (file.stat().st_uid != os.getuid() or file.stat().st_mode & 0o222):
            raise ProfileError(f'generation is not owned and read-only: {file}')
    _host(doc)
    _python(doc, path.parent)
    for native in doc['native_files']:
        verify_native_file(native, path, generation)
    _installed_wheels(doc, path, generation)
    checks = _checks(doc, path, generation, installation=False)
    return {'status': 'verified', 'ready': True, 'profile_sha256': sha256(path), 'generation': str(generation), 'checks': checks, 'live_services': 'not_attested'}


def execute(profile: str | Path, generation: str | Path, args: list[str]) -> None:
    environment = launch_environment()
    verify(profile, generation)
    if not args:
        raise ProfileError('exec requires Python arguments, for example -m biominer.cli')
    python = str(Path(generation).resolve() / 'env/bin/python')
    os.execve(python, [python, '-I', '-B', '-m', 'biominer.polars_runtime', '--', *args], environment)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('build', 'install', 'verify', 'exec', 'artifacts'):
        child = commands.add_parser(command, allow_abbrev=False)
        child.add_argument('--profile', required=True, type=Path)
        child.add_argument('--generation', type=Path)
        child.add_argument('--output', type=Path)
        if command == 'exec':
            child.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            if args.output is None:
                raise ProfileError('build requires --output')
            result = build(args.profile, args.output)
        elif args.command == 'artifacts':
            result = verify_artifacts(args.profile)
        elif args.command == 'verify':
            result = verify(args.profile, args.generation)
        else:
            if args.generation is None:
                raise ProfileError(f'{args.command} requires --generation')
            if args.command == 'exec':
                execute(args.profile, args.generation, args.args[1:] if args.args[:1] == ['--'] else args.args)
                return 0
            result = install(args.profile, args.generation)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ProfileError, OSError) as exc:
        print(json.dumps({'status': 'blocked', 'ready': False, 'reason': str(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
