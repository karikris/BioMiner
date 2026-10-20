"""Native admission tests compile actual ELF fixtures, independent of wheel build."""
from pathlib import Path
import importlib.util
import json
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def builder():
    path = ROOT / 'scripts/build_native_duckdb.py'
    assert path.is_file(), 'native DuckDB builder is missing'
    spec = importlib.util.spec_from_file_location('native_duckdb_builder', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_rejects_relative_dependency_and_duplicate_names(tmp_path):
    module = builder()
    for document in (
        {'extensions': [{'name':'inet', 'sha256':'0'*64, 'dependencies':[{'soname':'libx.so','path':'relative.so','sha256':'1'*64}]}]},
        {'extensions': [{'name':'inet','sha256':'0'*64,'dependencies':[]}] * 2},
        {'extensions': [{'name':'spatial','sha256':'0'*64,'dependencies':[]}]},
    ):
        path = tmp_path / 'policy.json'
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            module.load_policy(path)


def test_source_pins_are_complete():
    module = builder()
    assert set(module.SOURCES) == {'core', 'python', 'spatial', 'vcpkg'}
    for value in module.SOURCES.values():
        assert len(value['commit']) == 40
        assert len(value['sha256']) == 64


def compile_fixture(tmp_path, allowed, dependency=False, rpath=False):
    module = builder()
    source = tmp_path / 'extension.cpp'
    source.write_text('extern "C" int answer(){return 42;}\n')
    extension = tmp_path / 'inet.duckdb_extension'
    command=['g++','-shared','-fPIC',str(source),'-o',str(extension)]
    if dependency:
        command+=['-Wl,--no-as-needed','-lm']
    if rpath:
        command+=['-Wl,-rpath,/unapproved/path']
    subprocess.run(command, check=True)
    sha = module.sha256(extension)
    policy = {'extensions':[{'name':'inet','sha256':sha,'dependencies':[]}]} if allowed else {'extensions':[]}
    module.write_policy_header(policy, tmp_path / 'biominer_extension_policy.hpp')
    harness = tmp_path / 'harness.cpp'
    harness.write_text(r'''
#include "biominer_extension_admission.hpp"
#include <openssl/sha.h>
#include <iostream>
static std::string hash_fd(int fd) {
 SHA256_CTX ctx; SHA256_Init(&ctx); char buffer[65536]; ssize_t n;
 if(lseek(fd,0,SEEK_SET)<0)throw std::runtime_error("seek");
 while((n=read(fd,buffer,sizeof(buffer)))>0) SHA256_Update(&ctx,buffer,n);
 if(n<0)throw std::runtime_error("read"); unsigned char raw[32]; SHA256_Final(raw,&ctx);
 const char *digits="0123456789abcdef";std::string result;
 for(auto c:raw){result+=digits[c>>4];result+=digits[c&15];}return result;
}
int main(int argc,char **argv) {
 try {
  auto snapshot=biominer_native::Admit(argv[1],"inet",hash_fd);
  // Replace the original AFTER admission. Loading must retain sealed original bytes.
  if(argc>2){int output=open(argv[1],O_WRONLY|O_TRUNC);write(output,"bad",3);close(output);}
  void *handle=dlopen(snapshot.path().c_str(),RTLD_NOW|RTLD_LOCAL);
  if(!handle)throw std::runtime_error(dlerror());
  auto answer=(int(*)())dlsym(handle,"answer");
  if(!answer||answer()!=42)throw std::runtime_error("wrong extension");
  if(fcntl(snapshot.fd,F_GET_SEALS)!=(F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL))throw std::runtime_error("not sealed");
  dlclose(handle);std::cout<<"admitted";return 0;
 }catch(const std::exception &ex){std::cerr<<ex.what();return 1;}
}
''')
    executable=tmp_path/'harness'
    subprocess.run(['g++','-std=c++11','-I',str(tmp_path),'-I',str(ROOT/'patches/native-duckdb'),str(harness),'-ldl','-lcrypto','-o',str(executable)],check=True,capture_output=True)
    return executable, extension


def test_unknown_extension_fails_before_loading(tmp_path):
    executable, extension = compile_fixture(tmp_path, False)
    result=subprocess.run([str(executable),str(extension)], capture_output=True,text=True)
    assert result.returncode == 1
    assert 'not in compiled allowlist' in result.stderr


def test_sealed_snapshot_survives_path_replacement(tmp_path):
    executable, extension = compile_fixture(tmp_path, True)
    result=subprocess.run([str(executable),str(extension),'replace'],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert extension.read_bytes() == b'bad'


def test_hash_mismatch_fails_before_loading(tmp_path):
    executable, extension = compile_fixture(tmp_path, True)
    with extension.open('ab') as handle:
        handle.write(b'tampering')
    result=subprocess.run([str(executable),str(extension)],capture_output=True,text=True)
    assert result.returncode == 1
    assert 'SHA256 mismatch' in result.stderr


def test_unapproved_native_dependency_is_rejected(tmp_path):
    executable, extension = compile_fixture(tmp_path, True, dependency=True)
    result=subprocess.run([str(executable),str(extension)],capture_output=True,text=True)
    assert result.returncode == 1
    assert 'dependency is unapproved' in result.stderr


def test_rpath_discovery_is_rejected(tmp_path):
    executable, extension = compile_fixture(tmp_path, True, rpath=True)
    result=subprocess.run([str(executable),str(extension)],capture_output=True,text=True)
    assert result.returncode == 1
    assert 'RPATH/RUNPATH is unsupported' in result.stderr
