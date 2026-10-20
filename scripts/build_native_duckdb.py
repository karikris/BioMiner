#!/usr/bin/env python3
"""Build the pinned BioMiner DuckDB/Spatial wheel; never accepts stock substitution.

Usage: build_native_duckdb.py --python /absolute/venv/bin/python [--prepare-only]
The Linux x86_64 build is rootless; default four jobs share an eight-job budget. Native
artifacts stay outside Git. A failed receipt is deliberately not promotable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / 'patches/native-duckdb'
SOURCES = {
    'core': {'repo':'duckdb/duckdb','commit':'069cc9f9b5be802405797faecc284961b07c70ef','sha256':'77cfc249ad652104b770fa7a8bb1d0119c3232e5dd03aadaaa1c9c7554fe7c05'},
    'python': {'repo':'duckdb/duckdb-python','commit':'802a345bb627435d2a05d63cb74454fa98b60b5e','sha256':'7fcdd8b2eb63182f222514ce27c9a39af8694f0c5b1debad7e37cc93775cc3f9'},
    'spatial': {'repo':'duckdb/duckdb-spatial','commit':'04270fe7bb4eb412765cce8afde0276a9d81d3ba','sha256':'ad42d377bc59afd8f6dab903828f13254de0320efb5dbf52be7ba9ec38f79341'},
    'vcpkg': {'repo':'microsoft/vcpkg','commit':'84bab45d415d22042bd0b9081aea57f362da3f35','sha256':'e8731530753279ac15e760c010b1724439afdf057c4fb6cb2ffc04f7680a4953'},
}
OPENSSL_SHA256='f3091b7ccb41a11d877b7406b7afc9dd9c5003d299c6bb33cff882127bc06c91'


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path | None) -> dict:
    policy=json.loads(path.read_text()) if path else {'extensions':[]}
    if set(policy)!={'extensions'} or not isinstance(policy['extensions'],list):
        raise ValueError('policy must contain only an extensions list')
    names=set()
    for entry in policy['extensions']:
        if set(entry)!={'name','sha256','dependencies'} or not re.fullmatch(r'[a-z][a-z0-9_]*',entry['name']):
            raise ValueError('invalid extension policy entry')
        if entry['name'] in names:raise ValueError('duplicate extension name')
        names.add(entry['name'])
        if entry['name'] in {'core_functions','json','parquet','icu','spatial'}:
            raise ValueError('built-in extension names cannot be externally allowed')
        if not re.fullmatch('[0-9a-f]{64}',entry['sha256']):raise ValueError('invalid extension hash')
        if not isinstance(entry['dependencies'],list):raise ValueError('dependencies must be a list')
        sonames=set()
        for dep in entry['dependencies']:
            if set(dep)!={'soname','path','sha256'} or not Path(dep['path']).is_absolute():
                raise ValueError('dependencies require absolute approved paths')
            if not re.fullmatch(r'[A-Za-z0-9_.+-]+',dep['soname']) or dep['soname'] in sonames:
                raise ValueError('invalid or duplicate dependency soname')
            sonames.add(dep['soname'])
            if not re.fullmatch('[0-9a-f]{64}',dep['sha256']):raise ValueError('invalid dependency hash')
    return policy


def write_policy_header(policy: dict, path: Path, policy_id: str | None = None) -> str:
    canonical=json.dumps(policy,sort_keys=True,separators=(',',':')).encode()
    policy_id=policy_id or hashlib.sha256(canonical).hexdigest()
    quote=json.dumps
    entries=[]
    for entry in policy['extensions']:
        deps=','.join('{'+','.join(quote(dep[k]) for k in ('soname','path','sha256'))+'}' for dep in entry['dependencies'])
        entries.append('{'+quote(entry['name'])+','+quote(entry['sha256'])+', {'+deps+'}}')
    path.write_text('''// Generated from a reviewed policy; do not edit at runtime.
#ifndef BIOMINER_EXTENSION_POLICY_HPP
#define BIOMINER_EXTENSION_POLICY_HPP
#include <string>
#include <vector>
namespace biominer_native {
struct Dependency {std::string soname,path,sha256;};
struct Extension {std::string name,sha256;std::vector<Dependency> dependencies;};
inline const std::vector<Extension> &Policy(){static const std::vector<Extension> value={'''+','.join(entries)+'''};return value;}
inline const char *PolicyId(){return "'''+policy_id+'''";}
}
#endif
''')
    return policy_id


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before)!=1:raise RuntimeError('upstream patch anchor changed: '+before[:80])
    return text.replace(before,after,1)


def patch_core(core: Path, policy: dict, policy_id: str) -> dict:
    target=core/'src/main/extension'
    shutil.copy2(PATCHES/'biominer_extension_admission.hpp',target)
    write_policy_header(policy,target/'biominer_extension_policy.hpp',policy_id)
    source=target/'extension_load.cpp'
    text=source.read_text()
    text=replace_once(text,'#include "mbedtls_wrapper.hpp"','#include "mbedtls_wrapper.hpp"\n#include "biominer_extension_admission.hpp"\n\nextern "C" __attribute__((visibility("default"))) const char *biominer_extension_policy_id() {\n    return biominer_native::PolicyId();\n}\n')
    text=replace_once(text,'\tauto handle = fs.OpenFile(filename, FileFlags::FILE_FLAGS_READ);', '''
    if (Settings::Get<AllowUnsignedExtensionsSetting>(db) ||
        Settings::Get<AllowExtensionsMetadataMismatchSetting>(db)) {
        throw PermissionException("BioMiner extension signature/ABI bypass is prohibited");
    }
    auto original_filebase = fs.ExtractBaseName(filename);
    auto hash_fd = [&](int fd) {
        duckdb_mbedtls::MbedTlsWrapper::SHA256State state;
        if (lseek(fd, 0, SEEK_SET) < 0) throw IOException("snapshot seek failed");
        char buffer[65536]; ssize_t count;
        while ((count = ::read(fd, buffer, sizeof(buffer))) > 0) {
            state.AddString(std::string(buffer, static_cast<size_t>(count)));
        }
        if (count < 0) throw IOException("snapshot hash read failed");
        auto raw = state.Finalize(); std::string hex;
        const char *digits = "0123456789abcdef";
        for (unsigned char byte : raw) { hex += digits[byte >> 4]; hex += digits[byte & 15]; }
        return hex;
    };
    // Metadata, signature verification and dlopen all consume this same sealed snapshot.
    auto snapshot = [&]() {
        try { return biominer_native::Admit(filename, StringUtil::Lower(original_filebase), hash_fd); }
        catch (const std::exception &ex) { throw PermissionException("BioMiner native admission failed: %s", ex.what()); }
    }();
    auto sealed_filename = snapshot.path();
    auto handle = fs.OpenFile(sealed_filename, FileFlags::FILE_FLAGS_READ);''')
    text=replace_once(text,'\tauto filebase = fs.ExtractBaseName(filename);','\tauto filebase = original_filebase;')
    text=replace_once(text,'\tauto dopen_from = filename;','\tauto dopen_from = sealed_filename;')
    text=replace_once(text,'void ExtensionHelper::LoadExternalExtension(DatabaseInstance &db, FileSystem &fs, const string &extension) {',
        'void ExtensionHelper::LoadExternalExtension(DatabaseInstance &db, FileSystem &fs, const string &extension) {\n'
        '    if (IsFullPath(extension)) {\n'
        '        bool allowed = false;\n'
        '        for (const auto &entry : biominer_native::Policy()) {\n'
        '            if (entry.name == GetExtensionName(extension)) allowed = true;\n'
        '        }\n'
        '        if (!allowed) throw PermissionException("BioMiner external path is not in compiled allowlist");\n'
        '    }')
    source.write_text(text)
    settings=core/'src/main/settings/custom_settings.cpp'
    text=settings.read_text()
    text=replace_once(text,'void AllowUnsignedExtensionsSetting::OnSet(SettingCallbackInfo &info, Value &input) {\n\tif (info.db && input.GetValue<bool>()) {','void AllowUnsignedExtensionsSetting::OnSet(SettingCallbackInfo &info, Value &input) {\n\tif (input.GetValue<bool>()) {')
    text+='\nnamespace duckdb {\nvoid AllowExtensionsMetadataMismatchSetting::OnSet(SettingCallbackInfo &, Value &input) {\n    if (input.GetValue<bool>()) throw InvalidInputException(\"BioMiner metadata bypass is prohibited\");\n}\n}\n'
    settings.write_text(text)
    config_source=core/'src/main/config.cpp'
    text=config_source.read_text()
    text=replace_once(text,'DUCKDB_SETTING(AllowExtensionsMetadataMismatchSetting)',
                      'DUCKDB_SETTING_CALLBACK(AllowExtensionsMetadataMismatchSetting)')
    config_source.write_text(text)
    settings_header=core/'src/include/duckdb/main/settings.hpp'
    text=settings_header.read_text()
    text=replace_once(text,'static constexpr idx_t SettingIndex = 2;\n};','static constexpr idx_t SettingIndex = 2;\n\tstatic void OnSet(SettingCallbackInfo &info, Value &input);\n};')
    settings_header.write_text(text)
    cmake=core/'CMakeLists.txt'
    cmake.write_text(cmake.read_text()+'\n# BioMiner: distinct ELF linker maps, including the parent Python module.\nset(CMAKE_SHARED_LINKER_FLAGS \"-Wl,-Map,%\")\nset(CMAKE_MODULE_LINKER_FLAGS \"-Wl,-Map,%\")\nset(CMAKE_MODULE_LINKER_FLAGS \"-Wl,-Map,%\" PARENT_SCOPE)\n')
    # BUILD_EXTENSIONS reads this early config before DUCKDB_EXTENSION_CONFIGS.
    # Register local Spatial here so upstream DONT_LINK cannot win first.
    spatial_config=core/'.github/config/extensions/spatial.cmake'
    spatial=core.parent/('duckdb-spatial-'+SOURCES['spatial']['commit'])
    spatial_config.write_text('duckdb_extension_load(spatial SOURCE_DIR "'+str(spatial)+'" INCLUDE_DIR "'+str(spatial/'src/spatial')+'" EXTENSION_VERSION "'+SOURCES['spatial']['commit']+'")\n')
    return {str(p.relative_to(core)):sha256(p) for p in (source,settings,settings_header,cmake,spatial_config,target/'biominer_extension_admission.hpp',target/'biominer_extension_policy.hpp')}


def fetch_sources(work: Path) -> dict[str,Path]:
    destination=work/'sources';destination.mkdir(parents=True,exist_ok=True);result={}
    for key,spec in SOURCES.items():
        archive=destination/(key+'.tar.gz')
        if not archive.exists():urllib.request.urlretrieve(f"https://codeload.github.com/{spec['repo']}/tar.gz/{spec['commit']}",archive)
        if sha256(archive)!=spec['sha256']:raise RuntimeError('source checksum mismatch: '+key)
        directory=destination/(spec['repo'].split('/')[-1]+'-'+spec['commit'])
        if not directory.exists():
            with tarfile.open(archive) as handle:handle.extractall(destination,filter='data')
        result[key]=directory
    return result


def run(command: list[str], cwd: Path, env: dict, logfile: Path) -> None:
    with logfile.open('ab') as handle:
        handle.write(('COMMAND '+json.dumps(command)+'\n').encode());handle.flush()
        process=subprocess.run(command,cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError(f'command exited {process.returncode}; see {logfile}')


def prepare_overlay(work: Path, sources: dict[str,Path]) -> Path:
    port=work/'overlay/openssl';port.parent.mkdir(parents=True,exist_ok=True)
    if port.exists():shutil.rmtree(port)
    shutil.copytree(sources['vcpkg']/'ports/openssl',port)
    archive=work/'sources/openssl-3.6.5.tar.gz'
    if not archive.exists():urllib.request.urlretrieve('https://codeload.github.com/openssl/openssl/tar.gz/refs/tags/openssl-3.6.5',archive)
    if sha256(archive)!=OPENSSL_SHA256:raise RuntimeError('OpenSSL source checksum mismatch')
    digest=hashlib.sha512(archive.read_bytes()).hexdigest()
    recipe=(port/'portfile.cmake').read_text()
    recipe=re.sub(r'(SHA512 )([0-9a-f]{128})',lambda match:match[1]+digest,recipe,count=1)
    # This upstream fix is already present in 3.6.5. Retain the packaging patches;
    # vcpkg will fail closed if any of them no longer apply.
    recipe=re.sub(r'        aes_cfb128_vaes_encdec_wrapper.diff[^\n]*\n','',recipe)
    (port/'portfile.cmake').write_text(recipe)
    metadata=json.loads((port/'vcpkg.json').read_text());metadata['version']='3.6.5';metadata.pop('port-version',None)
    (port/'vcpkg.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return port.parent


def make_source_receipt(work: Path, sources: dict[str,Path], wheel: Path, python: Path) -> dict:
    """Bind actual wheel, link maps, native archives and compiler/source inputs."""
    maps=list((sources['python']/'build').rglob('_duckdb*.so.map'))
    if len(maps)!=1:raise RuntimeError('expected one Python module linker map')
    archives=set()
    text=maps[0].read_text(errors='strict')
    for match in re.finditer(r'(?m)^\s*(?:LOAD\s+)?([^\s()]+\.a)(?:\(|\s|$)',text):
        path=Path(match[1])
        archives.add(path if path.is_absolute() else maps[0].parent/path)
    # A map is a link receipt, not a search over every archive in the wheelhouse.
    names={path.name for path in archives}
    if not {'libcrypto.a','libssl.a'} <= names:raise RuntimeError('linker map does not bind both OpenSSL archives')
    crypto=next(path for path in archives if path.name=='libcrypto.a')
    versions=set(re.findall(rb'OpenSSL (3\.[0-9]+\.[0-9]+)(?: |\x00)',crypto.read_bytes()))
    if versions!={b'3.6.5'}:raise RuntimeError('linked crypto archive does not match approved OpenSSL source version')
    def item(path: Path) -> dict:
        path=path.absolute();return {'path':str(path),'sha256':sha256(path)}
    inputs=[item(work/'sources'/f'{key}.tar.gz') for key in sorted(SOURCES)]
    inputs += [item(work/'sources/openssl-3.6.5.tar.gz'),item(sources['python']/'pyproject.toml'),item(work/'extension_config.cmake'),item(work/'triplets/x64-linux-biominer.cmake')]
    inputs += [item(path) for path in sorted((work/'overlay').rglob('*')) if path.is_file()]
    inputs += [item(sources['core']/path) for path in ('src/main/extension/extension_load.cpp','src/main/extension/biominer_extension_admission.hpp','src/main/extension/biominer_extension_policy.hpp','src/main/settings/custom_settings.cpp','src/main/config.cpp','src/include/duckdb/main/settings.hpp','CMakeLists.txt','.github/config/extensions/spatial.cmake')]
    inputs.sort(key=lambda value:value['path'])
    toolchain={}
    for name in ('g++','gcc','cmake','ninja','uv'):
        executable=shutil.which(name)
        if not executable:raise RuntimeError('toolchain executable unavailable: '+name)
        toolchain[name]=item(Path(executable).resolve())
    toolchain['python']=item(python)
    toolchain['vcpkg']=item(sources['vcpkg']/'vcpkg')
    native=[];seen=set();audit=work/'native-audit';audit.mkdir(exist_ok=True)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.endswith('.so') and '.so.' not in member:continue
            payload=archive.read(member)
            if not payload.startswith(b'\x7fELF'):continue
            digest=hashlib.sha256(payload).hexdigest()
            native.append({'path':'{generation}/env/lib/python3.14/site-packages/'+member,'sha256':digest,'owner':'wheel:duckdb'})
            filename=audit/Path(member).name;filename.write_bytes(payload)
            output=subprocess.check_output(['ldd',str(filename)],text=True)
            if 'not found' in output:raise RuntimeError('unresolved native wheel dependency')
            for line in output.splitlines():
                match=re.search(r'(?:=>\s+)?(/\S+)\s+\(',line)
                if not match:continue
                library=Path(match[1]).resolve()
                if library in seen:continue
                seen.add(library)
                ownership=subprocess.check_output(['dpkg-query','-S',str(library)],text=True)
                owners={line.split(': ',1)[0].split(':',1)[0] for line in ownership.splitlines() if ': ' in line}
                if len(owners)!=1:raise RuntimeError('native dependency lacks one dpkg owner')
                native.append(dict(item(library),owner='dpkg:'+owners.pop()))
    if not native:raise RuntimeError('wheel contains no native ELF file')
    return {'wheel_sha256':sha256(wheel),'source_revision':SOURCES['core']['commit'],
            'sources':SOURCES,'toolchain':toolchain,'inputs':inputs,
            'inputs_sha256':hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'native_link_artifacts':[item(path) for path in sorted(archives)]+[item(maps[0])],
            'native_files':native,'effective_versions':{'openssl':'3.6.5'}}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,default=ROOT/'.native-runtime/duckdb')
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--jobs',type=int,default=4,choices=range(1,9))
    args=parser.parse_args();work=args.work_dir.absolute();work.mkdir(parents=True,exist_ok=True)
    receipt={'schema_version':1,'package':'duckdb','upstream_version':'1.5.6','status':'building','sources':SOURCES,'openssl_version':'3.6.5','platform':'linux_x86_64','python_abi':'cp314','compile_jobs':args.jobs,'overall_job_budget':8}
    receipt_path=work/'build-receipt.json';logfile=work/'build.log'
    try:
        if sys.platform!='linux' or os.uname().machine!='x86_64':raise RuntimeError('only Linux x86_64 is supported')
        policy=load_policy(args.policy);sources=fetch_sources(work)
        # Preserve mtimes for identical derived files: cached recompilation should
        # touch changed policy/loader inputs rather than every core translation unit.
        old={str(p.relative_to(sources['core'])):(sha256(p),p.stat().st_mtime_ns) for p in (sources['core']/'src/main/extension/extension_load.cpp',sources['core']/'src/main/extension/biominer_extension_admission.hpp',sources['core']/'src/main/extension/biominer_extension_policy.hpp',sources['core']/'src/main/settings/custom_settings.cpp',sources['core']/'src/main/config.cpp',sources['core']/'src/include/duckdb/main/settings.hpp',sources['core']/'CMakeLists.txt',sources['core']/'.github/config/extensions/spatial.cmake') if p.is_file()}
        # Re-extract pristine core for repeatable patching; no BioMiner Git mutations.
        with tarfile.open(work/'sources/core.tar.gz') as handle:handle.extractall(work/'sources',filter='data')
        inputs={'sources':SOURCES,'policy':policy,'patch_sha256':sha256(PATCHES/'biominer_extension_admission.hpp'),'builder_sha256':sha256(Path(__file__)),'openssl_sha256':OPENSSL_SHA256}
        policy_id=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        receipt.update(policy_id=policy_id,extension_policy=policy,patches=patch_core(sources['core'],policy,policy_id))
        for name,(digest,mtime) in old.items():
            path=sources['core']/name
            if path.is_file() and sha256(path)==digest:os.utime(path,ns=(path.stat().st_atime_ns,mtime))
        overlay=prepare_overlay(work,sources)
        receipt['overlay']={str(path.relative_to(work)):sha256(path) for path in sorted(overlay.rglob('*')) if path.is_file()}
        config=work/'extension_config.cmake';config.write_text(f'duckdb_extension_load(spatial SOURCE_DIR "{sources["spatial"]}" INCLUDE_DIR "{sources["spatial"]}/src/spatial")\n')
        triplets=work/'triplets';triplets.mkdir(exist_ok=True)
        (triplets/'x64-linux-biominer.cmake').write_text('set(VCPKG_TARGET_ARCHITECTURE x64)\nset(VCPKG_CRT_LINKAGE dynamic)\nset(VCPKG_LIBRARY_LINKAGE static)\nset(VCPKG_CMAKE_SYSTEM_NAME Linux)\nset(VCPKG_BUILD_TYPE release)\n')
        py=sources['python'];(py/'PKG-INFO').write_text('Metadata-Version: 2.1\nName: duckdb\nVersion: 1.5.6\n')
        (py/'duckdb_packaging/duckdb_version.txt').write_text('v1.5.6')
        version='1.5.6+biominer.'+policy_id[:12];receipt['version']=version
        # Upstream deliberately strips setuptools-scm pretend-version variables.
        # Use explicit project metadata while retaining the upstream core ABI version.
        with tarfile.open(work/'sources/python.tar.gz') as handle:handle.extractall(work/'sources',filter='data')
        (py/'PKG-INFO').write_text('Metadata-Version: 2.1\nName: duckdb\nVersion: '+version+'\n')
        (py/'duckdb_packaging/duckdb_version.txt').write_text('v1.5.6')
        metadata=(py/'pyproject.toml').read_text()
        metadata=replace_once(metadata,'dynamic = ["version"]','version = '+json.dumps(version))
        metadata=replace_once(metadata,'metadata.version.provider = "scikit_build_core.metadata.setuptools_scm"','')
        (py/'pyproject.toml').write_text(metadata)
        receipt['python_metadata_patch_sha256']=sha256(py/'pyproject.toml')
        (work/'registry-cache').mkdir(exist_ok=True)
        (work/'downloads').mkdir(exist_ok=True)
        env=dict(os.environ);env.update(CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),VCPKG_MAX_CONCURRENCY=str(args.jobs),XDG_CACHE_HOME=str(work/'cache'),VCPKG_REGISTRIES_CACHE=str(work/'registry-cache'),UV_CACHE_DIR=str(work/'uv-cache'),VCPKG_DOWNLOADS=str(work/'downloads'),VCPKG_BINARY_SOURCES='clear')
        flags={'DUCKDB_SOURCE_PATH':str(sources['core']),'DUCKDB_EXTENSION_CONFIGS':str(config),'BUILD_EXTENSIONS':'core_functions;parquet;icu;json;spatial','ENABLE_EXTENSION_AUTOINSTALL':'OFF','ENABLE_EXTENSION_AUTOLOADING':'OFF','SPATIAL_USE_NETWORK':'ON','SPATIAL_USE_GEOS':'ON','CMAKE_TOOLCHAIN_FILE':str(sources['vcpkg']/'scripts/buildsystems/vcpkg.cmake'),'VCPKG_MANIFEST_DIR':str(sources['spatial']),'VCPKG_TARGET_TRIPLET':'x64-linux-biominer','VCPKG_OVERLAY_TRIPLETS':str(triplets),'VCPKG_OVERLAY_PORTS':str(overlay),'VCPKG_INSTALLED_DIR':str(work/'vcpkg_installed'),'CMAKE_SHARED_LINKER_FLAGS':'-Wl,-Map,%','CMAKE_MODULE_LINKER_FLAGS':'-Wl,-Map,%'}
        receipt['cmake_flags']=flags
        receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        if args.prepare_only:receipt['status']='prepared';return 0
        if args.python.is_symlink():pass  # Preserve the venv entrypoint; never resolve it.
        # vcpkg needs Git objects for versioned registry ports; the archive pin
        # still verifies working files. This Git database is isolated build data.
        if not (sources['vcpkg']/'.git').exists():
            run(['git','init',str(sources['vcpkg'])],work,env,logfile)
            run(['git','-C',str(sources['vcpkg']),'fetch','--depth=1','https://github.com/microsoft/vcpkg.git',SOURCES['vcpkg']['commit']],work,env,logfile)
        run([str(sources['vcpkg']/'bootstrap-vcpkg.sh'),'-disableMetrics'],sources['vcpkg'],env,logfile)
        wheelhouse=work/'wheelhouse';wheelhouse.mkdir(exist_ok=True)
        command=['uv','build','--python',str(args.python.absolute()),'--wheel','--out-dir',str(wheelhouse)]
        for key,value in flags.items():command+=['--config-setting',f'cmake.define.{key}={value}']
        run(command,py,env,logfile)
        wheels=list(wheelhouse.glob('duckdb-'+version+'-*.whl'))
        if len(wheels)!=1:raise RuntimeError('expected exactly one custom DuckDB wheel')
        receipt['wheel']={'path':str(wheels[0]),'sha256':sha256(wheels[0])}
        normalized=make_source_receipt(work,sources,wheels[0],args.python.absolute())
        source_receipt=work/'source-receipt.json'
        source_receipt.write_text(json.dumps(normalized,indent=2,sort_keys=True)+'\n')
        receipt['source_receipt']={'path':str(source_receipt),'sha256':sha256(source_receipt)}
        receipt['native_files']=normalized['native_files']
        receipt['status']='built_unverified'
        return 0
    except Exception as exc:
        receipt['status']='failed';receipt['error']=str(exc);print(str(exc),file=sys.stderr);return 1
    finally:
        receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':raise SystemExit(main())
