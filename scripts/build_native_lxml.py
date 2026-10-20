#!/usr/bin/env python3
"""Build an isolated, source-bound lxml wheel and portable LGPL relink materials."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import zipfile


ZLIB_PATCH_COMMIT = "df84af25dc1942490e1d1c899a07619152a46148"
PATCHED_GZWRITE_SHA256 = "31d0da140edf382b82b13e2d87a22f59e158d9014ee026459e01cf437d0ab07d"
SOURCES = {
    "lxml": {
        "version": "6.1.3", "filename": "lxml-6.1.3.tar.gz",
        "url": "https://files.pythonhosted.org/packages/23/ad/28ecd7cb894d172f3c9c80a075eeeb2017ac62e3632cee05a5f9493547eb/lxml-6.1.3.tar.gz",
        "sha256": "45222d94ddd511536f3b2f7d9deae3b2339b4ce0f075f1ca25703b07cad9dd21",
    },
    "libxml2": {
        "version": "2.15.4", "filename": "libxml2-2.15.4.tar.xz",
        "url": "https://download.gnome.org/sources/libxml2/2.15/libxml2-2.15.4.tar.xz",
        "sha256": "98087fd181d9070724f3fbc65c7377db03038eb92bd882374daff44940138821",
    },
    "libxslt": {
        "version": "1.1.45", "filename": "libxslt-1.1.45.tar.xz",
        "url": "https://download.gnome.org/sources/libxslt/1.1/libxslt-1.1.45.tar.xz",
        "sha256": "9acfe68419c4d06a45c550321b3212762d92f41465062ca4ea19e632ee5d216e",
    },
    "libiconv": {
        "version": "1.19", "filename": "libiconv-1.19.tar.gz",
        "url": "https://ftp.gnu.org/pub/gnu/libiconv/libiconv-1.19.tar.gz",
        "sha256": "88dd96a8c0464eca144fc791ae60cd31cd8ee78321e67397e25fc095c4a19aa6",
    },
    "zlib": {
        "version": "1.3.2", "filename": "zlib-1.3.2.tar.gz",
        "url": "https://zlib.net/zlib-1.3.2.tar.gz",
        "sha256": "bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16",
    },
    "zlib_patch": {
        "filename": "zlib-df84af25.patch",
        "url": f"https://github.com/madler/zlib/commit/{ZLIB_PATCH_COMMIT}.patch",
        "sha256": "110ff14375733173d8aa54574473424fbd7dfe4b81f1ca34a759c6fe14b15b14",
    },
}
FEATURES = ["catalog", "html", "iconv", "regexp", "schematron", "xmlschema", "xpath", "zlib"]
CHECKS = {
    "xml", "xpath", "xmlschema", "schematron", "xslt", "html", "objectify",
    "encoding", "gzip", "gzip_default_disabled", "external_parameter_entity_blocked",
}

PROBE_SCRIPT = r'''
import ctypes, gzip, importlib.metadata, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from lxml import etree, html, objectify, isoschematron
root = pathlib.Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
native = ctypes.CDLL(etree.__file__)
zlib_version = native.zlibVersion
zlib_version.restype = ctypes.c_char_p
iconv = ctypes.c_int.in_dll(native, "_libiconv_version").value
tree = etree.fromstring(b'<root><item value="3">text</item></root>')
schema = etree.XMLSchema(etree.fromstring(b'<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema"><xs:element name="value" type="xs:integer"/></xs:schema>'))
schematron = isoschematron.Schematron(etree.fromstring(b'<schema xmlns="http://purl.oclc.org/dsdl/schematron"><pattern><rule context="root"><assert test="item">item required</assert></rule></pattern></schema>'))
xslt = etree.XSLT(etree.fromstring(b'<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform"><xsl:template match="/"><result><xsl:value-of select="root/item"/></result></xsl:template></xsl:stylesheet>'))
encoded = etree.Element("text")
encoded.text = "中文\ue7c7"
gzip_path = root / "sample.xml.gz"
gzip_path.write_bytes(gzip.compress(b"<root>compressed</root>", mtime=0))
try:
    etree.parse(str(gzip_path))
    gzip_default_disabled = False
except etree.XMLSyntaxError:
    gzip_default_disabled = True
dtd = root / "external.dtd"
dtd.write_text('<!ENTITY leaked "external-marker">')
external = ('<!DOCTYPE root [<!ENTITY % ext SYSTEM "' + dtd.as_uri() + '">%ext;]><root>&leaked;</root>').encode()
try:
    blocked = etree.fromstring(external).text != "external-marker"
except etree.XMLSyntaxError:
    blocked = True
checks = {
    "xml": etree.tostring(etree.fromstring(etree.tostring(tree))) == etree.tostring(tree),
    "xpath": tree.xpath("sum(item/@value)") == 3,
    "xmlschema": schema.validate(etree.fromstring(b"<value>42</value>")) and not schema.validate(etree.fromstring(b"<value>bad</value>")),
    "schematron": schematron.validate(tree),
    "xslt": xslt(tree).getroot().text == "text",
    "html": html.fromstring("<html><body><p>hello</p></body></html>").xpath("string(//p)") == "hello",
    "objectify": objectify.fromstring(b"<root><number>42</number></root>").number.pyval == 42,
    "encoding": etree.fromstring(etree.tostring(encoded, encoding="GB18030")).text == encoded.text,
    "gzip": etree.parse(str(gzip_path), etree.XMLParser(decompress=True)).getroot().text == "compressed",
    "gzip_default_disabled": gzip_default_disabled,
    "external_parameter_entity_blocked": blocked,
}
print(json.dumps({
    "lxml": importlib.metadata.version("lxml"),
    "lxml_compiled": etree.__version__,
    "libxml2_runtime": etree.LIBXML_VERSION,
    "libxml2_compiled": etree.LIBXML_COMPILED_VERSION,
    "libxslt_runtime": etree.LIBXSLT_VERSION,
    "libxslt_compiled": etree.LIBXSLT_COMPILED_VERSION,
    "libxml_features": sorted(etree.LIBXML_FEATURES),
    "libxml_compiled_features": sorted(etree.LIBXML_COMPILED_FEATURES),
    "zlib": zlib_version().decode(),
    "libiconv": f"{iconv >> 8}.{iconv & 255}",
    "checks": checks,
}))
'''

RELINK_SCRIPT = r'''#!/usr/bin/env python3
"""Relink the preserved extension objects, optionally replacing GNU libiconv."""
import argparse, hashlib, json, pathlib, subprocess

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--iconv-library", type=pathlib.Path)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parent
manifest = json.loads((root / "relink-manifest.json").read_text())
for item in manifest["inputs"]:
    path = root / item["path"]
    if sha(path) != item["sha256"]:
        raise ValueError("relink input checksum mismatch: " + str(path))
replacement = args.iconv_library.resolve() if args.iconv_library else None
if replacement and not replacement.is_file():
    raise ValueError("replacement library is missing")
original_iconv = str(pathlib.Path(manifest["original_root"]) / "prefix/lib/libiconv.a")
if replacement and not any(original_iconv in item["argv"] for item in manifest["commands"]):
    raise ValueError("replacement library is not used by captured commands")
output = args.output.absolute()
output.mkdir(parents=True, exist_ok=False)
results = []
for item in manifest["commands"]:
    command = [arg.replace(manifest["original_root"], str(root)) for arg in item["argv"]]
    if replacement:
        old = str(root / "prefix/lib/libiconv.a")
        if old in command:
            command = [str(replacement) if arg == old else arg for arg in command]
    target = output / pathlib.Path(item["output"]).name
    command[command.index("-o") + 1] = str(target)
    command = [arg for arg in command if not arg.startswith("-Wl,-Map,")]
    directory = item["cwd"].replace(manifest["original_root"], str(root))
    subprocess.run(command, cwd=directory, check=True, timeout=300)
    results.append({"path": str(target), "sha256": sha(target)})
print(json.dumps({"outputs": results}))
'''


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.absolute()), "sha256": sha256(path)}


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_command(command, *, cwd, env, log, timeout=1800):
    with Path(log).open("ab") as stream:
        stream.write(("\n$ " + shlex.join(command) + "\n").encode())
        stream.flush()
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def extract_source(archive: Path, directory: Path) -> None:
    with tarfile.open(archive) as source:
        source.extractall(directory, filter="data")


def validate_probe(probe: dict) -> None:
    expected = {
        "lxml": "6.1.3", "lxml_compiled": "6.1.3",
        "libxml2_runtime": [2, 15, 4], "libxml2_compiled": [2, 15, 4],
        "libxslt_runtime": [1, 1, 45], "libxslt_compiled": [1, 1, 45],
        "libiconv": "1.19", "zlib": "1.3.2",
        "libxml_features": FEATURES, "libxml_compiled_features": FEATURES,
    }
    for key, value in expected.items():
        if probe.get(key) != value:
            raise ValueError(f"native probe {key} mismatch: {probe.get(key)!r}")
    checks = probe.get("checks", {})
    if not isinstance(checks, dict) or any(checks.get(key) is not True for key in CHECKS):
        raise ValueError("native probe missing or failed capability checks")


def validate_linkage(text: str) -> None:
    if "not found" in text:
        raise ValueError("unresolved native library")
    if any(name in text for name in ("libxml2.so", "libxslt.so", "libexslt.so", "libiconv.so", "libz.so")):
        raise ValueError("XML wheel must use the private static dependency closure")


def linker_command(ldshared: str, compiler: str) -> list[str]:
    command = shlex.split(ldshared)
    command[0] = compiler
    # Keep the private XML symbols hidden while making native version evidence observable.
    return [
        "-Wl,--exclude-libs,libexslt.a:libxslt.a:libxml2.a"
        if item == "-Wl,--exclude-libs,ALL" else item
        for item in command
    ]


def build(python: Path, sources: Path, output: Path, jobs: int = 4) -> dict:
    python, sources, output = python.absolute(), sources.absolute(), output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"build output already exists: {output}")
    if jobs not in range(1, 5):
        raise ValueError("jobs must be between 1 and 4")
    for item in SOURCES.values():
        path = sources / item["filename"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"source checksum mismatch or missing input: {path}")
    tools = {}
    for name in ("cc", "ld", "ar", "ranlib", "make", "patch", "readelf", "nm"):
        tool = shutil.which(name)
        if not tool:
            raise ValueError(f"required build tool missing: {name}")
        tools[name] = str(Path(tool).resolve())
    identity = json.loads(subprocess.check_output(
        [str(python), "-I", "-B", "-c",
         "import json,platform,sysconfig; print(json.dumps({'version':platform.python_version(),'include':sysconfig.get_path('include'),'ldshared':sysconfig.get_config_var('LDSHARED')}))"],
        text=True, timeout=30,
    ))
    if identity["version"] != "3.14.7":
        raise ValueError("reviewed lxml build requires Python 3.14.7")
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _build(python, sources, output, jobs, tools, identity)
    except BaseException as exc:
        write_json(output / "FAILED.json", {
            "status": "failed", "ready": False, "error": f"{type(exc).__name__}: {exc}",
        })
        raise


def _build(python, source_inputs, work, jobs, tools, identity):
    inputs_dir, source_dir, prefix = work / "inputs", work / "sources", work / "prefix"
    for path in (inputs_dir, source_dir, prefix):
        path.mkdir()
    for item in SOURCES.values():
        source = source_inputs / item["filename"]
        target = inputs_dir / source.name
        shutil.copyfile(source, target)
        if sha256(target) != item["sha256"]:
            raise ValueError("source changed during snapshot")
        if source.name.endswith((".tar.gz", ".tar.xz")):
            extract_source(target, source_dir)
    directories = {name: source_dir / f"{name}-{item['version']}"
                   for name, item in SOURCES.items() if "version" in item}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LD_", "PYTHON", "PIP_", "UV_")) and key not in ("VIRTUAL_ENV",)}
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "CC": tools["cc"], "AR": tools["ar"], "RANLIB": tools["ranlib"],
        "CFLAGS": "-O2 -g0 -fPIC", "CPPFLAGS": f"-I{prefix / 'include'}",
        "LDFLAGS": f"-L{prefix / 'lib'} -Wl,-Bsymbolic-functions",
        "MAKEFLAGS": f"-j{jobs}", "PKG_CONFIG_PATH": str(prefix / "lib/pkgconfig"),
    })
    commands = []
    log = work / "build.log"

    def run(command, directory, timeout=1800):
        commands.append({"argv": command, "cwd": str(directory)})
        run_command(command, cwd=directory, env=env, log=log, timeout=timeout)

    run([tools["patch"], "--batch", "--fuzz=0", "-p1", "-i",
         str(inputs_dir / SOURCES["zlib_patch"]["filename"])], directories["zlib"], 30)
    patched_gzwrite = directories["zlib"] / "gzwrite.c"
    if sha256(patched_gzwrite) != PATCHED_GZWRITE_SHA256:
        raise ValueError("patched zlib source checksum mismatch")
    write_json(work / "patch-application.json", {
        "status": "passed", "fuzz": 0, "patch": record(inputs_dir / SOURCES["zlib_patch"]["filename"]),
        "post_patch": record(patched_gzwrite), "commit": ZLIB_PATCH_COMMIT,
    })
    configure = {
        "zlib": ["--static", f"--prefix={prefix}"],
        "libiconv": ["--disable-shared", "--enable-static", "--with-pic", f"--prefix={prefix}"],
        "libxml2": [
            "--disable-shared", "--enable-static", "--with-pic", "--without-python",
            "--with-schematron", f"--with-zlib={prefix}", f"--with-iconv={prefix}",
            f"--prefix={prefix}",
        ],
        "libxslt": [
            "--disable-shared", "--enable-static", "--with-pic", "--without-python",
            "--without-crypto", f"--with-libxml-prefix={prefix}", f"--prefix={prefix}",
        ],
    }
    for name in ("zlib", "libiconv", "libxml2", "libxslt"):
        directory = directories[name]
        run(["./configure", *configure[name]], directory, 300)
        run([tools["make"], f"-j{jobs}"], directory)
        run([tools["make"], "-j1", "install"], directory)

    library_names = ["libexslt.a", "libxslt.a", "libxml2.a", "libiconv.a", "libz.a"]
    libraries = [prefix / "lib" / name for name in library_names]
    if any(not path.is_file() for path in libraries):
        raise ValueError("private static library build is incomplete")
    lxml = directories["lxml"]
    licenses = lxml / "native-licenses"
    licenses.mkdir()
    license_sources = {
        "libiconv-COPYING.LIB": directories["libiconv"] / "COPYING.LIB",
        "libcharset-COPYING.LIB": directories["libiconv"] / "libcharset/COPYING.LIB",
        "zlib-LICENSE": directories["zlib"] / "LICENSE",
        "libxml2-Copyright": directories["libxml2"] / "Copyright",
        "libxslt-Copyright": directories["libxslt"] / "Copyright",
    }
    for name, source in license_sources.items():
        shutil.copyfile(source, licenses / name)
    (licenses / "NOTICE.txt").write_text(
        "This wheel statically includes GNU libiconv and libcharset 1.19 under LGPL 2.1.\n"
        "The accompanying relink-materials.tar.gz contains corresponding source,\n"
        "object files, static libraries and relink instructions. It permits replacement\n"
        "of GNU libiconv and relinking for modified-library debugging and personal use.\n"
        "It also includes zlib 1.3.2 with upstream fix " + ZLIB_PATCH_COMMIT + ",\n"
        "libxml2 2.15.4, and libxslt/libexslt 1.1.45. See the included license texts.\n"
        "Keep the relink archive available alongside any distribution of this wheel.\n"
    )
    import configparser
    config = configparser.ConfigParser()
    config.read(lxml / "setup.cfg")
    if not config.has_section("metadata"):
        config.add_section("metadata")
    config.set("metadata", "license_files",
               "\nLICENSE.txt\nLICENSES.txt\ndoc/licenses/*.txt\nnative-licenses/*")
    with (lxml / "setup.cfg").open("w") as stream:
        config.write(stream)
    write_json(work / "license-manifest.json", [record(path) for path in sorted(licenses.iterdir())])

    backend_paths = json.loads(subprocess.check_output(
        [str(python), "-I", "-B", "-c",
         "import importlib.metadata,json,pathlib; names=('setuptools','wheel','build','packaging','pyproject-hooks','Cython'); print(json.dumps([str(importlib.metadata.distribution(n).locate_file(p)) for n in names for p in importlib.metadata.files(n) if pathlib.Path(importlib.metadata.distribution(n).locate_file(p)).is_file() and not str(p).endswith('.pyc')]))"],
        text=True, timeout=30,
    ))
    backend_paths += [str(path) for path in Path(identity["include"]).rglob("*.h")]
    write_json(work / "build-backend-inputs.json",
               [record(Path(path)) for path in sorted(set(backend_paths))])
    link_log = work / "link-commands.jsonl"
    link_maps = work / "link-maps"
    link_maps.mkdir()
    linker = work / "record-link.py"
    link_command = linker_command(identity["ldshared"], tools["cc"])
    linker.write_text(
        f"#!{python}\n"
        "import json,pathlib,subprocess,sys\n"
        f"command={link_command!r}+sys.argv[1:]\n"
        "target=pathlib.Path(command[command.index('-o')+1]).absolute()\n"
        f"command.append('-Wl,-Map,'+str(pathlib.Path({str(link_maps)!r})/(target.name+'.map')))\n"
        "subprocess.run(command,check=True)\n"
        f"with open({str(link_log)!r},'a') as stream:\n"
        " stream.write(json.dumps({'argv':command,'cwd':str(pathlib.Path.cwd()),'output':str(target)})+'\\n')\n"
    )
    linker.chmod(0o755)
    env.pop("STATICBUILD", None)
    env.pop("STATIC_DEPS", None)
    env.update({
        "STATIC": "true", "WITHOUT_CYTHON": "true",
        "LXML_STATIC_INCLUDE_DIRS": os.pathsep.join(map(str, [prefix / "include", prefix / "include/libxml2"])),
        "LXML_STATIC_LIBRARY_DIRS": str(prefix / "lib"),
        "LXML_STATIC_BINARIES": os.pathsep.join(map(str, libraries)),
        "XML2_CONFIG": str(prefix / "bin/xml2-config"),
        "XSLT_CONFIG": str(prefix / "bin/xslt-config"),
        "LDSHARED": str(linker),
    })
    write_json(work / "build-environment.json", {
        key: env[key] for key in (
            "CC", "AR", "RANLIB", "CFLAGS", "CPPFLAGS", "LDFLAGS", "MAKEFLAGS",
            "PKG_CONFIG_PATH", "STATIC", "WITHOUT_CYTHON", "LXML_STATIC_INCLUDE_DIRS",
            "LXML_STATIC_LIBRARY_DIRS", "LXML_STATIC_BINARIES", "XML2_CONFIG",
            "XSLT_CONFIG", "LDSHARED",
        )
    })
    wheelhouse = work / "wheelhouse"
    wheelhouse.mkdir()
    run([str(python), "-I", "-B", "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(wheelhouse), str(lxml)], lxml, 2400)
    wheels = list(wheelhouse.glob("lxml-6.1.3-cp314-cp314-linux_x86_64.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one CPython 3.14 Linux lxml wheel")
    wheel = wheels[0]
    wheel_tree = work / "wheel-tree"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in license_sources:
            if not any(member.endswith("/" + name) for member in names):
                raise ValueError(f"wheel omits native license: {name}")
        archive.extractall(wheel_tree)
    probe_script = work / "probe.py"
    probe_script.write_text(PROBE_SCRIPT)
    probe = json.loads(subprocess.check_output(
        [str(python), "-I", "-B", str(probe_script), str(wheel_tree), str(work / "probe-fixtures")],
        text=True, timeout=60,
    ))
    validate_probe(probe)
    write_json(work / "native-probe.json", probe)
    linkage = {}
    native_files = {}
    for path in sorted(wheel_tree.rglob("*.so")):
        text = subprocess.check_output(["ldd", str(path)], text=True, timeout=30)
        validate_linkage(text)
        linkage[str(path.relative_to(wheel_tree))] = {
            "ldd": text,
            "dynamic": subprocess.check_output([tools["readelf"], "-d", str(path)], text=True, timeout=30),
        }
        for line in text.splitlines():
            words = line.split()
            candidates = [word for word in words if word.startswith("/")]
            if not candidates:
                continue
            library = Path(candidates[0]).resolve()
            owner = subprocess.check_output(["dpkg-query", "-S", str(library)], text=True, timeout=30).split(": ", 1)[0]
            native_files[str(library)] = record(library) | {"owner": "dpkg:" + owner.split(":")[0]}
    write_json(work / "native-linkage.json", linkage)
    write_json(work / "commands.json", commands)
    relink = make_relink_bundle(work, lxml, wheel, libraries, python)
    inputs = [record(path) for path in sorted(inputs_dir.iterdir())]
    inputs += [record(path) for path in (
        Path(__file__).absolute(), work / "build-backend-inputs.json",
        work / "license-manifest.json", lxml / "setup.cfg", work / "commands.json",
        work / "native-probe.json", work / "native-linkage.json", log, probe_script,
        linker, link_log, work / "build-environment.json",
        work / "patch-application.json", patched_gzwrite,
    )]
    inputs += [record(path) for path in sorted(licenses.iterdir())]
    inputs += [relink[key] for key in ("bundle", "manifest", "qualification")]
    links = [record(path) for path in libraries + sorted(link_maps.iterdir())]
    links += [record(path) for path in sorted(wheel_tree.rglob("*.so"))]
    receipt = {
        "schema_version": 1, "status": "built_unverified", "ready": False,
        "name": "lxml", "version": "6.1.3", "wheel": record(wheel),
        "wheel_sha256": sha256(wheel), "source_revision": SOURCES["lxml"]["sha256"],
        "sources": SOURCES, "toolchain": {name: record(Path(path)) for name, path in tools.items()},
        "inputs": inputs, "inputs_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "native_link_artifacts": links, "native_files": list(native_files.values()),
        "effective_versions": {name: SOURCES[name]["version"] for name in ("lxml", "libxml2", "libxslt", "libiconv", "zlib")},
        "native_policy": {
            "zlib_patch_commit": ZLIB_PATCH_COMMIT,
            "zlib_patch_sha256": SOURCES["zlib_patch"]["sha256"],
            "zlib_patched_gzwrite_sha256": PATCHED_GZWRITE_SHA256,
            "libxml_features": FEATURES,
        },
        "patch_application": record(work / "patch-application.json"),
        "runtime_probe": probe, "license_manifest": record(work / "license-manifest.json"),
        "relink": relink, "jobs": jobs,
    }
    receipt["toolchain"]["python"] = record(python)
    write_json(work / "receipt.json", receipt)
    return receipt


def make_relink_bundle(work, lxml, wheel, libraries, python):
    bundle = work / "relink-materials"
    bundle.mkdir()
    commands = [json.loads(line) for line in (work / "link-commands.jsonl").read_text().splitlines()]
    if not commands:
        raise ValueError("no captured native link commands")
    objects = list((lxml / "build").rglob("*.o"))
    if not objects:
        raise ValueError("LGPL relink object files missing")
    files = objects + libraries + list((work / "prefix/include").rglob("*"))
    files += list((work / "inputs").iterdir()) + list((lxml / "native-licenses").iterdir())
    files += [wheel, lxml / "setup.cfg", work / "license-manifest.json",
              work / "commands.json", work / "record-link.py", work / "build-environment.json",
              work / "patch-application.json", work / "sources/zlib-1.3.2/gzwrite.c"]
    files = sorted({path for path in files if path.is_file()})
    for path in files:
        target = bundle / path.relative_to(work)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    manifest = {
        "original_root": str(work), "commands": commands,
        "inputs": [{"path": str(path.relative_to(work)), "sha256": sha256(path)} for path in files],
    }
    write_json(bundle / "relink-manifest.json", manifest)
    (bundle / "relink.py").write_text(RELINK_SCRIPT)
    (bundle / "README.txt").write_text(
        "Relinking GNU libiconv in this lxml wheel\n\n"
        "Source archives and the zlib patch are in inputs/. All supplied sources retain\n"
        "their upstream licenses. Unpack and build a modified GNU libiconv as a PIC\n"
        "static library, then run:\n"
        "  python3 relink.py --iconv-library /path/to/replacement/libiconv.a --output new-links\n"
        "Use the original compiler/toolchain described in commands.json and the receipt.\n"
        "The command checks preserved inputs, keeps them unchanged, and writes new .so\n"
        "files. Replace matching lxml extension files in a copy of the supplied wheel,\n"
        "then use the wheel package's unpack/pack commands to regenerate wheel RECORD.\n"
        "The new wheel is your modified artifact; its original approval does not apply.\n"
        "Distribute these relink materials with the wheel, or provide equivalent access.\n"
    )
    # Exercise the distributed archive after relocation with a separate library copy.
    archive = work / "relink-materials.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(bundle, arcname="relink-materials")
    relocated = work / "relink-verification"
    relocated.mkdir()
    extract_source(archive, relocated)
    replacement = relocated / "replacement-libiconv.a"
    shutil.copyfile(work / "prefix/lib/libiconv.a", replacement)
    replay = relocated / "relink-materials/relink.py"
    result = json.loads(subprocess.check_output(
        [str(python), "-I", "-B", str(replay), "--iconv-library", str(replacement),
         "--output", str(relocated / "outputs")], text=True, timeout=600,
    ))
    original = {Path(item["output"]).name: sha256(Path(item["output"])) for item in commands}
    outputs = []
    for item in result["outputs"]:
        name = Path(item["path"]).name
        if item["sha256"] != original[name]:
            raise ValueError(f"relocated relink changed native extension: {name}")
        outputs.append({"name": name, "original_sha256": original[name], "relinked_sha256": item["sha256"]})
    qualification = work / "relink-qualification.json"
    write_json(qualification, {
        "status": "passed", "relocated_bundle": True,
        "replacement_library_sha256": sha256(replacement), "outputs": outputs,
    })
    return {"bundle": record(archive), "manifest": record(bundle / "relink-manifest.json"),
            "qualification": record(qualification)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    receipt = build(args.python, args.sources, args.output, args.jobs)
    print(json.dumps({"status": receipt["status"], "wheel": receipt["wheel"],
                      "receipt": str(args.output.absolute() / "receipt.json")}))


if __name__ == "__main__":
    main()
