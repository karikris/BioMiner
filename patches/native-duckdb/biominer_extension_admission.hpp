// BioMiner Linux x86_64 extension admission. MIT licensed; uses DuckDB's SHA256.
#ifndef BIOMINER_EXTENSION_ADMISSION_HPP
#define BIOMINER_EXTENSION_ADMISSION_HPP
#include "biominer_extension_policy.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace biominer_native {
struct Snapshot {
    int fd;
    explicit Snapshot(int value):fd(value){}
    Snapshot(const Snapshot &)=delete;
    Snapshot &operator=(const Snapshot &)=delete;
    Snapshot(Snapshot &&other):fd(other.fd){other.fd=-1;}
    ~Snapshot(){if(fd>=0)close(fd);}
    std::string path() const{return "/proc/self/fd/"+std::to_string(fd);}
};
inline void Require(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
inline void ReadAt(int fd,void *buffer,size_t size,size_t offset){
    size_t done=0;
    while(done<size){auto n=pread(fd,static_cast<char *>(buffer)+done,size-done,offset+done);
        if(n<0&&errno==EINTR)continue;Require(n>0,"extension snapshot read failed");done+=n;}
}
// Program headers rather than section headers: stripped ELF files remain auditable.
inline std::set<std::string> Needed(int fd){
    struct stat st;Require(fstat(fd,&st)==0,"ELF stat failed");
    Elf64_Ehdr eh;ReadAt(fd,&eh,sizeof(eh),0);
    Require(!memcmp(eh.e_ident,ELFMAG,SELFMAG)&&eh.e_ident[EI_CLASS]==ELFCLASS64&&
        eh.e_ident[EI_DATA]==ELFDATA2LSB&&eh.e_machine==EM_X86_64&&eh.e_type==ET_DYN,"unsupported extension ELF");
    Require(eh.e_phentsize==sizeof(Elf64_Phdr)&&eh.e_phnum>0&&eh.e_phnum<=256,"unsupported ELF headers");
    std::vector<Elf64_Phdr> ph(eh.e_phnum);ReadAt(fd,ph.data(),ph.size()*sizeof(Elf64_Phdr),eh.e_phoff);
    std::vector<Elf64_Dyn> dyn;
    for(auto &p:ph){
        Require(p.p_offset<=static_cast<uint64_t>(st.st_size)&&p.p_filesz<=static_cast<uint64_t>(st.st_size)-p.p_offset,"invalid ELF bounds");
        if(p.p_type==PT_DYNAMIC){Require(p.p_filesz<=1024*1024&&p.p_filesz%sizeof(Elf64_Dyn)==0,"invalid ELF dynamic table");
            dyn.resize(p.p_filesz/sizeof(Elf64_Dyn));ReadAt(fd,dyn.data(),p.p_filesz,p.p_offset);}
    }
    Require(!dyn.empty(),"ELF dynamic table missing");
    uint64_t address=0,length=0;std::vector<uint64_t> needed;
    for(auto &d:dyn){if(d.d_tag==DT_NULL)break;
        Require(d.d_tag!=DT_RPATH&&d.d_tag!=DT_RUNPATH,"extension RPATH/RUNPATH is unsupported");
        if(d.d_tag==DT_STRTAB)address=d.d_un.d_ptr;if(d.d_tag==DT_STRSZ)length=d.d_un.d_val;
        if(d.d_tag==DT_NEEDED)needed.push_back(d.d_un.d_val);
        Require(d.d_tag!=DT_AUDIT&&d.d_tag!=DT_DEPAUDIT&&d.d_tag!=DT_FILTER&&d.d_tag!=DT_AUXILIARY,"unsupported ELF dependency discovery");}
    Require(length>0&&length<=16*1024*1024,"invalid ELF strings");
    uint64_t offset=0;bool found=false;
    for(auto &p:ph){if(p.p_type==PT_LOAD&&address>=p.p_vaddr&&address-p.p_vaddr<=p.p_filesz&&length<=p.p_filesz-(address-p.p_vaddr)){
        offset=p.p_offset+(address-p.p_vaddr);found=true;break;}}
    Require(found,"ELF strings are not mapped");std::vector<char> strings(length);ReadAt(fd,strings.data(),length,offset);
    std::set<std::string> result;
    for(auto index:needed){Require(index<length,"invalid ELF dependency index");
        auto end=std::find(strings.begin()+index,strings.end(),'\0');Require(end!=strings.end(),"unterminated ELF dependency");
        std::string name(strings.begin()+index,end);Require(!name.empty()&&name.find('/')==std::string::npos,"invalid dependency name");result.insert(name);}
    return result;
}
// Require the entire recorded dependency closure to be loaded from exact approved
// paths already. This deliberately rejects unsupported lazy/custom discovery.
// The launcher verifies the rest of the process baseline; this narrows admission.
template<class Hash> void VerifyClosure(const Extension &entry,Hash hash){
    std::ifstream maps("/proc/self/maps");Require(bool(maps),"loaded library inventory unavailable");
    std::set<std::string> paths;std::string line;
    while(std::getline(maps,line)){auto start=line.find('/');if(start!=std::string::npos)paths.insert(line.substr(start));}
    std::set<std::string> names;for(const auto &dep:entry.dependencies){
        Require(names.insert(dep.soname).second,"duplicate dependency name");
        Require(paths.count(dep.path)>0,"approved dependency is not loaded");
        for(const auto &path:paths){auto slash=path.rfind('/');auto base=path.substr(slash+1);
            Require(base!=dep.soname||path==dep.path,"dependency loaded from an unapproved path");}
        int fd=open(dep.path.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);Require(fd>=0,"approved dependency unavailable");Snapshot library(fd);
        Require(hash(fd)==dep.sha256,"dependency SHA256 mismatch");
        auto needed=Needed(fd);for(const auto &name:needed){bool approved=false;
            for(const auto &candidate:entry.dependencies){if(candidate.soname==name)approved=true;}
            Require(approved,"dependency closure is incomplete");}}
}
template<class Hash> Snapshot Admit(const std::string &filename,const std::string &name,Hash hash){
    const Extension *entry=nullptr;for(const auto &candidate:Policy()){if(candidate.name==name){entry=&candidate;break;}}
    Require(entry!=nullptr,"extension is not in compiled allowlist");
    int original=open(filename.c_str(),O_RDONLY|O_CLOEXEC|O_NOFOLLOW);Require(original>=0,"extension file unavailable");Snapshot source(original);
    struct stat st;Require(fstat(original,&st)==0&&S_ISREG(st.st_mode)&&st.st_size>0&&st.st_size<=512LL*1024*1024,"invalid extension file");
    int fd=static_cast<int>(syscall(SYS_memfd_create,"biominer-audited-extension",MFD_CLOEXEC|MFD_ALLOW_SEALING));
    Require(fd>=0,"sealed extension snapshots unavailable");Snapshot snapshot(fd);
    char buffer[65536];off_t offset=0;
    while(offset<st.st_size){auto size=static_cast<size_t>(std::min<off_t>(sizeof(buffer),st.st_size-offset));ReadAt(original,buffer,size,offset);
        size_t done=0;while(done<size){auto n=write(fd,buffer+done,size-done);if(n<0&&errno==EINTR)continue;Require(n>0,"snapshot write failed");done+=n;}offset+=size;}
    Require(fcntl(fd,F_ADD_SEALS,F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL)==0,"snapshot sealing failed");
    Require(hash(fd)==entry->sha256,"extension SHA256 mismatch");
    auto needed=Needed(fd);std::set<std::string> declared;
    for(const auto &dep:entry->dependencies)declared.insert(dep.soname);
    for(const auto &name:needed)Require(declared.count(name)>0,"extension dependency is unapproved");
    VerifyClosure(*entry,hash);return snapshot;
}
}
#endif
