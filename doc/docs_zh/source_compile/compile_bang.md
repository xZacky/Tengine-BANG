# 源码编译（BANG）

## How to build

### Build for Linux

需要在安装好寒武纪驱动、CNToolkit和CNNL的Ubuntu/Debian/CentOS系统上进行Tengine（BANG）的源码编译。

寒武纪驱动、CNToolkit和CNNL的安装请参考寒武纪开发者社区的开发者文档。

### setup CNToolkit
```bash
export CNTOOLKIT_ROOT=<path to your CNToolkit>
```
### build
```bash
cd <tengine-lite-root-dir>
mkdir build-linux-bang
cd build-linux-bang
cmake -DTENGINE_ENABLE_BANG=ON ..

$ make
$ make install
```
编译完成后可以在build-linux-bang/install文件夹下得到以下文件：
```
install 
    ├── include 
    │     └── tengine 
    │     ├── c_api.h 
    │     └── defines.h 
    └── lib
         ├── libtengine-lite-static.a 
         └── libtengine-lite.so 
```
