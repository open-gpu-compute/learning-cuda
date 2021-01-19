## Installing PyTorch on AMD with ROCm 
### Option 1: Install using published PyTorch ROCm docker image
- Obtain docker image:
    `docker pull rocm/pytorch:latest-base`

- Clone PyTorch repository on the host:
```
cd ~
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
```

- Start a docker container using the downloaded image:
`sudo docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video rocm/pytorch:latest-base`
Note: This will mount your host home directory on /data in the container.

- Change to previous PyTorch checkout from within the running docker:
`cd /data/pytorch`

- Build PyTorch for ROCm:
By default, PyTorch will build for `gfx803`, `gfx900`, and `gfx906` simultaneously (to see which AMD uarch you have, run `/opt/rocm/bin/rocm_agent_enumerator`, `gfx900` are Vega10-type GPUs (MI25, Vega56, Vega64, ...) and work best). If you want to compile only for your uarch, `export PYTORCH_ROCM_ARCH=gfx900` to gfx803, gfx900, or gfx906. Then build with
`.jenkins/pytorch/build.sh`
This will first hipify the PyTorch sources and then compile, needing 16 GB of RAM to be available to the docker image.

Confirm working installation:
`.jenkins/pytorch/test.sh`
runs all CI unit tests and skips as appropriate on your system based on ROCm and, e.g., single or multi GPU configuration. No tests will fail if the compilation and installation is correct. Additionally, this step will install torchvision which most PyTorch script use to load models. E.g., running the PyTorch examples requires torchvision.
Individual test sets can be run with
`PYTORCH_TEST_WITH_ROCM=1 python test/test_nn.py --verbose`
Where `test_nn.py` can be replaced with any other test set.

Commit the container to preserve the pytorch install (from the host):
`sudo docker commit <container_id> -m 'pytorch installed'`

### Option 2: Install using minimal ROCm docker file
- Download pytorch dockerfile:
```
# This dockerfile is meant to be personalized, and serves as a template and demonstration.
# Modify it directly, but it is recommended to copy this dockerfile into a new build context (directory),
# modify to taste and modify docker-compose.yml.template to build and run it.

# It is recommended to control docker containers through 'docker-compose' https://docs.docker.com/compose/
# Docker compose depends on a .yml file to control container sets
# rocm-setup.sh can generate a useful docker-compose .yml file
# `docker-compose run --rm <rocm-terminal>`

# If it is desired to run the container manually through the docker command-line, the following is an example
# 'docker run -it --rm -v [host/directory]:[container/directory]:ro <user-name>/<project-name>'.

FROM ubuntu:18.04
MAINTAINER Michael Wootton <michael.wootton@amd>

# Initialize the image
# Modify to pre-install dev tools and ROCm packages
RUN apt-get update && apt-get install -y gnupg2
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && \
  curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
  sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list' && \
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  sudo \
  libelf1 \
  build-essential \
  bzip2 \
  ca-certificates \
  cmake \
  ssh \
  apt-utils \
  pkg-config \
  g++-multilib \
  gdb \
  git \
  less \
  libunwind-dev \
  libfftw3-dev \
  libelf-dev \
  libncurses5-dev \
  libomp-dev \
  libpthread-stubs0-dev \
  make \
  miopen-hip \
  python3-dev \
  python3-future \
  python3-yaml \
  python3-pip \
  vim \
  libssl-dev \
  libboost-dev \
  libboost-system-dev \
  libboost-filesystem-dev \
  libopenblas-dev \
  rpm \
  wget \
  net-tools \
  iputils-ping \
  libnuma-dev \
  rocm-dev \
  rocrand \
  rocblas \
  rocfft \
  hipcub \
  rocthrust \
  hipsparse && \
  curl -sL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
  sh -c 'echo deb [arch=amd64] http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main > /etc/apt/sources.list.d/llvm7.list' && \
  sh -c 'echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main >> /etc/apt/sources.list.d/llvm7.list' && \
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  clang-7 && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# fix capitalization in some cmake files...
RUN sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocsparse/lib/cmake/rocsparse/rocsparse-config.cmake
RUN sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocfft/lib/cmake/rocfft/rocfft-config.cmake
RUN sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/miopen/lib/cmake/miopen/miopen-config.cmake
RUN sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocblas/lib/cmake/rocblas/rocblas-config.cmake

# Grant members of 'sudo' group passwordless privileges
# Comment out to require sudo
#COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd

# This is meant to be used as an interactive developer container
# Create user rocm-user as member of sudo group
# Append /opt/rocm/bin to the system PATH variable
#RUN useradd --create-home -G sudo --shell /bin/bash rocm-user
#RUN usermod -a -G video rocm-user
#    sed --in-place=.rocm-backup 's|^\(PATH=.*\)"$|\1:/opt/rocm/bin"|' /etc/environment

#USER rocm-user
#WORKDIR /home/rocm-user
WORKDIR /root
ENV PATH="${PATH}:/opt/rocm/bin" HIP_PLATFORM="hcc"

#RUN \
#  curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
#  bash Anaconda3-5.0.1-Linux-x86_64.sh -b
#  rm Anaconda3-5.0.1-Linux-x86_64.sh

# The following are optional enhancements for the command-line experience
# Uncomment the following to install a pre-configured vim environment based on http://vim.spf13.com/
# 1.  Sets up an enhanced command line dev environment within VIM
# 2.  Aliases GDB to enable TUI mode by default
#RUN curl -sL https://j.mp/spf13-vim3 | bash && \
#    echo "alias gdb='gdb --tui'\n" >> ~/.bashrc

#RUN \
#  bash installers/Anaconda3-5.2.0-Linux-x86_64.sh -b

#ENV PATH="/home/rocm-user/anaconda3/bin:${PATH}" KMTHINLTO="1"
ENV KMTHINLTO="1" LANG="C.UTF-8" LC_ALL="C.UTF-8"

RUN \
  pip3 install setuptools

RUN \
  pip3 install pyyaml

RUN \
  pip3 install numpy scipy

RUN \
  pip3 install typing

RUN \
  pip3 install enum34

RUN \
  pip3 install hypothesis

RUN \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/clang-7 50 && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/clang++-7 50


# Default to a login shell
CMD ["bash", "-l"]
```



- Build docker image:
`cd pytorch_docker`
`sudo docker build .`
This should complete with a message "Successfully built <image_id>"

- Start a docker container using the new image:
`sudo docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device=/dev/dri --group-add video <image_id>`
Note: This will mount your host home directory on `/data` in the container.

- Clone pytorch master (on to the host):
```
cd ~
git clone https://github.com/pytorch/pytorch.git or git clone https://github.com/ROCmSoftwarePlatform/pytorch.git
cd pytorch
git submodule update --init --recursive
```
- Run "hipify" to prepare source code (in the container):
```
cd /data/pytorch/
python tools/amd_build/build_amd.py
```
- Build and install pytorch:
By default, PyTorch will build for gfx803, gfx900, and gfx906 simultaneously (to see which AMD uarch you have, run /opt/rocm/bin/rocm_agent_enumerator. If you want to compile only for your uarch, export PYTORCH_ROCM_ARCH=gfx900 to gfx803, gfx900, or gfx906. Then build with
`USE_MKLDNN=0 USE_ROCM=1 MAX_JOBS=4 python setup.py install --user`
Use MAX_JOBS=n to limit peak memory usage. If building fails try falling back to fewer jobs. 4 jobs assume available main memory of 16 GB or larger.

- Confirm working installation:
`.jenkins/pytorch/test.sh`

Individual test sets can be run with
`PYTORCH_TEST_WITH_ROCM=1 python test/test_nn.py --verbose`
Where test_nn.py can be replaced with any other test set.

- Commit the container to preserve the pytorch install (from the host):
`sudo docker commit <container_id> -m 'pytorch installed'`