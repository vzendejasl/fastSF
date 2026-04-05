#############################################################################################################################################
 # fastSF
 # 
 # Copyright (C) 2020, Mahendra K. Verma
 #
 # All rights reserved.
 # 
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions are met:
 #     1. Redistributions of source code must retain the above copyright
 #        notice, this list of conditions and the following disclaimer.
 #     2. Redistributions in binary form must reproduce the above copyright
 #        notice, this list of conditions and the following disclaimer in the
 #        documentation and/or other materials provided with the distribution.
 #     3. Neither the name of the copyright holder nor the
 #        names of its contributors may be used to endorse or promote products
 #        derived from this software without specific prior written permission.
 # 
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 # ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 # ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 #
 ############################################################################################################################################
 ##
 ##! \file Makefile
 #
 #   \brief Script to compile fastSF
 #
 #   \author Shubhadeep Sadhukhan, Shashwat Bhattacharya
 #   \date Feb 2020
 #   \copyright New BSD License
 #
 ############################################################################################################################################
##

MPICXX ?= mpic++
PKG_CONFIG ?= pkg-config
PKG_PACKAGES ?= yaml-cpp blitz
PREFIX ?=
H5SI_ROOT ?= $(PREFIX)
HDF5_ROOT ?= $(PREFIX)
CPPFLAGS += $(shell $(PKG_CONFIG) --cflags $(PKG_PACKAGES) 2>/dev/null)
LDLIBS += $(shell $(PKG_CONFIG) --libs $(PKG_PACKAGES) 2>/dev/null)
CXXFLAGS ?= -std=c++11 -O3 -DH5SI_ENABLE_MPI -DH5SI_ENABLE_MPI

ifneq ($(strip $(H5SI_ROOT)),)
CPPFLAGS += -I$(H5SI_ROOT)/include
LDFLAGS += -L$(H5SI_ROOT)/lib
RPATH_DIRS += $(H5SI_ROOT)/lib
endif

ifneq ($(strip $(HDF5_ROOT)),)
CPPFLAGS += -I$(HDF5_ROOT)/include
LDFLAGS += -L$(HDF5_ROOT)/lib
RPATH_DIRS += $(HDF5_ROOT)/lib
endif

RPATH_FLAGS := $(foreach dir,$(sort $(RPATH_DIRS)),-Wl,-rpath,$(dir))
LDLIBS += -lh5si -lhdf5

Structure: fastSF.cc
	$(MPICXX) $(CXXFLAGS) $(CPPFLAGS) fastSF.cc $(LDFLAGS) $(RPATH_FLAGS) $(LDLIBS) -o fastSF.out
	#mpic++ fastSF.cc -fstack-protector -O3 -lh5si -lhdf5 -lyaml-cpp -o fastSF.out
