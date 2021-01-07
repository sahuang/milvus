# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

DEMOS_SRC=$(wildcard demo_sift1M.cpp)
DEMOS=$(DEMOS_SRC:.cpp=)


all: $(DEMOS)

clean:
	rm -f $(DEMOS)

%: %.cpp ../libfaiss.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -o $@ $^ $(LDFLAGS) -I../ $(LIBS)


.PHONY: all clean
