F77 = mpif77
USE_MPI = yes
USE_INTEL = yes

EIGEN=/projects/sash2458/apps/eigen/
BOOST=/projects/sash2458/apps/boost_1_57_0/
LIBIGL=/projects/sash2458/apps/libigl/include/
PYSCF=/projects/sash2458/newApps/pyscf/pyscf/lib/
LIBCINT=/projects/sash2458/newApps/pyscf/pyscf/lib/deps/lib
#EIGEN=/projects/ilsa8974/apps/eigen/
#BOOST=/projects/ilsa8974/apps/boost_1_66_0/
#LIBIGL=/projects/ilsa8974/apps/libigl/include/

OPT = -std=c++14 -g  -O3
#OPT = -std=c++14 -g 
FLAGS =  -I./VMC -I./utils -I./Wavefunctions -I./Wavefunctions/RealSpace -I${EIGEN} -I${BOOST} -I${BOOST}/include -I${LIBIGL} -I/opt/local/include/openmpi-mp/ #-DComplex


LFLAGS = -L${PYSCF} -lcgto -lnp_helper -L${LIBCINT} -lcint




ifeq ($(USE_INTEL), yes) 
	FLAGS += -qopenmp
	DFLAGS += -qopenmp
	ifeq ($(USE_MPI), yes) 
		CXX = mpiicpc
		CC = mpiicpc
		LFLAGS += -L${BOOST}/stage/lib -lboost_serialization -lboost_mpi
	else
		CXX = icpc
		CC = icpc
		LFLAGS += -L${BOOST}/stage/lib -lboost_serialization-mt
		FLAGS += -DSERIAL
		DFLAGS += -DSERIAL
	endif
else
	FLAGS += -openmp
	DFLAGS += -openmp
	ifeq ($(USE_MPI), yes) 
		CXX = mpicxx
		CC = mpicxx
		LFLAGS += -L/opt/local/lib -lboost_serialization-mt -lboost_mpi-mt
	else
		CXX = g++
		CC = g++
		LFLAGS += -L/opt/local/lib -lboost_serialization-mt
		FLAGS += -DSERIAL
		DFLAGS += -DSERIAL
	endif
endif

# Host specific configurations.
HOSTNAME := $(shell hostname)
ifneq ($(filter dft node%, $(HOSTNAME)),)
include dft.mk
endif

OBJ_VMC = obj/staticVariables.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Slater.o \
	obj/AGP.o \
	obj/Pfaffian.o \
	obj/rJastrow.o \
	obj/JastrowTerms.o \
	obj/gaussianBasis.o\
	obj/slaterBasis.o\
	obj/rWalker.o \
	obj/rWalkerHelper.o \
	obj/Jastrow.o \
	obj/Gutzwiller.o \
	obj/CPS.o \
	obj/Correlator.o \
	obj/ShermanMorrisonWoodbury.o\
	obj/excitationOperators.o\
    obj/statistics.o \
    obj/sr.o \
    obj/VMC.o \
    obj/evaluateE.o 

OBJ_SLATERTOGAUSSIAN = obj/slaterToGaussian.o \
	obj/_slaterToGaussian.o

OBJ_GFMC = obj/staticVariables.o \
	obj/GFMC.o \
	obj/input.o \
	obj/integral.o\
	obj/SHCIshm.o \
	obj/Determinants.o \
	obj/Slater.o \
	obj/Pfaffian.o \
	obj/Jastrow.o \
	obj/gaussianBasis.o\
	obj/slaterBasis.o\
	obj/Gutzwiller.o \
	obj/CPS.o \
	obj/evaluateE.o \
	obj/excitationOperators.o\
	obj/ShermanMorrisonWoodbury.o\
	obj/statistics.o \
	obj/sr.o \
	obj/Correlator.o


obj/%.o: %.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: Wavefunctions/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: Wavefunctions/RealSpace/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: utils/%.cpp  
	$(CXX) $(FLAGS) $(OPT) -c $< -o $@
obj/%.o: VMC/%.cpp  
	$(CXX) $(FLAGS) -I./VMC $(OPT) -c $< -o $@
obj/%.o: GFMC/%.cpp  
	$(CXX) $(FLAGS) -I./GFMC $(OPT) -c $< -o $@
obj/%.o: executables/%.cpp  
	$(CXX) $(FLAGS) -I./GFMC -I./VMC $(OPT) -c $< -o $@


all: bin/VMC bin/GFMC bin/slaterToGaussian #bin/sPT  bin/GFMC

bin/GFMC	: $(OBJ_GFMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/GFMC $(OBJ_GFMC) $(LFLAGS)

bin/VMC	: $(OBJ_VMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/VMC $(OBJ_VMC) $(LFLAGS)

bin/slaterToGaussian	: 
	icpc $(OPT) -I./utils/ -I$(BOOST) -I$(EIGEN) -c executables/slaterToGaussian.cpp -o obj/slaterToGaussian.o
	ifort -O3 -c utils/_slaterToGaussian.f -o obj/_slaterToGaussian.o
	icpc $(OPT) -o  bin/slaterToGaussian $(OBJ_SLATERTOGAUSSIAN) obj/slaterBasis.o -limf -lifcore


bin/sPT	: $(OBJ_sPT) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/sPT $(OBJ_sPT) $(LFLAGS)

bin/CI	: $(OBJ_CI) 
	$(CXX)   $(FLAGS) $(OPT) -o  bin/CI $(OBJ_CI) $(LFLAGS)

VMC2	: $(OBJ_VMC) 
	$(CXX)   $(FLAGS) $(OPT) -o  VMC2 $(OBJ_VMC) $(LFLAGS)

clean :
	find . -name "*.o"|xargs rm 2>/dev/null;rm -f bin/* >/dev/null 2>&1

