#!/bin/bash

printf "\n\nRunning Tests for VMC/GFMC/FCIQMC\n"
printf "======================================================\n"

MPICOMMAND="mpirun -np 4"
VMCPATH="../../bin/VMC vmc.json"
CIPATH="../../bin/VMC ci.json"
LANCZOSPATH="../../bin/VMC lanczos.dat"
GFMCPATH="../../bin/GFMC gfmc.json"
DMCPATH="../../bin/DMC dmc.json"
FCIQMCPATH="../../../bin/FCIQMC fciqmc.json"
DIRECTPATH="../../bin/VMC direct.json"
NONDIRECTPATH="../../bin/VMC nondirect.json"
here=`pwd`
tol=1.0e-7
clean=0

# VMC tests

cd $here/hubbard_1x10
../clean.sh
printf "...running hubbard_1x10\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
#printf "...running hubbard_1x10 lanczos\n"
#$MPICOMMAND $LANCZOSPATH > lanczos.out
#python2 ../testEnergy.py 'lanczos' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/hubbard_1x10ghf
../clean.sh
printf "...running hubbard_1x10 ghf\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/hubbard_1x10agp
../clean.sh
printf "...running hubbard_1x10 agp\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/hubbard_1x14
../clean.sh
printf "...running hubbard_1x14\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
printf "...running hubbard_1x14 ci\n"
$MPICOMMAND $CIPATH > ci.out
python2 ../testEnergy.py 'ci' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/hubbard_1x22
../clean.sh
printf "...running hubbard_1x22\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/hubbard_1x50
../clean.sh
printf "...running hubbard_1x50\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/hubbard_1x6
../clean.sh
printf "...running hubbard_1x6\n"
$VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/hubbard_18_tilt/
../clean.sh
printf "...running hubbard_18_tilt uhf\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
printf "...running hubbard_18_tilt gfmc\n"
$MPICOMMAND $GFMCPATH > gfmc.out
python2 ../testEnergy.py 'gfmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

#cd $here/h6/
#../clean.sh
#printf "...running h6\n"
#$MPICOMMAND $VMCPATH > vmc.out
#python2 ../testEnergy.py 'vmc' $tol
#if [ $clean == 1 ]
#then
#    ../clean.sh
#fi

cd $here/h4_ghf_complex/
../clean.sh
printf "...running h4 ghf complex\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

#cd $here/h10sr/
#../clean.sh
#printf "...running h10 sr nondirect\n"
#$MPICOMMAND $NONDIRECTPATH > nondirect.out
#python2 ../testEnergy.py 'nondirect' $tol
#../clean.sh
#printf "...running h10 sr direct\n"
#$MPICOMMAND $DIRECTPATH > direct.out
#python2 ../testEnergy.py 'direct' $tol
#if [ $clean == 1 ]
#then
#    ../clean.sh
#fi

#cd $here/h10lm/
#../clean.sh
#printf "...running h10 lm nondirect\n"
#$MPICOMMAND $NONDIRECTPATH > nondirect.out
#python2 ../testEnergy.py 'nondirect' $tol
#../clean.sh
#printf "...running h10 lm direct\n"
#$MPICOMMAND $DIRECTPATH > direct.out
#python2 ../testEnergy.py 'direct' $tol
#if [ $clean == 1 ]
#then
#    ../clean.sh
#fi

cd $here/h4sr/
../clean.sh
printf "...running h4 sr nondirect\n"
$MPICOMMAND $NONDIRECTPATH > nondirect.out
python2 ../testEnergy.py 'nondirect' $tol
../clean.sh
printf "...running h4 sr direct\n"
$MPICOMMAND $DIRECTPATH > direct.out
python2 ../testEnergy.py 'direct' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/h4lm/
../clean.sh
printf "...running h4 lm nondirect\n"
$MPICOMMAND $NONDIRECTPATH > nondirect.out
python2 ../testEnergy.py 'nondirect' $tol
../clean.sh
printf "...running h4 lm direct\n"
$MPICOMMAND $DIRECTPATH > direct.out
python2 ../testEnergy.py 'direct' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi


cd $here/h4_pfaffian_complex/
../clean.sh
printf "...running h4 pfaffian complex\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

#cd $here/h10sr/
#../clean.sh
#printf "...running h10 sr\n"
#$MPICOMMAND $VMCPATH > vmc.out
#python2 ../testEnergy.py 'vmc' $tol
#if [ $clean == 1 ]
#then
#    ../clean.sh
#fi

cd $here/h10pfaff/
../clean.sh
printf "...running h10 pfaffian\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/h20/
../clean.sh
printf "...running h20\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/h20ghf/
../clean.sh
printf "...running h20 ghf\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/c2/
../clean.sh
printf "...running c2\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/nociBenzene/
../clean.sh
printf "...running nociBenzene\n"
$TRANSPATH > trans.out 2>/dev/null
python2 ../testEnergy.py 'trans' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi


cd $here/rBe/
../clean.sh
printf "...running rBe ghf\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/rBe_complex/
../clean.sh
printf "...running rBe complex ghf\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/rBe_lm/
../clean.sh
printf "...running rBe direct lm\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/rMg_pp/
../clean.sh
printf "...running rMg with pseudopotentials\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/rMg_pp_lm/
../clean.sh
printf "...running rMg with pseudopotentials and direct lm opt\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/rH2_nc/
../clean.sh
printf "...running rH2 with number counting jastrows and direct lm opt\n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/rH2_dmc/
../clean.sh
printf "...running rH2 \n"
$MPICOMMAND $VMCPATH > vmc.out
python2 ../testEnergy.py 'vmc' $tol
printf "...running rH2 dmc\n"
$MPICOMMAND $DMCPATH > dmc.out
python2 ../testEnergy.py 'dmc' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

cd $here/h6_ghf_trans/
../clean.sh
#printf "...running H6 transcorrelated\n"
#$MPICOMMAND $TRANSPATH > trans.out
#python2 ../testEnergy.py 'trans' $tol
if [ $clean == 1 ]
then
    ../clean.sh
fi

#cd $here/FCIQMC/He2
#../../clean.sh
#printf "...running FCIQMC/He2\n"
#$MPICOMMAND $FCIQMCPATH > fciqmc.out
#python2 ../../testEnergy.py 'fciqmc' $tol
#if [ $clean == 1 ]
#then
#    ../clean.sh
#fi
#
#cd $here/FCIQMC/Ne_plateau
#../../clean.sh
#printf "...running FCIQMC/Ne_plateau\n"
#$MPICOMMAND $FCIQMCPATH > fciqmc.out
#python2 ../../testEnergy.py 'fciqmc' $tol
#if [ $clean == 1 ]
#then
#    ../clean.sh
#fi

cd $here
