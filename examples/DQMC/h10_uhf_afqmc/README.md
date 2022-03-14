```
python -u hChain.py > hChain.out
```
The following block at the top of the script needs to be modified:
```
nproc = 10
dice_binary = "/projects/anma2640/relDice/Dice/ZDice2"
vmc_root = "/projects/anma2640/VMC/dqmc_uihf/VMC/"
```

Scripts used: QMCUtils.py, shci.py, fcidump_rel.py are all in the [scripts directory](https://github.com/sanshar/VMC/tree/uihf/scripts)
Code used for UHF HCI: https://github.com/xubwa/Dice/tree/uhf
Compiling Dice is very similar to compiling the DQMC binary, provide the correct paths at the top of the Makefile

https://github.com/sanshar/VMC/tree/uihf/scripts

