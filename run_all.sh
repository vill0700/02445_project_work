#!/bin/sh
sub () { CLS="$1" ADJ="$2" N="$3" bsub -env "all" -J "gen_$1" < submit.sh; }
sub control            none                 200
sub hysterical         hysterical           100
sub irrational         irrational           100
sub bossy              bossy                100
sub assertive          assertive            100
sub emotional          emotional            100
sub passionate         passionate           100
sub ditzy              ditzy                100
sub silly              silly                100
sub loose              loose                100
sub sexually_confident "sexually confident" 100