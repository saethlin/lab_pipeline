set -e

for snapdir in h113_HR_sn1dy300ro100ss/snapdir*
do
    cd $snapdir
    python ~/lab_pipeline/pltcolt.py tot_SMC.h5 -o "snapshot_$(cut -f4 -d'_' <<< $PWD).png" --dpi=100
    cd ../..
done

