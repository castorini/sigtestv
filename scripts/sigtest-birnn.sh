for i in 3 5 10 15 20 25 30 40 50 75 100; do
python -m sigtestv.run.sigtest_results -f data/birnn-sts-4e-04.tsv data/birnn-sts-5e-04.tsv -n $i -c pearsonr -it 1000 --test power > data/tests/birnn-sts-45-$i-pr.tsv &
python -m sigtestv.run.sigtest_results -f data/birnn-sts-3e-04.tsv data/birnn-sts-5e-04.tsv -n $i -c pearsonr -it 1000 --test power > data/tests/birnn-sts-35-$i-pr.tsv &
python -m sigtestv.run.sigtest_results -f data/birnn-sts-3e-04.tsv data/birnn-sts-4e-04.tsv -n $i -c pearsonr -it 1000 --test power > data/tests/birnn-sts-34-$i-pr.tsv &
python -m sigtestv.run.sigtest_results -f data/birnn-sts-3e-04.tsv data/birnn-sts-3e-04.tsv -n $i -c pearsonr -it 1000 --test type1 > data/tests/birnn-sts-3-$i-t1.tsv &
python -m sigtestv.run.sigtest_results -f data/birnn-sts-4e-04.tsv data/birnn-sts-3e-04.tsv -n $i -c pearsonr -it 1000 --test type1 > data/tests/birnn-sts-4-$i-t1.tsv &
python -m sigtestv.run.sigtest_results -f data/birnn-sts-5e-04.tsv data/birnn-sts-3e-04.tsv -n $i -c pearsonr -it 1000 --test type1 > data/tests/birnn-sts-5-$i-t1.tsv;
done




