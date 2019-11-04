for i in 3 5 10 15 20 25 30 40 50 75 100; do
python -m sigtestv.run.sigtest_results -f data/bbase-sts-4e-05.tsv data/bbase-sts-5e-05.tsv -n $i -c pearson -it 1000 --test power > data/tests/bbase-sts-45-$i-pr.tsv &
python -m sigtestv.run.sigtest_results -f data/bbase-sts-3e-05.tsv data/bbase-sts-5e-05.tsv -n $i -c pearson -it 1000 --test power > data/tests/bbase-sts-35-$i-pr.tsv &
python -m sigtestv.run.sigtest_results -f data/bbase-sts-2e-05.tsv data/bbase-sts-5e-05.tsv -n $i -c pearson -it 1000 --test power > data/tests/bbase-sts-25-$i-pr.tsv &
python -m sigtestv.run.sigtest_results -f data/bbase-sts-3e-05.tsv data/bbase-sts-4e-05.tsv -n $i -c pearson -it 1000 --test power > data/tests/bbase-sts-34-$i-pr.tsv &
python -m sigtestv.run.sigtest_results -f data/bbase-sts-2e-05.tsv data/bbase-sts-4e-05.tsv -n $i -c pearson -it 1000 --test power > data/tests/bbase-sts-24-$i-pr.tsv &
python -m sigtestv.run.sigtest_results -f data/bbase-sts-2e-05.tsv data/bbase-sts-3e-05.tsv -n $i -c pearson -it 1000 --test power > data/tests/bbase-sts-23-$i-pr.tsv &
python -m sigtestv.run.sigtest_results -f data/bbase-sts-2e-05.tsv data/bbase-sts-3e-05.tsv -n $i -c pearson -it 1000 --test type1 > data/tests/bbase-sts-2-$i-t1.tsv &
python -m sigtestv.run.sigtest_results -f data/bbase-sts-3e-05.tsv data/bbase-sts-3e-05.tsv -n $i -c pearson -it 1000 --test type1 > data/tests/bbase-sts-3-$i-t1.tsv &
python -m sigtestv.run.sigtest_results -f data/bbase-sts-4e-05.tsv data/bbase-sts-3e-05.tsv -n $i -c pearson -it 1000 --test type1 > data/tests/bbase-sts-4-$i-t1.tsv &
python -m sigtestv.run.sigtest_results -f data/bbase-sts-5e-05.tsv data/bbase-sts-3e-05.tsv -n $i -c pearson -it 1000 --test type1 > data/tests/bbase-sts-5-$i-t1.tsv;
done




