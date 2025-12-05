# Makefile for HPC Group Project

setup:
	pip install -r requirements.txt

test:
	python src/monte_carlo.py --samples 1000 --seed 42

run-pi:
	mpiexec -n 4 python src/monte_carlo.py --samples 10000000 --seed 42

run-options:
	mpiexec -n 4 python src/options.py --samples 10000000 --seed 42

clean:
	rm -f results/*.csv results/*.png