# Makefile for HPC Group Project

setup:
	pip install -r env/requirements.txt

test:
	python src/monte_carlo.py --samples 1000

run-pi:
	mpiexec -n 4 python src/monte_carlo.py --samples 10000000

run-options:
	mpiexec -n 4 python src/options.py --samples 10000000

clean:
	rm -f results/*.csv results/*.png