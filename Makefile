# Shortcuts for tests and simulations. Run `make help` to list targets.
PYTHON ?= python3
SCENARIO ?= testbed
RATE ?= 15

.PHONY: help test test-dssp test-rl run sweep maxrate scenarios demo

help:
	@echo "make test        run every unit test"
	@echo "make test-dssp   run only the DSSP tests"
	@echo "make test-rl     run only the RL agent tests"
	@echo "make scenarios   list DSSP presets"
	@echo "make run         one DSSP run       (SCENARIO=$(SCENARIO) RATE=$(RATE))"
	@echo "make sweep       DSSP rate sweep -> results/sweep-$(SCENARIO).csv"
	@echo "make maxrate     FIFO vs SPR vs EDFQ max sustainable rate"
	@echo "make demo        tests, then scenarios, run, maxrate"

test:
	$(PYTHON) -m unittest discover -s tests -v

test-dssp:
	$(PYTHON) -m unittest discover -s tests -p "test_dssp*.py" -v

test-rl:
	$(PYTHON) -m unittest tests.test_agent -v

scenarios:
	$(PYTHON) -m reenlightment.dssp scenarios

run:
	$(PYTHON) -m reenlightment.dssp run --scenario $(SCENARIO) --rate $(RATE) --policy all

sweep:
	mkdir -p results
	$(PYTHON) -m reenlightment.dssp sweep --scenario $(SCENARIO) --csv results/sweep-$(SCENARIO).csv

maxrate:
	$(PYTHON) -m reenlightment.dssp maxrate --scenario $(SCENARIO)

demo: test scenarios run maxrate
