include common.mk

PROJECT_DIR := .

help-more:
	@echo "gear_doc - build all doc images"

.PHONY: gear_doc always

gear_doc:
	$(PYTHON) gear_doc.py --batch

always:

orrery/something_y_45_new.nc:	always
	python3 ./gears.py --algo=new -m 1.16336 -s 5 -t 32 -k 2 --clear_max_angle 5 --relief 1.1 --roots -1 -T cutter45.cfg orrery/something_y_45_new.nc

something: orrery/something_y_45_new.nc
	python3 ./render.py orrery/something_y_45_new.nc -azp -t 6
