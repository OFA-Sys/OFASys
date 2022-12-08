.PHONY: style test clean

check_dirs := ofa test

wheel:
	python setup.py bdist_wheel
	ossutil64 cp dist/ofasys-0.1.0-py3-none-any.whl oss://ofasys/pkg/ofasys-0.1.0-py3-none-any.whl
	ossutil64 set-acl oss://ofasys/pkg/ofasys-0.1.0-py3-none-any.whl public-read

style:
	flake8 $(check_dirs) \
		--max-line-length=89 \
		--ignore=E203,W503,F821,F811 \
		--exclude=ofa/metric/pyciderevalcap,ofa/model/taming

test:
	python -m unittest -v

clean:
	find . -type d -name __pycache__ -name "*.py[co]" -exec rm -r {} \+
