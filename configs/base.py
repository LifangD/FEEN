from pathlib import Path

BASE_DIR = Path ('.')
SOURCE_DIR=Path('/home/dlf/pyprojects/InvestEventExtractor')

config = {
	'data_dir': SOURCE_DIR / 'data',
	'output': BASE_DIR / 'output',

}


