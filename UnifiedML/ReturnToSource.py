import glob
import os
import shutil


def copy():
    root, _ = __file__.rsplit('/', 1)

    UnifiedML = glob.glob(f'{root}/src/ML/*.py', recursive=True) + \
                glob.glob(f'{root}/src/ML/*/*.py', recursive=True) + \
                glob.glob(f'{root}/src/ML/*/*/*.py', recursive=True) + \
                glob.glob(f'{root}/src/ML/*/*/*/*.py', recursive=True) + \
                glob.glob(f'{root}/src/ML/*/*.yaml', recursive=True) + \
                glob.glob(f'{root}/src/ML/*/*/*.yaml', recursive=True) + \
                glob.glob(f'{root}/src/ML/*/*/*/*.yaml', recursive=True) + \
                glob.glob(f'{root}/src/ML/*.yml', recursive=True) + \
                glob.glob(f'{root}/src/ML/*.md', recursive=True)

    for file in UnifiedML:
        source = f'{root}/../../UnifiedML' + file.rsplit('/src/ML', 1)[1]

        os.makedirs(os.path.dirname(source), exist_ok=True)
        shutil.copyfile(file, source)


if __name__ == '__main__':
    # answer = input('Do you want to copy files from Builder into ../../UnifiedML? '
    #                'Unpushed changes may be lost. (y/n)')
    # if answer == 'y':
        copy()
