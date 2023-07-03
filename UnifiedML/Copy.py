import glob
import os
import shutil


def copy():
    print('Copying. Make sure you have switched to the correct UnifiedML branch.')

    root, _ = __file__.rsplit('/', 1)

    UnifiedML = glob.glob(f'{root}/../../AGI.__init__/UnifiedML/*.py', recursive=True) + \
                glob.glob(f'{root}/../../AGI.__init__/UnifiedML/*/*.py', recursive=True) + \
                glob.glob(f'{root}/../../AGI.__init__/UnifiedML/*/*/*.py', recursive=True) + \
                glob.glob(f'{root}/../../AGI.__init__/UnifiedML/*/*/*/*.py', recursive=True) + \
                glob.glob(f'{root}/../../AGI.__init__/UnifiedML/*/*.yaml', recursive=True) + \
                glob.glob(f'{root}/../../AGI.__init__/UnifiedML/*/*/*.yaml', recursive=True) + \
                glob.glob(f'{root}/../../AGI.__init__/UnifiedML/*/*/*/*.yaml', recursive=True) + \
                glob.glob(f'{root}/../../AGI.__init__/UnifiedML/*.yml', recursive=True) + \
                glob.glob(f'{root}/../../AGI.__init__/UnifiedML/*.md', recursive=True)

    Added = [file for file in glob.glob(f'{root}/../../UnifiedML/World/**/*.py', recursive=True) if 'Data/' not in file] + \
            glob.glob(f'{root}/../../UnifiedML/Hyperparams/minihydra.py', recursive=True)

    for file in UnifiedML + Added:
        file = file.split(f'{root}/../..', 1)[1].split('/AGI.__init__', 1)[-1]
        source = f'{root}/../..' + file
        destination = f'{root}/src' + file.replace('UnifiedML', 'ML')

        if os.path.exists(source):
            os.makedirs(destination.rsplit('/', 1)[0], exist_ok=True)

            shutil.copyfile(source, destination)


if __name__ == '__main__':
    copy()
