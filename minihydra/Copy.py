import glob
import os
import shutil

print('Copying. Make sure you have switched to the correct UnifiedML branch.')

UnifiedML = glob.glob('../../AGI.__init__/UnifiedML/*.py', recursive=True) + \
            glob.glob('../../AGI.__init__/UnifiedML/*/*.py', recursive=True) + \
            glob.glob('../../AGI.__init__/UnifiedML/*/*/*.py', recursive=True) + \
            glob.glob('../../AGI.__init__/UnifiedML/*/*/*/*.py', recursive=True) + \
            glob.glob('../../AGI.__init__/UnifiedML/*/*.yaml', recursive=True) + \
            glob.glob('../../AGI.__init__/UnifiedML/*/*/*.yaml', recursive=True) + \
            glob.glob('../../AGI.__init__/UnifiedML/*/*/*/*.yaml', recursive=True) + \
            glob.glob('../../AGI.__init__/UnifiedML/*.yml', recursive=True) + \
            glob.glob('../../AGI.__init__/UnifiedML/*.md', recursive=True)

Added = [file for file in glob.glob('../../UnifiedML/World/**/*.py', recursive=True) if 'Data/' not in file] + \
        glob.glob('../../UnifiedML/Hyperparams/minihydra.py', recursive=True)

for file in UnifiedML + Added:
    file = file.split('../..', 1)[1].split('/AGI.__init__', 1)[-1]
    source = '../..' + file
    destination = './src' + file.replace('UnifiedML', 'ML')
    print(source, destination)

    if os.path.exists(source):
        os.makedirs(destination.rsplit('/', 1)[0], exist_ok=True)

        shutil.copyfile(source, destination)


class Cow:
    def __init__(self):
        pass
